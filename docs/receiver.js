// A trained Arecibo Zero receiver running in the browser (plain JS, float32).
// Mirrors arecibo/receivers.py: token + position embeddings, 2 pre-LN blocks
// (causal multi-head attention, GELU MLP), final LayerNorm, linear head.
export class Receiver {
  constructor(json) {
    this.cfg = json;
    this.W = {};
    for (const [k, v] of Object.entries(json.weights)) {
      const bin = atob(v.b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      this.W[k] = { shape: v.shape, data: new Float32Array(buf.buffer) };
    }
  }
  static erf(x) { // Abramowitz & Stegun 7.1.26
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  ln(x, T, d, name) {
    const w = this.W[name + '.weight'].data, b = this.W[name + '.bias'].data, out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let m = 0, v = 0;
      for (let i = 0; i < d; i++) m += x[t * d + i]; m /= d;
      for (let i = 0; i < d; i++) { const z = x[t * d + i] - m; v += z * z; } v /= d;
      const r = 1 / Math.sqrt(v + 1e-5);
      for (let i = 0; i < d; i++) out[t * d + i] = (x[t * d + i] - m) * r * w[i] + b[i];
    }
    return out;
  }
  lin(x, T, name) { // torch layout: weight (out, in)
    const W = this.W[name + '.weight'], b = this.W[name + '.bias'].data;
    const [o, n] = W.shape, w = W.data, out = new Float32Array(T * o);
    for (let t = 0; t < T; t++) {
      const xo = t * n;
      for (let j = 0; j < o; j++) {
        let s = b[j]; const wo = j * n;
        for (let i = 0; i < n; i++) s += x[xo + i] * w[wo + i];
        out[t * o + j] = s;
      }
    }
    return out;
  }
  // returns logits at the last position
  forward(tokens) {
    const { d, heads, layers, ctx } = this.cfg;
    tokens = tokens.slice(-ctx);
    const T = tokens.length, dh = d / heads;
    const tok = this.W['tok.weight'].data, pos = this.W['pos.weight'].data;
    let h = new Float32Array(T * d);
    for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) h[t * d + i] = tok[tokens[t] * d + i] + pos[t * d + i];
    for (let l = 0; l < layers; l++) {
      const p = `blocks.${l}.`;
      const qkv = this.lin(this.ln(h, T, d, p + 'ln1'), T, p + 'qkv');
      const a = new Float32Array(T * d), sc = new Float32Array(T), scale = 1 / Math.sqrt(dh);
      for (let hd = 0; hd < heads; hd++) {
        for (let t = 0; t < T; t++) {
          let mx = -Infinity;
          for (let u = 0; u <= t; u++) {
            let s = 0;
            for (let j = 0; j < dh; j++) s += qkv[t * 3 * d + hd * dh + j] * qkv[u * 3 * d + d + hd * dh + j];
            sc[u] = s * scale; if (sc[u] > mx) mx = sc[u];
          }
          let z = 0;
          for (let u = 0; u <= t; u++) { sc[u] = Math.exp(sc[u] - mx); z += sc[u]; }
          for (let j = 0; j < dh; j++) {
            let s = 0;
            for (let u = 0; u <= t; u++) s += sc[u] * qkv[u * 3 * d + 2 * d + hd * dh + j];
            a[t * d + hd * dh + j] = s / z;
          }
        }
      }
      const pr = this.lin(a, T, p + 'proj');
      for (let i = 0; i < h.length; i++) h[i] += pr[i];
      const m1 = this.lin(this.ln(h, T, d, p + 'ln2'), T, p + 'mlp.0');
      for (let i = 0; i < m1.length; i++) m1[i] = 0.5 * m1[i] * (1 + Receiver.erf(m1[i] / Math.SQRT2));
      const m2 = this.lin(m1, T, p + 'mlp.2');
      for (let i = 0; i < h.length; i++) h[i] += m2[i];
    }
    const last = this.ln(h.slice((T - 1) * d), 1, d, 'ln');
    return this.lin(last, 1, 'head');
  }
  // greedy continuation until a stop symbol; returns [{id, p}]
  generate(tokens, stopIds, maxNew = 8) {
    const out = [];
    let seq = tokens.slice();
    for (let k = 0; k < maxNew; k++) {
      const lg = this.forward(seq);
      let best = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[best]) best = i;
      let mx = lg[best], z = 0; for (const v of lg) z += Math.exp(v - mx);
      out.push({ id: best, p: 1 / z });
      if (stopIds.includes(best)) break;
      seq.push(best);
    }
    return out;
  }
}
