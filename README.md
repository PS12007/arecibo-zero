# Arecibo Zero

**Can a mind with no shared language and no built-in knowledge learn arithmetic from a message alone?**

Arecibo Zero simulates first contact. A sender writes a transmission: numbers in some radix and digit order, plus sums, products, comparisons and counting. The receivers are randomly initialised neural networks (transformers and LSTMs of several sizes and seeds). They see nothing but that transmission. Afterwards each receiver sits an exam of sums that never appeared in the message.

Questions:

1. **Radix.** Which way of writing numbers is understood with the fewest transmitted bits?
2. **Content.** At a fixed budget, does teaching other mathematics (counting, comparison, products) help a receiver learn addition?
3. **Channel.** Raw bits with unknown symbol width plus random bit flips: is repetition coding worth its bits, or is arithmetic its own error-correcting code?
4. **Blind decipherment.** We receive a message in an unknown design. Can we recover the alphabet, digit values, digit order and operator meanings by training a receiver on it and *reading that receiver*?
5. **Evolved messages.** A genetic search designs messages for a population of receivers. Does a message tuned for one kind of mind work for another?

Work in progress; results, report and interactive demo coming.

```
arecibo/      message generator, receivers, batched trainer, bit channel, decipherment
experiments/  one script per question, results to results/*.jsonl
run_all.sh    everything, resumable
```
