#!/bin/bash
# Runs every experiment in order (resumable: finished runs are skipped).
set -e
cd "$(dirname "$0")/experiments"
PY=${PY:-python}
$PY radix.py
BASE=10 LSB=1 $PY content.py
BASE=10 LSB=1 $PY channel.py
$PY decipher.py
ENS_K=10 $PY evolve.py tf
$PY evolve.py lstm
$PY evolve.py transfer
$PY figures.py
