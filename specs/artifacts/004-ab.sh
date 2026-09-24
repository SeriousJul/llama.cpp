#!/bin/bash
# paired A/B with an ABBA order per round, to separate the change from clock drift
set -euo pipefail
SRC=/home/seriousjul/src/llama.cpp
DST=$SRC/build-004/bin/libggml-cuda.so.0.25.0
BIN=$SRC/build-004/bin/llama-bench
A=/tmp/004/A/libggml-cuda.so.0.25.0
AHASH=cc79ddf3e6dcf30c27e6cab1500b8e7f
BHASH=99f765e0484c2104e32b736bc7c64638
MODEL=$HOME/bench-fp8/qwen35-9b-iq4_xs.gguf

run() {
  local tag=$1 lib=$2 want=$3
  cp "$lib" "$DST"
  local got
  got=$(md5sum "$DST" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then echo "MD5 MISMATCH for $tag: $got != $want"; exit 1; fi
  local ts clk
  ts=$(timeout 300 "$BIN" -m "$MODEL" -o json -ngl 99 -p 4096 -n 0 -r 3 -b 2048 -ub 512 2>/dev/null |
       python3 -c "import json,sys; d=json.load(sys.stdin); print('%.1f' % (d[0]['n_prompt']*1e9/d[0]['avg_ns']))")
  clk=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits)
  printf '%s  %8s t/s   sm=%s MHz  t=%s C  p=%s W\n' "$tag" "$ts" $(echo "$clk" | tr ',' ' ')
}

for round in 1 2 3; do
  echo "--- round $round ---"
  run "A-base" "$A" "$AHASH"
  run "B-pref" "/tmp/004/new-cuda.so" "$BHASH"
  run "B-pref" "/tmp/004/new-cuda.so" "$BHASH"
  run "A-base" "$A" "$AHASH"
done

# leave the tree in the patched state
cp /tmp/004/new-cuda.so "$DST"
echo "restored B: $(md5sum "$DST" | cut -d' ' -f1)"
