#!/usr/bin/env bash
# specs/006 stage 3: paired A/B/B/A-style comparison of the chunked configurations.
#
# Absolute t/s drifts with the SM clock on this box, so a config is only ever compared to the
# baseline inside the same round, the baseline brackets the variants on both sides, and the library
# on disk is md5-verified before every run. Clock, temperature and power are sampled per run so a
# drift shows up in the record instead of in the conclusion.
#
# Usage: specs/artifacts/006-ab.sh [rounds]        (default 3)
set -euo pipefail

SRC=${SRC:-/home/seriousjul/src/llama.cpp}
DST=$SRC/build-004/bin/libggml-cuda.so.0.25.0
BIN=$SRC/build-004/bin/llama-bench
MODEL=$HOME/bench-fp8/qwen35-9b-iq4_xs.gguf
LIBS=${LIBS:-/tmp/006-ab}
ROUNDS=${1:-3}

declare -A HASH
for tag in ${TAGS:-A B C}; do
    H=$(md5sum "$LIBS/$tag/libggml-cuda.so.0.25.0" | cut -d' ' -f1)
    HASH[$tag]=$H
    echo "$tag = $H"
done
echo "A = flag off (recurrent kernel), B = chunked C=16, C = chunked C=64"
echo

run() {  # $1 = tag
    local tag=$1 lib="$LIBS/$1/libggml-cuda.so.0.25.0" want=${HASH[$1]}
    cp "$lib" "$DST"
    local got
    got=$(md5sum "$DST" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then echo "MD5 MISMATCH for $tag: $got != $want"; exit 1; fi

    local out pp ppf tg tgf
    out=$(timeout 600 "$BIN" -m "$MODEL" -ngl 99 -p 4096 -n 256 -r 2 -b 2048 -ub 512 -o json 2>/dev/null)
    pp=$(echo "$out" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for e in d:
    if e['n_prompt'] and not e.get('n_gen'): print('%.1f' % (e['n_prompt']*1e9/e['avg_ns']))
" | head -1)
    tg=$(echo "$out" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for e in d:
    if e.get('n_gen') and not e['n_prompt']: print('%.2f' % (e['n_gen']*1e9/e['avg_ns']))
" | head -1)
    local clk
    clk=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits | tr ',' ' ')
    printf '  %s  pp4096=%8s t/s   tg256=%6s t/s   sm=%s MHz t=%s C p=%s W\n' "$tag" "$pp" "$tg" $clk
}

for r in $(seq 1 "$ROUNDS"); do
    echo "--- round $r ---"
    for tag in ${TAGS:-A B C C B A}; do run "$tag"; done
done

# leave the tree on the baseline library, which is the flag-off build
cp "$LIBS/A/libggml-cuda.so.0.25.0" "$DST"
echo "restored A: $(md5sum "$DST" | cut -d' ' -f1)"
