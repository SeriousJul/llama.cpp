#!/usr/bin/env bash
# Window 7: 011 phase-2b lookahead prefetch. Capture-legality probe, then cliff ABBA + fit row.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service
B11=$REPO/build-011/bin/llama-cli
LOG=$OUT/window9.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

run() { # tag prefetch kv ctx prompt tmo
  local tag=$1 pf=$2 kv=$3 ctx=$4 pf_file=$5 tmo=$6
  local f="$OUT/w9-$tag.txt"
  say "w7 $tag start clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_UVM_PREFETCH="$pf" timeout "$tmo" "$B11" -m "$MODEL" -ngl 99 -fa on \
    -b 2048 -ub 512 -t 16 --temp 0 --no-display-prompt -fit off --ignore-eos -st \
    -c "$ctx" -ctk "$kv" -ctv "$kv" -n 128 -f "$pf_file" > "$f" 2>&1
  say "w7 $tag rc=$? $(grep -oE 'Prompt: *[0-9.]+ t/s \| Generation: *[0-9.]+ t/s' "$f" | tail -1)"
}

P2=$OUT/prompt270k.txt
P1=$OUT/prompt260k.txt

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used lib=$(md5sum $REPO/build-011/bin/libggml-cuda.so | cut -c1-12)"

say "capture-legality probe"
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 timeout 300 "$B11" -m "$MODEL" -ngl 99 -fa on -c 8192 -ctk q8_0 -ctv q8_0 \
  -b 512 -ub 512 -t 16 --temp 0 -fit off -st -n 32 -p "Question: what is 17 times 23? Answer:" > "$OUT/w9-probe.txt" 2>&1
say "probe rc=$? answer=$(grep -A1 'Answer:' "$OUT/w9-probe.txt" | tail -1 | tr -d '\r')"

run PREF-f16-a 1 f16 262144 "$P2" 1200
run CTRL-f16-a 0 f16 262144 "$P2" 1200
run CTRL-f16-b 0 f16 262144 "$P2" 1200
run PREF-f16-b 1 f16 262144 "$P2" 1200
run PREF-q8-a  1 q8_0 160000 "$P1" 600
run CTRL-q8-a  0 q8_0 160000 "$P1" 600
say "window closed"
