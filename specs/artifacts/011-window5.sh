#!/usr/bin/env bash
# Window 5: 011 phase-1 A/B. HINT = build-011 (SetAccessedBy), CTRL = build-004. ABBA on the cliff row.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service
LOG=$OUT/window5.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

run() { # tag hints kv ctx prompt tmo
  local tag=$1 hints=$2 kv=$3 ctx=$4 pf=$5 tmo=$6
  local f="$OUT/w5-$tag.txt"
  say "w5 $tag start clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_UVM_HINTS="$hints" timeout "$tmo" "$B11" -m "$MODEL" -ngl 99 -fa on \
    -b 2048 -ub 512 -t 16 --temp 0 --no-display-prompt -fit off --ignore-eos -st \
    -c "$ctx" -ctk "$kv" -ctv "$kv" -n 128 -f "$pf" > "$f" 2>&1
  say "w5 $tag rc=$? $(grep -oE 'Prompt: *[0-9.]+ t/s \| Generation: *[0-9.]+ t/s' "$f" | tail -1)"
}

B11=$REPO/build-011/bin/llama-cli
P2=$OUT/prompt270k.txt
P1=$OUT/prompt260k.txt

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used md5=$(md5sum "$B11" | cut -d' ' -f1) lib=$(md5sum "$REPO/build-011/bin/libggml-cuda.so" | cut -d' ' -f1)"

run HINT-f16-209k  1 f16 262144 "$P2" 1800
run CTRL-f16-209k  0 f16 262144 "$P2" 1800
run CTRL-f16-209k-b 0 f16 262144 "$P2" 1800
run HINT-f16-209k-b 1 f16 262144 "$P2" 1800
run HINT-q8-118k   1 q8_0 160000 "$P1" 900
run CTRL-q8-118k   0 q8_0 160000 "$P1" 900
say "window closed"
