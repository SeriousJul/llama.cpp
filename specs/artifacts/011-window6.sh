#!/usr/bin/env bash
# Window 6: 011 phase-2. Correctness gate under UVM, then cliff ABBA (prefetch on/off).
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service
B11=$REPO/build-011/bin/llama-cli
OPS=$REPO/build-011/bin/test-backend-ops
LOG=$OUT/window6.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

run() { # tag prefetch kv ctx prompt tmo
  local tag=$1 pf=$2 kv=$3 ctx=$4 pf_file=$5 tmo=$6
  local f="$OUT/w6-$tag.txt"
  say "w6 $tag start clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_UVM_PREFETCH="$pf" timeout "$tmo" "$B11" -m "$MODEL" -ngl 99 -fa on \
    -b 2048 -ub 512 -t 16 --temp 0 --no-display-prompt -fit off --ignore-eos -st \
    -c "$ctx" -ctk "$kv" -ctv "$kv" -n 128 -f "$pf_file" > "$f" 2>&1
  say "w6 $tag rc=$? $(grep -oE 'Prompt: *[0-9.]+ t/s \| Generation: *[0-9.]+ t/s' "$f" | tail -1)"
}

P2=$OUT/prompt270k.txt
P1=$OUT/prompt260k.txt

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used lib=$(md5sum $REPO/build-011/bin/libggml-cuda.so | cut -c1-12)"

say "op suite (UVM + prefetch)"
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 timeout 900 "$OPS" -b CUDA0 > "$OUT/w6-ops-uvm.log" 2>&1
rc=$?
say "ops rc=$rc pass=$(grep -c '^pass' "$OUT/w6-ops-uvm.log") fail=$(grep -c '^FAIL' "$OUT/w6-ops-uvm.log")"
if (( rc != 0 && rc != 124 )); then
  grep '^FAIL' "$OUT/w6-ops-uvm.log" | head -5 | tee -a "$LOG"
fi

run PREF-f16-a 1 f16 262144 "$P2" 1500
run CTRL-f16-a 0 f16 262144 "$P2" 1500
run CTRL-f16-b 0 f16 262144 "$P2" 1500
run PREF-f16-b 1 f16 262144 "$P2" 1500
run PREF-q8-a  1 q8_0 160000 "$P1" 600
run CTRL-q8-a  0 q8_0 160000 "$P1" 600
say "window closed"
