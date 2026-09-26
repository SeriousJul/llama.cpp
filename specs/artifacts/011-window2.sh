#!/usr/bin/env bash
# Window 2: 011 overflow-regime probes + 012 fixed phase capture.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
CLI=$REPO/build-004/bin/llama-cli
SRV=$REPO/build-004/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service
mkdir -p "$OUT"
LOG=$OUT/window2.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

# ~260K token filler (~1.1 MB, filler repeats ~4.3 tok/repetition)
python3 -c "print(('the quick brown fox jumps over the lazy dog while sensors record the ambient telemetry of the plant floor ' * 6200), end='')" > "$OUT/prompt260k.txt"
wc -c "$OUT/prompt260k.txt" >> "$LOG"

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open, gpu used=$used"

COMMON=(-m "$MODEL" -ngl 99 -fa on -b 2048 -ub 512 -t 16 --temp 0 --no-display-prompt -fit off --ignore-eos)

for cfg in "OFF f16 262144" "ON f16 262144" "OFF f16 163840" "ON f16 163840" "ON q8_0 262144"; do
  read -r mode kv ctx <<< "$cfg"
  enva=(); [[ $mode == ON ]] && enva=(GGML_CUDA_ENABLE_UNIFIED_MEMORY=1) || enva=(-u GGML_CUDA_ENABLE_UNIFIED_MEMORY)
  say "cli $mode $kv ctx=$ctx start"
  env "${enva[@]}" "$CLI" "${COMMON[@]}" -c "$ctx" -ctk "$kv" -ctv "$kv" -n 128 \
    -f "$OUT/prompt260k.txt" > "$OUT/cli-$mode-$kv-$ctx.txt" 2>&1
  rc=$?
  grep -E "prompt eval time|eval time|total time|tokens requested|print_info:.*n_ctx|parse_single: prompt_file = " "$OUT/cli-$mode-$kv-$ctx.txt" >> "$LOG" 2>/dev/null
  grep -E "get_default_params|llama_init|ERROR|error:" "$OUT/cli-$mode-$kv-$ctx.txt" | tail -3 >> "$LOG"
  say "cli $mode $kv ctx=$ctx rc=$rc done"
done

say "012 capture (ignore_eos, n_predict 2048, 1 Hz sampling)"
setsid "$SRV" --model "$MODEL" --host 127.0.0.1 --port 36099 \
  --ctx-size 160000 --batch-size 2048 --ubatch-size 512 --n-gpu-layers all \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --no-kv-unified --fit off --threads 16 --threads-batch 16 --ignore-eos -v \
  > "$OUT/012-server.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 120); do curl -sf http://127.0.0.1:36099/health >/dev/null && break; sleep 2; done
say "srv up, used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
P=$(python3 -c 'import json;print(json.dumps(open("'"$OUT"'/012-prompt.txt").read()))')
( for _ in $(seq 90); do echo "$(date +%s.%N) $(nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader,nounits)"; sleep 1; done > "$OUT/012-samples.txt" ) &
SAMPLER=$!
curl -s "http://127.0.0.1:36099/completion" -d "{\"prompt\": $P, \"n_predict\": 2048, \"temperature\": 0, \"ignore_eos\": true}" > "$OUT/012-req2.json"
wait "$SAMPLER" 2>/dev/null; kill "$SAMPLER" 2>/dev/null
kill -TERM -- -"$SRV_PID" 2>/dev/null
for _ in $(seq 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
say "window closed"
