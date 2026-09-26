#!/usr/bin/env bash
# 011 + 012 baselines, one exclusive GPU window.
# Order per ctx: OFF ON ON OFF (paired ABBA, README rule). md5-gated, clocks sampled.
set -uo pipefail

REPO=/home/seriousjul/src/llama.cpp
BENCH=$REPO/build-004/bin/llama-bench
SRV=$REPO/build-004/bin/llama-server
LIBCUDA=$REPO/build-004/bin/libggml-cuda.so
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
CTXS=(32768 65536 131072 160000)
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service

mkdir -p "$OUT"
CSV=$OUT/baseline.csv
LOG=$OUT/run.log
: > "$CSV"; : > "$LOG"
echo "mode,ctx,pp_ts,tg_ts,clocks_sm,mkvib_used,pp_s,tg_s" >> "$CSV"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

restart_svc() {
  systemctl --user start "$SVC" && say "service restarted (trap)"
}
trap restart_svc EXIT INT TERM

md5_gate() {
  local want=3afb8f9217fc378043b65297587776ad
  local got; got=$(md5sum "$BENCH" | cut -d' ' -f1)
  if [[ $got != $want ]]; then say "FATAL bench md5 $got != $want"; exit 1; fi
}

gpu_free() {
  for _ in $(seq 20); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    (( used < 1000 )) && return 0
    sleep 2
  done
  say "FATAL GPU still busy"; return 1
}

clocks() { nvidia-smi --query-gpu=clocks.sm,memory.used --format=csv,noheader,nounits; }

row() { # mode ctx csvline
  local mode=$1 ctx=$2 out=$3
  local pp tg clk mk
  pp=$(echo "$out" | awk -F, '$2=="pp" {print $7" "$8}')
  tg=$(echo "$out" | awk -F, '$2=="tg" {print $7" "$8}')
  clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)
  mk=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$mode,$ctx,${pp:-NA},${tg:-NA},$clk,$mk,${out##*$'\n'}" >> "$CSV"
}

say "stopping $SVC"
systemctl --user stop "$SVC"
gpu_free || exit 1
say "GPU free, window open"
md5_gate

for ctx in "${CTXS[@]}"; do
  for mode in OFF ON ON OFF; do
    env_arg=()
    [[ $mode == ON ]] && env_arg=(GGML_CUDA_ENABLE_UNIFIED_MEMORY=1)
    [[ $mode == OFF ]] && env_arg=(-u GGML_CUDA_ENABLE_UNIFIED_MEMORY)
    say "bench $mode ctx=$ctx start clocks=$(clocks)"
    out=$(env "${env_arg[@]}" "$BENCH" -m "$MODEL" -ngl 99 -fa on \
      -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -r 1 -pg "$ctx",256 -o csv 2>>"$LOG")
    echo "==== $mode $ctx ====" >> "$LOG"; echo "$out" >> "$LOG"
    say "bench $mode ctx=$ctx done clocks=$(clocks)"
  done
done

say "011 matrix done; starting 012 capture"
PORT=36099
setsid "$SRV" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --ctx-size 160000 --batch-size 2048 --ubatch-size 512 --n-gpu-layers all \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --no-kv-unified --fit off --threads 16 --threads-batch 16 \
  --no-context-shift -v > "$OUT/012-server.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 2
done
say "server up, mem after load $(clocks)"
# ~4K token synthetic prompt
python3 - <<'EOF' > "$OUT/012-prompt.txt"
print(("the quick brown fox jumps over the lazy dog while sensors record the ambient telemetry of the plant floor " * 1) * 70)
EOF
say "sampling baseline memory, then pp4096+200"
curl -s "http://127.0.0.1:$PORT/completion" \
  -d "{\"prompt\": $(python3 -c 'import json;print(json.dumps(open("'"$OUT"'/012-prompt.txt").read()))'), \"n_predict\": 200, \"temperature\": 0}" \
  > "$OUT/012-req.json"
say "after request, mem $(clocks)"
# 200 more decode tokens to sit in the decode phase
curl -s "http://127.0.0.1:$PORT/completion" \
  -d "{\"prompt\": $(python3 -c 'import json;print(json.dumps(open("'"$OUT"'/012-prompt.txt").read()))'), \"n_predict\": 256, \"temperature\": 0}" \
  > /dev/null
sleep 5
say "idle decode phase, mem $(clocks)"
kill -TERM -- -"$SRV_PID" 2>/dev/null
for _ in $(seq 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
say "012 server stopped; window closing"
# trap restarts the service
