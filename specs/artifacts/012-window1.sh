#!/usr/bin/env bash
# Window 1: split 012's "unaccounted" band into CUDA graph bytes vs the rest.
# A/B is GGML_CUDA_DISABLE_GRAPHS (off switch lives in common.cuh:1289 is_enabled()).
# Direct read is GGML_CUDA_GRAPH_PROBE (build-012 only, specs/artifacts/012-graph-probe.patch).
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
SRV=$REPO/build-012/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
SRC=$REPO/specs/artifacts/011-sweep
OUT=$REPO/specs/artifacts/012-sweep
PORT=36100
SVC=llama-server.service
LOG=$OUT/window1.log; : > "$LOG"
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

P=$(python3 -c 'import json;print(json.dumps(open("'"$SRC"'/012-prompt.txt").read()))')
PT=$("$REPO/build-004/bin/llama-tokenize" -m "$MODEL" --stdin --show-count < "$SRC/012-prompt.txt" 2>&1 | grep -oE "[0-9]+$")

run() { # tag graphs batch n_predict ctx
  local tag=$1 graphs=$2 b=$3 n=$4 ctx=${5:-160000}
  local f="$OUT/w1-$tag.log"
  local enva=(GGML_CUDA_GRAPH_PROBE=1)
  [[ $graphs == OFF ]] && enva=(GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GRAPH_PROBE=1)
  say "run $tag (graphs=$graphs b=$b n=$n) clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  setsid env "${enva[@]}" "$SRV" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$ctx" --batch-size "$b" --ubatch-size 512 --n-gpu-layers all \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --no-kv-unified --fit off --threads 16 --threads-batch 16 -v > "$f" 2>&1 &
  local sp=$!
  for _ in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  ( for _ in $(seq 200); do echo "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 1; done > "$OUT/w1-$tag.sm" ) &
  local sampler=$!
  curl -s "http://127.0.0.1:$PORT/completion" \
    -d "{\"prompt\": $P, \"n_predict\": $n, \"temperature\": 0, \"ignore_eos\": true}" > "$OUT/w1-$tag.req"
  kill "$sampler" 2>/dev/null
  say "  used_peak=$(sort -n "$OUT/w1-$tag.sm" | tail -1) MiB"
  kill -TERM -- -"$sp" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$sp" 2>/dev/null || break; sleep 1; done
  local mb=$(grep "CUDA0" "$f" | grep "=" | tail -1)
  local t=$(grep -E "prompt eval time|eval time|graphs reused" "$f" | tail -3 | tr '\n' '|')
  say "  $tag breakdown: $mb"
  say "  $tag timing: $t"
  grep 'graph-probe' "$f" > "$OUT/w1-$tag.probe"
  say "  $tag captures=$(grep -c 'post instantiate' "$f")"
  python3 - "$OUT/w1-$tag.probe" <<'PY' || true
import re, sys
rows = []
for line in open(sys.argv[1]):
    m = re.search(r'graph-probe: (\w+(?: \w+)?) free =\s+([\d.]+) MiB, captures = (\d+)', line)
    if m:
        rows.append((m.group(1), float(m.group(2)), int(m.group(3))))
seq, prev = [], None
for where, free, cap in rows:
    if where == 'pre capture':
        prev = free
    elif where == 'post instantiate' and prev is not None:
        seq.append(round(prev - free, 1)); prev = None
print(f"  probe: n_graphs={len(seq)} held_per_graph_MiB={seq}")
PY
}

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used lib=$(md5sum $REPO/build-012/bin/libggml-cuda.so | cut -c1-12) prompt_tokens=$PT"

run b2048-ON-n8    ON  2048 8
run b2048-OFF-n8   OFF 2048 8
run b2048-OFF-n1024 OFF 2048 1024
run b2048-ON-n1024 ON  2048 1024
run b512-ON-n8     ON  512  8
run b512-OFF-n8    OFF 512  8
run b512-OFF-n1024 OFF 512  1024
run b512-ON-n1024  ON  512  1024
run c262k-ON-n8    ON  2048 8 262144
run c262k-OFF-n8   OFF 2048 8 262144

say "window closed"
