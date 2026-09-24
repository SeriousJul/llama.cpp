set -euo pipefail
DST=/home/seriousjul/src/llama.cpp/build-004/bin/libggml-cuda.so.0.25.0
BIN=/home/seriousjul/src/llama.cpp/build-004/bin/llama-bench
M9=$HOME/bench-fp8/qwen35-9b-iq4_xs.gguf
M27=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
A=/tmp/004/final-cuda.so;  AH=52e20b02f06114f14cb91049317fe640
G=/tmp/004/glu-fused.so;   GH=b56c071f5464b683e6c09fa94e6a268b
CSV=/tmp/004/glu.csv; : > "$CSV"
one() { cp "$2" "$DST"; got=$(md5sum "$DST"|cut -d' ' -f1); [ "$got" = "$3" ] || { echo MISMATCH; exit 1; }
  v=$(timeout 1200 "$BIN" -m "$4" -o json $5 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print('%.2f'%(d[0]['$6']*1e9/d[0]['avg_ns']))")
  echo "$1,$v,$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)" | tee -a "$CSV" >/dev/null; }
P9="-ngl 99 -p 4096 -n 0 -r 3 -b 2048 -ub 512"; TG="-ngl 99 -p 0 -n 256 -r 3 -b 2048 -ub 512"
P27="-ngl 99 -p 4096 -n 0 -r 2 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0"
for r in 1 2; do one p9A "$A" "$AH" "$M9" "$P9" n_prompt; one p9G "$G" "$GH" "$M9" "$P9" n_prompt; one p9G "$G" "$GH" "$M9" "$P9" n_prompt; one p9A "$A" "$AH" "$M9" "$P9" n_prompt
                 one tgA "$A" "$AH" "$M9" "$TG" n_gen; one tgG "$G" "$GH" "$M9" "$TG" n_gen; one tgG "$G" "$GH" "$M9" "$TG" n_gen; one tgA "$A" "$AH" "$M9" "$TG" n_gen; done
one w27 "$G" "$GH" "$M27" "$P27" n_prompt
for r in 1 2; do one p27A "$A" "$AH" "$M27" "$P27" n_prompt; one p27G "$G" "$GH" "$M27" "$P27" n_prompt; one p27G "$G" "$GH" "$M27" "$P27" n_prompt; one p27A "$A" "$AH" "$M27" "$P27" n_prompt; done
cp "$A" "$DST"
