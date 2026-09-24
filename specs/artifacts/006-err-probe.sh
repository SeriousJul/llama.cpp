#!/usr/bin/env bash
# specs/006 stage 3: measure the chunked path's per-case error at the op seam.
#
# The harness only prints an error when a case fails, and it prints it with no newline beside a
# case name that appears twice, so pairs cannot be trusted from a full-suite log. This script runs
# one case per process and reads exactly one ERR line. To force the print it makes the tolerance
# negative for the duration of the measurement, and to prove which build a number came from it
# prints the launcher's dynamic smem request, which identifies the tile layout. Everything is
# restored at the end and the tree is rebuilt.
#
# Usage: specs/artifacts/006-err-probe.sh [C ...]     (default: 16 and 64)
set -uo pipefail

REPO=${REPO:-/home/seriousjul/src/llama.cpp}
BUILD=${BUILD:-$REPO/build-004}
SRC=$REPO/ggml/src/ggml-cuda/gated_delta_net.cu
TST=$REPO/tests/test-backend-ops.cpp
WORK=/tmp/006-err-probe

cd "$REPO" || exit 1
mkdir -p "$WORK"
cp "$SRC" "$WORK/kernel.keep"
cp "$TST" "$WORK/tests.keep"

restore() {
    cp "$WORK/kernel.keep" "$SRC"
    cp "$WORK/tests.keep"  "$TST"
    cmake --build "$BUILD" --target ggml-cuda test-backend-ops -j 24 > /dev/null 2>&1
    echo "(tree restored and rebuilt)"
}
trap restore EXIT INT TERM

CASES=(
  "head_size=32,n_seq_tokens=64,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=64,n_seq_tokens=64,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=64,n_seq_tokens=65,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=64,n_seq_tokens=100,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=64,n_seq_tokens=127,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=64,n_seq_tokens=200,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=64,n_seq_tokens=256,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=128,n_seq_tokens=64,n_seqs=1,v_repeat=2,permuted=0,kda=0,K=1"
  "head_size=128,n_seq_tokens=65,n_seqs=1,v_repeat=2,permuted=0,kda=0,K=1"
  "head_size=128,n_seq_tokens=100,n_seqs=2,v_repeat=1,permuted=0,kda=0,K=1"
  "head_size=128,n_seq_tokens=512,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
)

python3 - "$TST" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "struct test_gated_delta_net : public test_case {"
assert a in s
open(p, "w").write(s.replace(a, a + "\n    double max_nmse_err() override { return -1.0; }\n", 1))
PY

python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "    const int    id    = ggml_cuda_get_device();\n    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;\n"
assert a in s
s = s.replace(a, a + "    fprintf(stderr, \"PROBE nbytes %zu\\n\", nbytes_shared);\n", 1)
open(p, "w").write(s)
PY

CS=("$@")
[ ${#CS[@]} -eq 0 ] && CS=(16 64)
for c in "${CS[@]}"; do
    sed -i -E "s/^#define GGML_CUDA_GDN_CHUNK_C [0-9]+$/#define GGML_CUDA_GDN_CHUNK_C $c/" "$SRC"
    sed -i -E "s/^#define GGML_CUDA_GDN_CHUNKED [0-9]+\$/#define GGML_CUDA_GDN_CHUNKED 1/" "$SRC"
    if ! cmake --build "$BUILD" --target ggml-cuda test-backend-ops -j 24 > "$WORK/build.log" 2>&1; then
        echo "C=$c: build failed"; grep -iE "error" "$WORK/build.log" | head -3; continue
    fi
    nb=$(timeout 300 "$BUILD/bin/test-backend-ops" -o GATED_DELTA_NET -b CUDA0 \
         -p ".*head_size=128,n_seq_tokens=512.*kda=0,K=1" 2>&1 | grep -m1 -oE "PROBE nbytes [0-9]+" | awk '{print $3}')
    echo "=== C=$c   d=128 smem request: ${nb:-?} B"
    worst=0
    for cs in "${CASES[@]}"; do
        e=$(timeout 300 "$BUILD/bin/test-backend-ops" -o GATED_DELTA_NET -b CUDA0 -p ".*$cs" 2>&1 \
            | grep -m1 -oE "GATED_DELTA_NET\] ERR = [0-9.e+-]+" | awk '{print $NF}')
        [ -z "$e" ] && e=0
        printf "    %-52s ERR=%-12s %s\n" "$(echo "$cs" | cut -d, -f1-2)" "$e" \
            "$(python3 -c "print('OVER' if $e > 1e-7 else '')")"
        worst=$(python3 -c "print(max($worst, $e))")
    done
    rep=$(timeout 300 "$BUILD/bin/test-backend-ops" -o GATED_DELTA_NET -b CUDA0 \
          -p ".*head_size=128,n_seq_tokens=512.*kda=0,K=1" 2>&1 | grep -m1 -oE "GATED_DELTA_NET\] ERR = [0-9.e+-]+" | awk '{print $NF}')
    echo "    worst: $worst   (seam default 1e-7)"
    echo "    determinism control, same case re-run: ERR=$rep"
done
