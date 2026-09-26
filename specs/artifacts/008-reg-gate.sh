#!/usr/bin/env bash
# 008 register gate. Asks one question for one shape: how many registers does a CTA that owns a
# slice of DV need, and what does it pay in local memory to force it into the budget that shape's
# launch bounds imply. Answers acceptance criterion 1 of specs/008 before any grid or fixup
# plumbing is written. Nothing links and nothing runs; it costs one file compile per shape.
#
#   bash specs/artifacts/008-reg-gate.sh [DV] [nthreads] [occupancy] [ncols ...]
#     defaults: 128 256 2, ncols 32 and 64
#   bash specs/artifacts/008-reg-gate.sh 64 256 2 64        # a quarter DV at the production tile
#
# Give it one ncols per call: an illegal (nthreads, ncols) combination fails the whole compile,
# and taking several per call then loses the shapes that would have compiled.
#
# Each requested (DV, ncols) gets exactly one config row: two rows on the same
# (DKQ, DV, ncols) key make the first one win silently, which is how an earlier version of this
# probe reported registers for a shape it never compiled. The instantiation TU is generated into
# /tmp rather than kept in the tree, so it cannot drift from the rows it is testing.
#
# Needs build-008 configured CUDA-only on arch 89:
#   cmake -B build-008 -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
#         -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
set -euo pipefail
REPO=/home/seriousjul/src/llama.cpp
BD=$REPO/build-008/ggml/src/ggml-cuda
CU=$REPO/ggml/src/ggml-cuda/fattn-mma-f16.cuh
DV=${1:-128}
NTHREADS=${2:-256}
OCCUPANCY=${3:-2}
shift $(( $# >= 3 ? 3 : $# )) || true
NCOLS_LIST=("$@")
[ ${#NCOLS_LIST[@]} -eq 0 ] && NCOLS_LIST=(32 64)
OUT=/tmp/008/probe-DV${DV}-${NTHREADS}x${OCCUPANCY}.o
TU=/tmp/008/probe-DV${DV}-${NTHREADS}x${OCCUPANCY}.cu
DEF=/tmp/008/probe.defines

mkdir -p /tmp/008
git -C "$REPO" checkout -- "$CU"
trap 'git -C "$REPO" checkout -- "$CU"' EXIT

python3 - "$CU" "$TU" "$DV" "$NTHREADS" "$OCCUPANCY" "${NCOLS_LIST[@]}" <<'PY'
import sys
cu, tu, dv, nthreads, occupancy = sys.argv[1:6]
ncols_list = [int(n) for n in sys.argv[6:]]
src = open(cu).read()

# nbatch_fa has to feed every warp in the KQ_C loops (static_assert "bad loop size", "zero-sized
# variable KQ_C"). nbatch_V2 and nbatch_combine are half2 counts along DV.
rows = "\n    // 008 probe rows, one per (DV, ncols) key.\n"
for ncols in ncols_list:
    # keep the shipping tile knobs and change only what is under test: nbatch_fa 32 at every
    # width the shipping rows use it at. The one exception is real, not a convenience: a 32-column
    # tile cannot feed 8 warps at nbatch_fa 32, so narrow + 256 threads has to widen the batch.
    nbf = 64 if (ncols < 64 and int(nthreads) > 128) else 32
    rows += ("    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, %3s, %2s, %3s, %s, %3s, 128, %3s, %3s, 2, true);\n"
             % (dv, ncols, nthreads, occupancy, nbf, int(dv) // 2, int(dv) // 2))
anchor = "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,"
assert anchor in src
open(cu, "w").write(src.replace(anchor, rows + anchor, 1))

decls = "#include \"fattn-mma-f16.cuh\"\n"
for ncols in ncols_list:
    # any (ncols1, ncols2) that multiplies to ncols compiles the same register shape
    n2 = 8 if ncols % 8 == 0 else 4
    decls += "DECL_FATTN_MMA_F16_CASE(256, %s, %2s, %s);\n" % (dv, ncols // n2, n2)
open(tu, "w").write(decls)
PY

# Codegen-relevant subset of build-008's CUDA_FLAGS; the trailing -Xcompiler warning list does not
# affect register allocation.
FLAGS=(-O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]"
       -Xcompiler=-fPIC -use_fast_math -extended-lambda)

# One define per line. shlex, because CUDA_DEFINES contains -DGGML_CUDA_FA_QUANTS="a,b,c" and a
# word split would read those commas as extra -D separators.
python3 - "$BD/CMakeFiles/ggml-cuda.dir/flags.make" "$DEF" <<'PY'
import shlex, sys
line = open(sys.argv[1]).read().split("CUDA_DEFINES = ", 1)[1].splitlines()[0]
open(sys.argv[2], "w").write("\n".join(shlex.split(line)) + "\n")
PY
mapfile -t DEFINES < "$DEF"

cd "$BD"
nvcc "${FLAGS[@]}" "${DEFINES[@]}" --options-file CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp \
    -I"$REPO/ggml/src/ggml-cuda" -c "$TU" -o "$OUT"

echo "== DKQ=256 DV=$DV nthreads=$NTHREADS occupancy=$OCCUPANCY, budget $((65536 / NTHREADS / OCCUPANCY)) registers =="
cuobjdump -res-usage "$OUT" | python3 -c '
import re, sys
dv_wanted, budget = int(sys.argv[1]), 65536 // int(sys.argv[2]) // int(sys.argv[3])
seen = set()
for name, usage in re.findall(r"Function (\S+):\s*\n\s*(REG:\d+[^\n]*)", sys.stdin.read()):
    t = re.search(r"ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", name)
    if not t:
        continue
    dkq, dv, ncols1, ncols2 = (int(g) for g in t.groups())
    if dv != dv_wanted:
        continue
    reg = int(re.search(r"REG:(\d+)", usage).group(1))
    stk = int(re.search(r"STACK:(\d+)", usage).group(1))
    loc = int(re.search(r"LOCAL:(\d+)", usage).group(1))
    key = (ncols1 * ncols2, reg, stk)
    if key in seen:                       # the softcap and sparse clones of one shape agree
        continue
    seen.add(key)
    fits = reg <= budget and stk <= 16 and loc == 0
    print("  ncols=%-2s  REG %3s / budget %3s  STACK %4s  LOCAL %3s  %s" % (
        ncols1 * ncols2, reg, budget, stk, loc, "clean" if fits else "over budget or spills"))
' "$DV" "$NTHREADS" "$OCCUPANCY"
