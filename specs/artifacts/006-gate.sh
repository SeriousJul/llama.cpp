#!/usr/bin/env bash
# specs/006 gate, stages 1 and 2, in one command.
#
#   1. arch 89 builds, flag off and flag on
#   2. with the flag off the tree behaves identically: the device code of the whole translation
#      unit is byte-for-byte the committed tree's, and no chunked symbol exists
#   3. with the flag on the shipped kernels are still untouched
#   4. it compiles for a Wave64 build
#   5. registers and shared memory come from the compile, not from reading declarations
#   6. the op suite is green through the chunked path, repeatedly, because a marginal tolerance
#      overshoot shows up as an intermittent failure and not as a clean one
#
# No benchmark is run: stages 1 and 2 make no perf claim.
#
# Controls are compiled from a copied header/source tree under /tmp, never by editing the repo, so
# the working tree is only ever touched by the flag flips below.
#
# Usage: specs/artifacts/006-gate.sh
set -uo pipefail

REPO=${REPO:-/home/seriousjul/src/llama.cpp}
BUILD=${BUILD:-$REPO/build-004}
SRC=$REPO/ggml/src/ggml-cuda/gated_delta_net.cu
OBJ=$BUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/gated_delta_net.cu.o
LIB=$BUILD/bin/libggml-cuda.so.0.25.0
WORK=/tmp/006-gate
rc=0

RSP=$BUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp
[ -f "$RSP" ] || { echo "missing $RSP: configure $BUILD first" >&2; exit 1; }

rm -rf "$WORK"; mkdir -p "$WORK"
cp "$SRC" "$WORK/scaffold.cu"
trap 'cp "$WORK/scaffold.cu" "$SRC"; cmake --build "$BUILD" --target ggml-cuda -j 24 >/dev/null 2>&1; echo "(tree restored; flag = $(grep -m1 "^#define GGML_CUDA_GDN_CHUNKED" "$SRC" | awk "{print \$3}"))"' EXIT INT TERM

set_flag() { sed -i "s/^#define GGML_CUDA_GDN_CHUNKED .$/#define GGML_CUDA_GDN_CHUNKED $1/" "$SRC"; }
build()    { touch "$SRC"; cmake --build "$BUILD" --target ggml-cuda -j 24 > "$WORK/build.log" 2>&1; }
errs()     { grep -iE "error" "$WORK/build.log" | grep -v "ggml-impl" | head -5; }

# compile one source file out-of-tree, with a header overlay that may be patched
compile_one() {  # $1 = src, $2 = out .o, $3 = overlay include dir or empty, $4... = defines
    local src=$1 out=$2 inc=$3; shift 3
    set +e
    /opt/cuda/bin/nvcc -forward-unknown-to-host-compiler \
        -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
        -DGGML_CUDA_FA_F16_F16=1 -DGGML_CUDA_FA_Q4_0_Q4_0=1 -DGGML_CUDA_FA_Q8_0_Q8_0=1 \
        -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -Dggml_cuda_EXPORTS \
        ${inc:+-I"$inc"} --options-file "$RSP" \
        -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89] \
        -Xcompiler=-fPIC -use_fast_math -extended-lambda -compress-mode=size \
        "$@" -x cu -c "$src" -o "$out" > "$WORK/one.log" 2>&1
    local st=$?
    set -e
    [ $st -eq 0 ] && ! grep -qiE "error" "$WORK/one.log"
}

echo "=== 1. arch 89, flag off"
set_flag 0; build
if [ -n "$(errs)" ]; then echo "FAIL: build errored"; errs; rc=1; else
    echo "ok, $(nm -C "$OBJ" | grep -c chunked) chunked symbols in the object"
    cuobjdump -sass "$OBJ" | grep -v '^ *$' > "$WORK/sass-off.txt"
    cuobjdump -res-usage "$OBJ" > "$WORK/res-off.txt"
    cp "$OBJ" "$WORK/off.o"
fi

echo
echo "=== 2. arch 89, flag on"
set_flag 1; build
if [ -n "$(errs)" ]; then echo "FAIL: build errored"; errs; rc=1; else
    echo "ok, $(nm -C "$OBJ" | grep -c chunked) chunked symbols in the object"
    cuobjdump -sass "$OBJ" | grep -v '^ *$' > "$WORK/sass-on.txt"
    cuobjdump -res-usage "$OBJ" > "$WORK/res-on.txt"
fi

echo
echo "=== 3. flag off changes nothing, against the committed tree"
mkdir -p "$WORK/head/ggml-cuda"
cp -r "$REPO/ggml/src/ggml-cuda/." "$WORK/head/ggml-cuda/"
git -C "$REPO" show HEAD:ggml/src/ggml-cuda/gated_delta_net.cu > "$WORK/head/ggml-cuda/gated_delta_net.cu"
if compile_one "$WORK/head/ggml-cuda/gated_delta_net.cu" "$WORK/head.o" "$WORK/head"; then
    cuobjdump -sass "$WORK/head.o" | grep -v '^ *$' > "$WORK/sass-head.txt"
    if diff -q "$WORK/sass-head.txt" "$WORK/sass-off.txt" > /dev/null; then
        echo "ok, flag-off device SASS is identical to the committed tree ($(wc -l < "$WORK/sass-head.txt") lines)"
    else
        echo "FAIL: flag-off device SASS differs from the committed tree"; diff "$WORK/sass-head.txt" "$WORK/sass-off.txt" | head -10; rc=1
    fi
    if diff -q <(cuobjdump -res-usage "$WORK/head.o") "$WORK/res-off.txt" > /dev/null; then
        echo "ok, flag-off resource usage identical too"
    else
        echo "FAIL: flag-off resource usage differs"; rc=1
    fi
    # and the shipped kernels must survive the scaffold being compiled in
    strip_chunked() { awk '/^ *Function /{keep = ($0 !~ /chunked/)} keep' "$1"; }
    if diff <(strip_chunked "$WORK/sass-head.txt") <(strip_chunked "$WORK/sass-on.txt") > /dev/null; then
        echo "ok, flag on leaves every shipped kernel's device code identical"
    else
        echo "FAIL: the flag-on build perturbed a shipped kernel"; rc=1
    fi
else
    echo "FAIL: the committed tree did not compile in the overlay"; tail -15 "$WORK/one.log"; rc=1
fi

echo
echo "=== 4. registers and shared memory, from the compile (flag on)"
python3 - "$WORK/res-on.txt" <<'PY'
import re, sys
t = open(sys.argv[1]).read().split('\n')
for i, l in enumerate(t):
    m = re.search(r'chunked_cudaILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)', l)
    if not m or i + 1 >= len(t):
        continue
    r = re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', t[i + 1])
    if r:
        print(f"  S_v={m.group(1):>3} C={m.group(2):>2} DV_TILE={m.group(3):>3} LANES={m.group(4):>2}"
              f"   REG={r.group(1):>3}  STACK={r.group(2)}  LOCAL={r.group(4)}")
        if r.group(2) != '0' or r.group(4) != '0':
            print("  ^^ stack or local traffic: this instantiation spills")
PY
echo "  SHARED is 0 because the tiles are one dynamic request, sized by"
echo "  ggml_cuda_gdn_chunked_smem<S_v,C,DV_TILE>::bytes and pinned by:"
grep -n "static_assert(ggml_cuda_gdn_chunked_smem" "$SRC" | sed 's/^/    /'

echo
echo "=== 5. Wave64"
if bash "$REPO/specs/artifacts/006-wave64-compile.sh" > "$WORK/wave64.log" 2>&1; then
    grep -E "^ok:|^FAIL" "$WORK/wave64.log" | sed 's/^/  /'
else
    echo "FAIL: wave64 probe"; tail -25 "$WORK/wave64.log"; rc=1
fi
grep -E "^ok:|^FAIL" "$WORK/wave64.log" | grep -q "CHUNKED=1  0 chunked" && { echo "FAIL: flag-on wave64 shows no chunked symbols, the probe is not measuring"; rc=1; }

echo
echo "=== 6. stage 2 correctness gate: the op suite through the chunked path, flag on"
set_flag 1; build
if [ -n "$(errs)" ]; then echo "FAIL: flag-on build errored"; errs; rc=1; else
    bad=0
    for i in 1 2 3 4 5 6 7 8; do
        timeout 900 "$BUILD/bin/test-backend-ops" -o GATED_DELTA_NET -b CUDA0 > "$WORK/suite$i.log" 2>&1
        grep -q "tests passed" "$WORK/suite$i.log" && ! grep -q "FAIL" "$WORK/suite$i.log" \
            || { bad=$((bad+1)); echo "  suite run $i FAILED"; grep -m2 "ERR =" "$WORK/suite$i.log"; }
    done
    if [ $bad -eq 0 ]; then
        echo "  ok, 8/8 suite runs green with the chunked path selected"
    else
        echo "  FAIL: $bad of 8 suite runs failed, see $WORK/suite*.log"; rc=1
    fi
fi

echo
echo "=== 7. restore flag off and report the library later stages will A/B against"
set_flag 0; build
if [ -n "$(errs)" ]; then echo "FAIL: flag-off rebuild errored"; rc=1; else
    [ "$(nm -C "$OBJ" | grep -c chunked)" = "0" ] || { echo "FAIL: chunked symbols present with the flag off"; rc=1; }
    md5sum "$LIB" | sed 's|^|  flag-off baseline  |'
fi

if [ $rc -eq 0 ]; then echo; echo "GATE: PASS"; else echo; echo "GATE: FAIL"; fi
exit $rc
