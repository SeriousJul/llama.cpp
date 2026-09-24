#!/usr/bin/env bash
# specs/006 stage 1 gate: the chunked GDN kernel must compile for arch 89 and for a Wave64 build.
#
# There is no ROCm on this box, so a real gfx9 build is not available. This probe checks the thing
# that actually differs for a wave64 build: it recompiles the translation unit with the file-wide
# lane-count constant, ggml_cuda_get_physical_warp_size(), pinned to 64, which is what a wave64
# (gfx9/gfx8) build hands every kernel in the file. If any code in the file assumes 32 lanes at
# compile time, ptxas fails here.
#
# The tree is never modified: the kernel and the headers are copied to an overlay dir, and only the
# copy of common.cuh is patched. The kernel is compiled from inside the overlay so its quoted
# includes resolve to the patched header.
#
# Usage: specs/artifacts/006-wave64-compile.sh  (also run by 006-gate.sh)
set -euo pipefail

REPO=${REPO:-/home/seriousjul/src/llama.cpp}
BUILD=${BUILD:-$REPO/build-004}
WORK=${WORK:-/tmp/006-wave64}

RSP=$BUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp
[ -f "$RSP" ] || { echo "missing $RSP: configure $BUILD first" >&2; exit 1; }

rm -rf "$WORK"
mkdir -p "$WORK/inc"
cp -r "$REPO/ggml/src/ggml-cuda" "$WORK/inc/ggml-cuda"

python3 - "$WORK/inc/ggml-cuda/common.cuh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = """static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}"""
new = """static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 64;  // 006 stage 1 probe: pin the lane count a wave64 build reports
}"""
assert s.count(old) == 1, "warp-size function changed shape; the probe patch is stale"
open(p, "w").write(s.replace(old, new))
print("overlay pinned ggml_cuda_get_physical_warp_size() to 64")
PY

compile_one() {  # $1 = log file, $2.. = extra flags
    local log=$1; shift
    set +e
    /opt/cuda/bin/nvcc -forward-unknown-to-host-compiler \
        -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
        -DGGML_CUDA_FA_F16_F16=1 -DGGML_CUDA_FA_Q4_0_Q4_0=1 -DGGML_CUDA_FA_Q8_0_Q8_0=1 \
        -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -Dggml_cuda_EXPORTS \
        -I"$WORK/inc" --options-file "$RSP" \
        -O3 -DNDEBUG -std=c++17 \
        --generate-code=arch=compute_89,code=[compute_89,sm_89] \
        -Xcompiler=-fPIC -use_fast_math -extended-lambda -compress-mode=size \
        "$@" -x cu -c "$WORK/inc/ggml-cuda/gated_delta_net.cu" -o "$WORK/out.o" > "$log" 2>&1
    local st=$?
    set -e
    [ $st -eq 0 ]
}

report() {  # $1 = label, $2 = log, $3 = status (0 ok), $4 = extra note
    local label=$1 log=$2 st=$3
    if [ "$st" -ne 0 ] || grep -qiE "error" "$log"; then
        echo "FAIL: $label"; tail -25 "$log"; return 1
    fi
    echo "ok:   $label  ${4:-}"
    return 0
}

rc=0

# control: the shipped file must already build at 64 lanes, or this probe proves nothing
git -C "$REPO" show HEAD:ggml/src/ggml-cuda/gated_delta_net.cu > "$WORK/inc/ggml-cuda/gated_delta_net.cu"
compile_one "$WORK/master.log"; st=$?
report "master (recurrent kernel only) at wave64, arch 89" "$WORK/master.log" $st || rc=1

cp "$REPO/ggml/src/ggml-cuda/gated_delta_net.cu" "$WORK/inc/ggml-cuda/gated_delta_net.cu"

for flag in 0 1; do
    compile_one "$WORK/tree$flag.log" "-DGGML_CUDA_GDN_CHUNKED=$flag"; st=$?
    cp "$WORK/out.o" "$WORK/tree$flag.o" 2>/dev/null
    n=$(nm -C "$WORK/tree$flag.o" 2>/dev/null | grep -c chunked || true)
    report "working tree at wave64, GGML_CUDA_GDN_CHUNKED=$flag" "$WORK/tree$flag.log" $st \
           "$n chunked symbols in the object" || rc=1
    if [ "$flag" = "1" ] && [ $st -eq 0 ]; then cp "$WORK/out.o" "$WORK/flag-on.o"; fi
done

echo
echo "--- wave64 chunked instantiations ptxas accepted, from the flag-on object:"
cuobjdump -res-usage "$WORK/flag-on.o" 2>/dev/null | grep -A1 "chunked" | paste - - - | sed 's/--//' || true

if [ $rc -eq 0 ]; then
    echo
    echo "stage 1 wave64 gate: PASS"
fi
exit $rc
