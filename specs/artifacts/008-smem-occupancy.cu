// 008: is shared memory, independent of registers, a second gate on attention occupancy?
// Asks the same API launch_fattn asks (cudaOccupancyMaxActiveBlocksPerMultiprocessor) with a
// dummy kernel whose only relevant property is its thread count, fed the exact
// nbytes_shared_total the launcher computes for each shape. Register pressure is measured
// separately by 008-reg-gate.sh; a shape is only reachable if it passes both.
//
//   nvcc -O2 -o /tmp/008/008-smem-occupancy 008-smem-occupancy.cu
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "cuda error at %d: %s\n", __LINE__, cudaGetErrorString(e_));          \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

extern __shared__ char smem[];

__global__ void dummy(float * p) {
    if (p && threadIdx.x == 999999u) {
        smem[0] = 1;
    }
}

// flash_attn_mma_shared_bytes mirrors ggml_cuda_flash_attn_ext_mma_f16_case, DKQ = 256.
// Q_in_reg makes tile_K alias tile_Q, so the total is a max, not a sum.
struct shape {
    const char * what;
    int ncols, ncols1, nwarps;
    int nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine;
    int nstages;
    bool q_in_reg;
};

static size_t shared_bytes(const shape & s) {
    const int stride_Q  = 256 / 2 + 4;
    // swizzling needs a half2 stride that is a multiple of 32, else the tile keeps row padding
    const int stride_K  = (s.nbatch_K2 % 32 == 0) ? s.nbatch_K2 : s.nbatch_K2 + 4;
    const int stride_V  = (s.nbatch_V2 % 32 == 0) ? s.nbatch_V2 : s.nbatch_V2 + 4;
    const size_t Q      = (size_t)s.ncols    * stride_Q * 4;
    const size_t KV     = s.nstages > 1 ? (size_t)s.nbatch_fa * (stride_K + stride_V) * 4
                                        : (size_t)s.nbatch_fa * (stride_K > stride_V ? stride_K : stride_V) * 4;
    const size_t mask   = (size_t)s.ncols1   * (s.nbatch_fa / 2 + 4) * 4;
    const size_t combine = (size_t)s.nwarps * 16 * (s.nbatch_combine + 4) * 4;
    return s.q_in_reg ? (combine > (Q > KV + mask ? Q : KV + mask) ? combine : (Q > KV + mask ? Q : KV + mask))
                      : Q + KV + mask;
}
static int blocks_for(size_t nbytes, int nthreads, int * limit_by_smem) {
    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dummy, nthreads, nbytes) != cudaSuccess) {
        return -1;
    }
    *limit_by_smem = blocks;
    return blocks;
}

int main() {
    int device = 0;
    CK(cudaSetDevice(device));
    CK(cudaFree(0));

    int per_sm = 0, per_block = 0, regs_per_sm = 0, nsm = 0;
    CK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CK(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CK(cudaDeviceGetAttribute(&regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device));
    printf("device: %d B shared per SM, %d B max per CTA, %d registers per SM, %d SMs\n\n",
           per_sm, per_block, regs_per_sm, nsm);

    const shape shapes[] = {
        // the shipping production row: DKQ=256, DV=256, ncols=64, 4 warps, 2 pipeline stages
        { "ships        ncols=64 DV=256 n128 nstages=2",      64, 8, 4, 32, 128, 128, 128, 2, true },
        // the frozen design: same tile, half DV, 8 warps, and the budget it needs is 128 registers
        { "DV split     ncols=64 DV=128 n256 nstages=2",      64, 8, 8, 32, 128,  64,  64, 2, true },
        { "DV split     ncols=64 DV=128 n256 nstages=1",      64, 8, 8, 32, 128,  64,  64, 1, true },
        // the same at 4 warps, which is where 3 CTAs was supposed to come from
        { "DV split     ncols=64 DV=128 n128 nstages=2",      64, 8, 4, 32, 128,  64,  64, 2, true },
        // half DV with the fallback tile width the register gate says is the only spill-free one
        { "DV split nmw ncols=32 DV=128 n128 nstages=2",      32, 8, 4, 32, 128,  64,  64, 2, true },
        { "DV split nmw ncols=32 DV=128 n128 nstages=1",      32, 8, 4, 32, 128,  64,  64, 1, true },
        // no split, narrow tile, single stage: could existing knobs alone raise the warp count?
        { "narrow       ncols=32 DV=256 n128 nstages=1",      32, 8, 4, 32, 128, 128, 128, 1, true },
        { "narrow       ncols=16 DV=256 n128 nstages=1",      16, 8, 4, 32, 128, 128, 128, 1, true },
    };

    for (const shape & s : shapes) {
        const size_t bytes = shared_bytes(s);
        int by_smem = 0;
        const int nthreads = s.nwarps * 32;
        if (blocks_for(bytes, nthreads, &by_smem) < 0) {
            printf("%-42s query failed\n", s.what);
            continue;
        }
        printf("%-44s %6zu B/CTA  smem allows %d CTA/SM = %2d warps  (register budget %d)\n",
               s.what, bytes, by_smem, by_smem * s.nwarps, regs_per_sm / (nthreads * by_smem));
    }
    return 0;
}
