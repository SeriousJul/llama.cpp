// specs/006 stage 3: does a whole chunk body fit in the tensor pipe, at the grid the design gets?
//
// 006-mma-rate.cu timed one product at a time on 256 blocks. That is not what the kernel does: the
// real chunk issues seven products that share one shared-memory budget and a barrier between each,
// and the (C, DV_TILE) choice decides both the FLOPs (the column split duplicates T and Q K^T) and
// how many CTAs exist, so a config can have a comfortable roof share and still starve the machine.
//
// This probe runs the seven-product sequence per chunk, for one (C, DV_TILE), on exactly the CTA
// count the wrapper would launch (H_v * DV/DV_TILE), with no GDN semantics and no correctness claim
// about the recurrence. The state update reads its A operand with the wrong transpose for a real
// product (the shape and the instruction counts are right, the bank pattern is approximate), and
// `us/(layer,ubatch)` is only comparable to 320.8 for the configs whose CHUNKS * C is 512; the
// TFLOP/s figure and the verdict column are independent of both. It reports aggregate TFLOP/s and, next to it, what 4x on the recurrent
// kernel's 320.8 us would need.
//
// Not part of the build. Build and run one config per process:
//   for c in 16 32 64; do nvcc -arch=sm_89 -O3 -DCC=$c -o /tmp/body$c specs/artifacts/006-mma-chunkbody.cu; done
//   /tmp/body16 ; /tmp/body32 ; /tmp/body64

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda: %s\n", cudaGetErrorString(e_)); return 1; } } while (0)

#ifndef CC
#define CC 64
#endif
#ifndef DVT
#define DVT 32
#endif
#ifndef CHUNKS
#define CHUNKS 32
#endif
#ifndef NBUF
#define NBUF 8                                 // distinct global operand sets to rotate through
#endif

constexpr int DK    = 128;
constexpr int H_V   = 32;                       // 9B value heads
constexpr int WARPS = 8;
constexpr int NTHR  = 32 * WARPS;

// one warp covers a 16-row slab and NTILE_N 8-column tiles; A is [i][k], B is [n][k] with k fastest
// NTILE_N 8-column tiles per warp, so a product of N columns needs N == WARPS_N * NTILE_N * 8
template <int NTILE_N, int KT_STEPS>
// A and B are the tile bases for THIS warp's row slab and column group, so the row offset of the
// slab lives in the caller; only the column group is derived from wn here.
static __device__ __forceinline__ void prod(const half * A, const half * B, int lda, int ldb,
                                            float (*acc)[4], int wn, int lane) {
#pragma unroll
    for (int kk = 0; kk < KT_STEPS; kk++) {
        unsigned a0, a1, a2, a3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "l"(__cvta_generic_to_shared(A + (size_t) (lane % 16) * lda + (lane / 16) * 8 + kk * 16)));
#pragma unroll
        for (int n = 0; n < NTILE_N; n++) {
            unsigned b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b0), "=r"(b1)
                         : "l"(__cvta_generic_to_shared(B + ((size_t) ((wn % 8)*NTILE_N + n) * 8 + (lane % 8)) * ldb + (wn / 8) * 8 + kk * 16)));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[n][0]), "+f"(acc[n][1]), "+f"(acc[n][2]), "+f"(acc[n][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
}

// one chunk of the real body: KS, QS, T, QK^T, inverse applied to rhs, intra output, state update
template <int C, int DV_TILE>
__global__ void __launch_bounds__(NTHR) chunk_body(const half * __restrict__ src, float * sink) {
    extern __shared__ char sm[];
    constexpr size_t h_kq   = (size_t) 2 * C * DK;          // k, q  [C][DK]
    constexpr size_t h_s    = (size_t) DV_TILE * DK;        // state [DV_TILE][DK]
    constexpr size_t h_cc   = (size_t) C * C;               // mat scratches [C][C]
    constexpr size_t h_cd   = (size_t) DV_TILE * C;         // rhs / delta, transposed [DV_TILE][C]
    constexpr size_t h_kt   = (size_t) DK * C;            // k^T, the state update's A operand

    half * sK = (half *) sm;
    half * sQ = sK + h_kq / 2;
    half * sS = sQ + h_kq / 2;
    half * sT = sS + h_s;
    half * sA = sT + h_cc;
    half * sD = sA + h_cc;
    half * sR = sD + h_cd;
    half * sKT = sR + h_cd;

    // Restage every chunk from one of NBUF global operand sets. This is what the real kernel does,
    // and it is what makes the probe honest: with the tiles written once before the loop the
    // operands are loop invariant, so nvcc folds CHUNKS identical iterations into one and reports a
    // rate above the hardware roof. Rotating the source per chunk cannot be folded.
    constexpr int NB4 = (int) ((h_kq + h_s + h_cc * 2 + h_cd * 2 + h_kt) / 4);
    auto stage = [&](int set) {
        const uint2 * g = (const uint2 *) (src + (size_t) set * (h_kq + h_s + h_cc * 2 + h_cd * 2 + h_kt));
#pragma unroll 4
        for (int i = threadIdx.x; i < NB4; i += NTHR) ((uint2 *) sK)[i] = g[i];
    };
    stage(0);
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float acc[8][4];   // NTILE_N is at most 8: DV_TILE or C over W_N*8
    constexpr int W_M = 4, W_N = 2;                       // warp grid: 8 warps
    static_assert(DV_TILE % (W_N * 8) == 0 && C % (W_N * 8) == 0, "columns must divide the warp grid");

    for (int ch = 0; ch < CHUNKS; ch++) {
        const int wm = warp / W_N, wn = warp % W_N;

        if (ch + 1 < CHUNKS) stage((blockIdx.x + ch + 1) % NBUF);
        __syncthreads();

        // KS = k @ S and QS = q @ S                      : M=C, N=DV_TILE, K=DK
#pragma unroll
        for (int mi = wm; mi < C / 16; mi += W_M) {
            prod<DV_TILE / (W_N * 8), DK / 16>(sK + (size_t) mi * 16 * DK, sS, DK, DK, acc, wn, lane);
            prod<DV_TILE / (W_N * 8), DK / 16>(sQ + (size_t) mi * 16 * DK, sS, DK, DK, acc, wn, lane);
        }
        __syncthreads();

        // T = k @ k^T and A = q @ k^T                    : M=C, N=C, K=DK
#pragma unroll
        for (int mi = wm; mi < C / 16; mi += W_M) {
            prod<C / (W_N * 8), DK / 16>(sK + (size_t) mi * 16 * DK, sK, DK, DK, acc, wn, lane);
            prod<C / (W_N * 8), DK / 16>(sQ + (size_t) mi * 16 * DK, sK, DK, DK, acc, wn, lane);
        }
        __syncthreads();

        // delta = mat @ rhs and out_intra = mat @ delta  : M=C, N=DV_TILE, K=C
#pragma unroll
        for (int mi = wm; mi < C / 16; mi += W_M) {
            prod<DV_TILE / (W_N * 8), C / 16>(sT + (size_t) mi * 16 * C, sR, C, C, acc, wn, lane);
            prod<DV_TILE / (W_N * 8), C / 16>(sA + (size_t) mi * 16 * C, sD, C, C, acc, wn, lane);
        }
        __syncthreads();

        // state update = k^T @ delta                     : M=DK, N=DV_TILE, K=C
#pragma unroll
        for (int mi = wm; mi < DK / 16; mi += W_M) {
            prod<DV_TILE / (W_N * 8), C / 16>(sKT + (size_t) mi * 16 * C, sD, C, C, acc, wn, lane);
        }
        __syncthreads();

        // the fragments of this chunk are consumed by the next chunk's rhs in the real kernel; here
        // a per-chunk store keeps them alive
        if (threadIdx.x == 0 && sink) {
            float keep = 0.0f;
#pragma unroll
            for (int n = 0; n < 8; n++) {
#pragma unroll
                for (int r = 0; r < 4; r++) keep += acc[n][r];
            }
            sink[1] = keep;
        }

    }

    // Store the fragments out once per chunk, which is what the real kernel does and what makes the
    // measurement honest. A conditional keep-alive was deleted by nvcc whenever a config wrote fewer
    // column tiles than the accumulator array holds: it left the staging in place and removed every
    // HMMA, and the resulting rate was above the hardware roof.
    if (threadIdx.x == 0 && sink) {
        float keep = 0.0f;
#pragma unroll
        for (int n = 0; n < 8; n++) {
#pragma unroll
            for (int r = 0; r < 4; r++) keep += acc[n][r];
        }
        *sink = keep;
    }
}

template <int C, int DV_TILE>
static int bench(const char * label) {
    constexpr size_t halves = (size_t) 2 * C * DK + (size_t) DV_TILE * DK + 2 * (size_t) C * C
                            + 2 * (size_t) DV_TILE * C + (size_t) DK * C;
    constexpr size_t bytes = halves * sizeof(half);
    constexpr int ctas = H_V * (DK / DV_TILE);
    auto k = chunk_body<C, DV_TILE>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes));

    half * dsrc;
    std::vector<half> hs((size_t) NBUF * halves);
    for (size_t i = 0; i < hs.size(); i++) hs[i] = __float2half_rn((float) ((i * 2654435761u) % 1000) * 0.001f - 0.5f);
    CK(cudaMalloc(&dsrc, hs.size() * sizeof(half)));
    CK(cudaMemcpy(dsrc, hs.data(), hs.size() * sizeof(half), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float * dsink; CK(cudaMalloc(&dsink, 2 * sizeof(float)));
    k<<<ctas, NTHR, bytes>>>(dsrc, dsink);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int r = 0; r < 5; r++) k<<<ctas, NTHR, bytes>>>(dsrc, dsink);
    cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5.0f;

    // FLOPs per (CTA, chunk): KS QS, T A, delta intra, state update
    const double f = 2.0*(C*DV_TILE*DK*2.0) + 2.0*(C*C*DK*2.0) + 2.0*(C*DV_TILE*C*2.0) + (DV_TILE*DK*C*2.0);   // includes k^T as its own tile
    const double gf_per_call = f * ctas * CHUNKS / 1e9;
    const double tf = f * (double) ctas * CHUNKS / ((ms * 1e-3) * 1e12);
    // one launch covers CHUNKS * C tokens; normalize to a 512-token ubatch so us is comparable
    const double per_512 = 512.0 / (CHUNKS * C);
    const double us_per_call = ms * 1e3 * per_512;
    const double gf_512 = gf_per_call * per_512;
    const double need4 = (f * ctas * (512.0 / C) * 1e-9) / (320.8 / 4.0 * 1e-6) / 1e3;

    printf("%-22s CTA %4d  smem %6zu B  %7.1f us/(layer,512tok)  %5.2f GFLOP  %6.1f TFLOP/s"
           "   4x needs %5.1f  -> %s\n",
           label, ctas, bytes, us_per_call, gf_512, tf, need4,
           tf >= need4 ? "4x reachable" : (tf >= need4/2 ? "2x at best" : "no"));
    return 0;
}

int main() {
    printf("chunk body: 7 products, %d chunks per launch, 8 warps, C=%d DV_TILE=%d\n", CHUNKS, CC, DVT);
    bench<16,  32>("C=16  DVT= 32");
    bench<16,  64>("C=16  DVT=  64");
    bench<16, 128>("C=16  DVT=128");
    bench<32,  64>("C=32  DVT= 64");
    bench<32,  32>("C=32  DVT= 32");
    bench<32, 128>("C=32  DVT=128");
    bench<64,  64>("C=64  DVT= 64");
    bench<64,  32>("C=64  DVT= 32");
    return 0;
}
