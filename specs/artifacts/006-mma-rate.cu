// specs/006 stage 3: can the tensor pipe feed itself at these tile shapes?
//
// The 4x target needs 13.9 TFLOP/s aggregate (8 % of the 165 TFLOP/s f16 roof); the chunked SIMT
// form manages 3.46. What decides stage 3 is not the mma issue rate but operand delivery, so this
// probe holds f16 tiles in shared memory, moves them through ldmatrix into fragments and accumulates
// with mma.sync at fp32, on the shapes the six chunk products use.
//
// RESULT: the mma path clears this direction's requirement, and the layout has to come from
// mma.cuh, not from reading PTX docs. My hand-written B load used ldmatrix.x2.trans; mma.cuh's
// tile<8,8,T> load_ldmatrix uses x2 with NO transpose, same address pattern. With that one change
// the CPU-matmul check passes (max abs diff ~2e-6, which is f16 rounding of the reference) and the
// rates below appear. Before it, every lane's two column registers held the same n.
//
// Measured on a 4090, 8 warps, 256 blocks, operands resident in shared memory, fp32 accumulate:
//   KS/QS      M64 N32 K128   40.2 TFLOP/s   (24 % of the 165 f16 roof)
//   T, QK^T    M64 N64 K128   54.0 TFLOP/s   (33 %)
//   delta/out  M64 N32 K64    39.4 TFLOP/s   (24 %)
//   state upd  M128 N32 K64   53.1 TFLOP/s   (32 %)
// 4x on the GDN kernel needs 13.9 TFLOP/s aggregate and the chunked SIMT form manages 3.46, so
// the tensor path has roughly 3x margin on the weakest shape. Treat these as an upper bound on
// operand delivery, not a kernel prediction: this probe does one product at a time, with no global
// traffic past the initial tile load, no barriers in the loop, no staging and no epilogue, and the
// real kernel alternates shapes and shares one smem budget.
//
// The timing pass accumulates across iterations, which keeps every mma dependent on the one before
// it, so the loop cannot be hoisted out.
//
// Not part of the build, and it makes no claim about the recurrence.
//   nvcc -arch=sm_89 -O3 -o /tmp/006-mma-rate specs/artifacts/006-mma-rate.cu && /tmp/006-mma-rate

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda: %s\n", cudaGetErrorString(e_)); return 1; } } while (0)

// one warp owns a 16 x (NTILE_N * 8) slab: A via ldmatrix.x4 (16 x 16), each 8-column slab via
// ldmatrix.x2 from the [n][k] tile (a step along n is a whole row of that tile), D as 4 fp32 per
// 16 x 8 tile. The B load takes no .trans, which is what mma.cuh's tile<8,8,T> load_ldmatrix does:
// same address pattern, x2, untransposed.
template <int KT, int NTILE_N>
static __device__ __forceinline__ void mma_step(const half * A, const half * B, int ld, float * acc, int lane) {
    unsigned a0, a1, a2, a3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"((unsigned) __cvta_generic_to_shared(A + (lane % 16) * ld + (lane / 16) * 8)));

#pragma unroll
    for (int n = 0; n < NTILE_N; n++) {
        unsigned b0, b1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(b0), "=r"(b1)
                     : "r"((unsigned) __cvta_generic_to_shared(B + (size_t) ((lane % 8) + n * 8) * ld + (lane / 8) * 8)));

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[n * 4 + 0]), "+f"(acc[n * 4 + 1]), "+f"(acc[n * 4 + 2]), "+f"(acc[n * 4 + 3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

// MT x NT out = (MT x KT) * (NT x KT)^T, 8 warps laid out WARPS_M x WARPS_N over 16-row slabs
template <int MT, int NT, int KT, int WARPS_M, int WARPS_N, int NITER>
__global__ void probe(const half * __restrict__ gA, const half * __restrict__ gB, float * gD) {
    extern __shared__ char sm[];
    half * sA = (half *) sm;                                  // [MT][KT]
    half * sB = sA + (size_t) MT * KT;                        // [NT][KT]

    for (int i = threadIdx.x; i < MT * KT / 4; i += blockDim.x) {
        *(uint2 *) &sA[i * 4] = *(const uint2 *) &gA[(size_t) blockIdx.x * MT * KT + i * 4];
    }
    for (int i = threadIdx.x; i < NT * KT / 4; i += blockDim.x) {
        *(uint2 *) &sB[i * 4] = *(const uint2 *) &gB[(size_t) blockIdx.x * NT * KT + i * 4];
    }
    __syncthreads();

    constexpr int NTILE_N = NT / WARPS_N / 8;                 // 8-column tiles per warp
    static_assert(WARPS_M * WARPS_N == 8, "the probe launches 8 warps");
    static_assert(NT % (WARPS_N * 8) == 0, "columns must divide the warp grid");
    static_assert(MT % (WARPS_M * 16) == 0, "rows must divide the warp grid");

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;

    float acc[NTILE_N][4];
#pragma unroll
    for (int n = 0; n < NTILE_N; n++) {
#pragma unroll
        for (int r = 0; r < 4; r++) acc[n][r] = 0.0f;
    }

    for (int it = 0; it < NITER; it++) {
#pragma unroll
        for (int k = 0; k < KT; k += 16) {
            mma_step<KT, NTILE_N>(sA + (size_t) (wm * 16) * KT + k,
                                  sB + (size_t) (wn * NTILE_N * 8) * KT + k, KT, &acc[0][0], lane);
        }
    }

    if (gD) {
        const int row = wm * 16 + lane / 4;
        const int col = wn * NTILE_N * 8 + (lane % 4) * 2;
#pragma unroll
        for (int n = 0; n < NTILE_N; n++) {
            float * d = gD + (size_t) blockIdx.x * MT * NT;
            d[(size_t) row * NT + col + n * 8]     = acc[n][0];
            d[(size_t) row * NT + col + n * 8 + 1] = acc[n][1];
            d[(size_t) (row + 8) * NT + col + n * 8]     = acc[n][2];
            d[(size_t) (row + 8) * NT + col + n * 8 + 1] = acc[n][3];
        }
    }
}

static float frand() { return (float) rand() / RAND_MAX - 0.5f; }

template <int MT, int NT, int KT, int WARPS_M, int WARPS_N, int NITER, int BLOCKS>
static int run(const char * label) {
    std::vector<half> hA((size_t) BLOCKS * MT * KT), hB((size_t) BLOCKS * NT * KT);
    for (auto & x : hA) x = __float2half_rn(frand());
    for (auto & x : hB) x = __float2half_rn(frand());

    half * dA; half * dB; float * dD;
    CK(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CK(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dD, (size_t) MT * NT * sizeof(float)));

    constexpr size_t smem = ((size_t) MT * KT + (size_t) NT * KT) * sizeof(half);
    auto k_check = probe<MT, NT, KT, WARPS_M, WARPS_N, 1>;
    CK(cudaFuncSetAttribute(k_check, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    // 1. correctness of the fragment mapping, against a CPU matmul of block 0
    k_check<<<1, 256, smem>>>(dA, dB, dD);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t) MT * NT);
    CK(cudaMemcpy(got.data(), dD, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (int i = 0; i < MT; i++) {
        for (int j = 0; j < NT; j++) {
            float ref = 0.0f;
            for (int k = 0; k < KT; k++) {
                ref += __half2float(hA[(size_t) i * KT + k]) * __half2float(hB[(size_t) j * KT + k]);
            }
            worst = fmax(worst, fabs(ref - got[(size_t) i * NT + j]));
        }
    }
    if (worst > 1e-2) {
        printf("%-26s FRAGMENT MAPPING WRONG: max abs diff %.3e\n", label, worst);
        cudaFree(dA); cudaFree(dB); cudaFree(dD);
        return 1;
    }

    // 2. rate
    auto k_rate = probe<MT, NT, KT, WARPS_M, WARPS_N, NITER>;
    CK(cudaFuncSetAttribute(k_rate, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    k_rate<<<BLOCKS, 256, smem>>>(dA, dB, nullptr);   // warm
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    k_rate<<<BLOCKS, 256, smem>>>(dA, dB, nullptr);
    cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);

    const double flop = (double) BLOCKS * NITER * MT * NT * KT * 2.0;
    const double tflops = flop / (ms * 1e-3) / 1e12;
    printf("%-26s %-14s max diff %.1e   %7.2f TFLOP/s   (%.1f %% of the 165 f16 roof)%s\n",
           label, "map ok", worst, tflops, 100.0 * tflops / 165.0,
           tflops >= 13.9 ? "   >= 4x target" : "");

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return 0;
}

int main() {
    srand(1234);
    int rc = 0;
    // KS / QS at C = 64: 64 x 32 out, 128-deep; and the same shape at C = 16
    rc |= run<64, 32, 128, 4, 2, 64, 256>("KS/QS  M64 N32 K128");
    rc |= run<64, 32, 128, 4, 2, 64, 128>("  same, 128 blocks");
    rc |= run<64, 32, 128, 4, 2, 64, 64>("  same,  64 blocks");
    rc |= run<64, 32, 128, 4, 2, 64, 32>("  same,  32 blocks");
    // T and Q K^T: 64 x 64 out, 128-deep
    rc |= run<64, 64, 128, 4, 2, 64, 256>("T, QK^T  M64 N64 K128");
    rc |= run<64, 64, 128, 4, 2, 64, 32>("  same,  32 blocks");
    // inverse applied to rhs and the intra output term: 64 x 32 out, 64-deep
    rc |= run<64, 32, 64, 4, 2, 64, 256>("delta/out  M64 N32 K64");
    // state update: 128 x 32 out, 64-deep
    rc |= run<128, 32, 64, 8, 1, 64, 256>("state upd  M128 N32 K64");
    rc |= run<128, 32, 64, 8, 1, 64, 32>("  same,  32 blocks");
    rc |= run<128, 32, 64, 8, 1, 64, 128>("  same, 128 blocks");
    printf("\n4x on the GDN kernel needs 13.9 TFLOP/s aggregate; the SIMT form measures 3.46.\n");
    return rc;
}
