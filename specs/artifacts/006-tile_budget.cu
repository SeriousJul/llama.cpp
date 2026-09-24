// Feasibility stub for 006 stage 3: does the chunked GDN tile fit Ada's shared
// memory and register file at a block shape that can hold the tensor pipe busy?
// Not part of the build. Compile only, read the ptxas report.
//
// Shapes: S_k = S_v = 128, chunk C = 64, one block owns one (head, sequence) and
// loops over chunks. half for mma operands, fp32 accumulators.

#include <cuda_fp16.h>
#include <cstdint>

#ifndef NWARP
#define NWARP 8            // warps per block: 8 -> 256 threads
#endif
#define NT      (NWARP * 32)
#define DV      128
#define DK      128
#define C       64

// Shared-memory layout, as the kernel would need it. Aliasing is explicit so the
// numbers can be checked against the 99 KB per-CTA limit on sm_89.
__global__ void __launch_bounds__(NT, 1) gdn_chunked_probe(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        const float * __restrict__ s0, float * __restrict__ dst, float * __restrict__ state_out,
        int n_chunks) {
    __shared__ __half S[DK * DV];                 // recurrent state, persistent: 32 KB
    __shared__ __half KQ[C * DK];                // one of Q / K at a time:      16 KB
    __shared__ __half K2[C * DK];                // the other of Q / K:          16 KB
    __shared__ __half Vd[C * DV];                // V, later overwritten by delta: 16 KB
    __shared__ float  Tm[C * C / 2];             // strictly-lower T, half lives in Vd alias: 8 KB
    __shared__ float  red[NT / 32][4];           // cross-warp partials for the solve

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;

    for (int i = tid; i < DK * DV; i += NT) S[i] = __float2half(s0[i]);

    for (int ch = 0; ch < n_chunks; ch++) {
        // stage q, k, v for the chunk
        for (int i = tid; i < C * DK; i += NT) {
            KQ[i] = __float2half(q[ch * C * DK + i]);
            K2[i] = __float2half(k[ch * C * DK + i]);
        }
        for (int i = tid; i < C * DV; i += NT) Vd[i] = __float2half(v[ch * C * DV + i]);
        __syncthreads();

        // KS = K @ S and QS = Q @ S, fp32 accumulators held in registers:
        // each thread owns (C * DV) / NT accumulator elements.
        constexpr int NE = (C * DV) / NT;
        float acc[NE];
#pragma unroll
        for (int e = 0; e < NE; ++e) acc[e] = 0.0f;

        for (int t = warp; t < C; t += NWARP) {
            const int c0 = (lane / 4) * 8;
#pragma unroll
            for (int e = 0; e < NE; ++e) {
                const int col = ((lane % 4) * 2 + (e % 2)) + (e / 2) * 32;
                acc[e] += __half2float(K2[t * DK + c0]) * __half2float(S[c0 * DV + col])
                        + __half2float(K2[t * DK + c0 + 1]) * __half2float(S[(c0 + 1) * DV + col]);
            }
        }
        __syncthreads();

        // unit-lower triangular solve over T, fp32 SIMT, then delta = (I+T)^-1 rhs
        for (int i = warp; i < C; i += NWARP) {
            for (int j = lane; j < C; j += 32) {
                if (j < i) Tm[(i * C + j) / 2] *= -1.0f;
            }
        }
        __syncthreads();

        float s = 0.0f;
#pragma unroll
        for (int e = 0; e < NE; ++e) s += acc[e] * __half2float(Vd[e]);
        red[warp][lane % 4] = s;
        __syncthreads();

        if (tid == 0) {
            float t = 0.0f;
            for (int w = 0; w < NWARP; ++w) t += red[w][0];
            dst[ch] = t;
        }

        // state update S = gam*S + K^T @ delta, reusing K2 as delta
        for (int i = tid; i < DK * DV; i += NT) {
            S[i] = __hadd(S[i], __hmul(K2[i % (C * DK)], Vd[i % (C * DV)])); (void)KQ;
        }
        __syncthreads();
    }

    for (int i = tid; i < DK * DV; i += NT) state_out[i] = __half2float(S[i]);
}
