// which fragment order does m16n8k16.row.col.f16 actually want? one warp, four structured probes.
#include <cstdio>
#include <cuda_fp16.h>
#define KT 16
__device__ __forceinline__ void one(const half* A, const half* B, float* acc, unsigned* ra, unsigned* rb) {
    int lane = threadIdx.x % 32;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"((unsigned)__cvta_generic_to_shared(A + (lane % 16) * KT + (lane / 16) * 8)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(rb[0]), "=r"(rb[1])
        : "r"((unsigned)__cvta_generic_to_shared(B + (lane % 8) * KT + (lane / 8) * 8)));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
}
// mode 0: A[i][k] = 1 only at k==0   -> D[i][n] = B[n][0]
// mode 1: A[i][k] = 1 only at k==8   -> D[i][n] = B[n][8]
// mode 2: A[i][k] = 1 only at k==2   -> D[i][n] = B[n][2]
// mode 3: A[i][k] = (k==0)? (i+1) : 0 -> D[i][n] = (i+1) * B[n][0]
__global__ void k(int mode, float* out) {
    __shared__ half A[16*KT], B[8*KT];
    for (int i = threadIdx.x; i < 16*KT; i += 32) {
        int r = i / KT, c = i % KT;
        float v = 0;
        if (mode == 0) v = (c == 0);
        if (mode == 1) v = (c == 8);
        if (mode == 2) v = (c == 2);
        if (mode == 3) v = (c == 0) ? (float)(r + 1) : 0.f;
        A[i] = __float2half_rn(v);
    }
    for (int i = threadIdx.x; i < 8*KT; i += 32) {
        int n = i / KT, c = i % KT;
        B[i] = __float2half_rn((float)(n + 1) + (c == 0 ? 0.f : 0.f) + (c == 8 ? 100.f : 0.f) + (c == 2 ? 1000.f : 0.f));
    }
    __syncwarp();
    float acc[4] = {0,0,0,0}; unsigned ra[4], rb[2];
    one(A, B, acc, ra, rb);
    int row = threadIdx.x / 4, col = (threadIdx.x % 4) * 2;
    out[threadIdx.x*4+0] = acc[0]; out[threadIdx.x*4+1] = acc[1];
    out[threadIdx.x*4+2] = acc[2]; out[threadIdx.x*4+3] = acc[3];
    if (threadIdx.x == 0) printf("  (lane0 holds D[row0, col0..1] = %g %g ; D[row8, col0..1] = %g %g)\n", acc[0], acc[1], acc[2], acc[3]);
}
int main() {
    float* d; cudaMalloc(&d, 32*4*sizeof(float)); float h[128];
    for (int m = 0; m < 4; m++) {
        printf("mode %d:\n", m);
        k<<<1,32>>>(m, d); cudaDeviceSynchronize();
        cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
        // report what lane 0 saw and whether every lane's value matches B[n][k] expectations
        int bad = 0;
        for (int l = 0; l < 32; l++) {
            int row = l/4, col0 = (l%4)*2;      // assumed D layout
            for (int q = 0; q < 4; q++) {
                int r = row + (q >= 2 ? 8 : 0), c = col0 + (q % 2);
                float expct;
                int kk = (m == 1) ? 8 : (m == 2) ? 2 : 0;
                float bv = (float)(c + 1) + (kk == 8 ? 100.f : 0.f) + (kk == 2 ? 1000.f : 0.f);
                expct = (m == 3) ? (float)(r + 1) * ((c+1) + 0.f) : bv;
                if (m == 3) expct = (float)(r + 1) * (float)(c + 1);
                if (fabsf(expct - h[l*4+q]) > 1e-3) bad++;
            }
        }
        printf("  mismatches under the assumed layout: %d/128\n", bad);
    }
    return 0;
}
