// specs/006 stage 1: does the scaffold's shared-memory layout actually launch, and how many
// blocks per SM does the driver give it? The gate table in the spec is arithmetic; this is the
// measured answer, for both chunk sizes the design left open.
//
// Not part of the build. Compile and run:
//   nvcc -arch=sm_89 -O3 -o /tmp/006-launch-smem specs/artifacts/006-launch-smem.cu && /tmp/006-launch-smem
//
// The smem sizes mirror ggml_cuda_gdn_chunked_smem<S_v=128, C> in gated_delta_net.cu, which is
// itself pinned by static_asserts to 44 KiB (C=16) and 80 KiB (C=64).

#include <cstdio>
#include <cuda_fp16.h>

template <int S_v, int C>
static constexpr size_t smem_bytes() {
    const size_t state = (size_t) S_v * S_v * sizeof(half);
    const size_t kq    = (size_t) 2 * C * S_v * sizeof(half);
    const size_t vt    = (size_t) C * S_v * sizeof(half);
    const size_t tt    = (size_t) C * C * sizeof(float);
    return state + kq + (vt > tt ? vt : tt);
}

__device__ int g_sink;

template <int NBYTES>
__global__ void __launch_bounds__(256) probe(const float * in, float * out) {
    extern __shared__ char sm[];
    sm[0] = (char) in[threadIdx.x];
    sm[NBYTES - 1] = (char) threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) g_sink = sm[0] + sm[NBYTES - 1];
    if (out) out[0] = sm[1];
}

template <int S_v, int C>
static void run(const char * label) {
    constexpr int N = (int) smem_bytes<S_v, C>();
    static_assert(N % 16 == 0, "tile sizes must stay 16 B aligned");

    float * in; float * out;
    cudaMalloc(&in, 4096);
    cudaMalloc(&out, 4096);
    cudaMemset(in, 1, 4096);

    auto k = probe<N>;
    cudaError_t sa = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, N);
    if (sa != cudaSuccess) {
        printf("%-8s C=%-3d %6d B  cudaFuncSetAttribute FAILED: %s\n",
               label, C, N, cudaGetErrorString(sa));
        cudaFree(in); cudaFree(out);
        return;
    }

    int blocks_per_sm = -1;
    cudaError_t oc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, k, 256, N);

    k<<<8, 256, N>>>(in, out);
    cudaError_t la = cudaGetLastError();
    cudaError_t sy = cudaDeviceSynchronize();

    printf("%-8s C=%-3d %6d B (%.1f KiB)  launch: %-10s  sync: %-10s  blocks/SM: %d%s\n",
           label, C, N, N / 1024.0, cudaGetErrorString(la), cudaGetErrorString(sy), blocks_per_sm,
           oc != cudaSuccess ? "  (occupancy query failed)" : "");

    cudaFree(in); cudaFree(out);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("device %s  sm_%d%d\n", p.name, p.major, p.minor);
    printf("  sharedMemPerBlock %zu B  optin %d B  perSM %zu B\n\n",
           p.sharedMemPerBlock, p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor);

    run<128, 16>("gate");
    run<128, 32>("gate");
    run<128, 64>("gate");
    return 0;
}
