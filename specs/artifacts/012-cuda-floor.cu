// 012 probe: what sits in the "unaccounted" MiB band of common_memory_breakdown_print,
// i.e. bytes the process holds that no ggml buffer claims. Measures, in order:
//   1. the CUDA context (NVML per-process usage; cudaMemGetInfo cannot see it, the
//      call that reports it already created it)
//   2. cuBLAS handles
//   3. what a cudaMalloc/cudaFree round trip leaves behind in the driver allocator
//   4. the driver-side cost of an instantiated CUDA graph
// Build:
//   nvcc -O2 -o 012-cuda-floor 012-cuda-floor.cu -lcublas -lnvidia-ml \
//        -I/opt/cuda/targets/x86_64-linux/include
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "cuda error at %d: %s\n", __LINE__, cudaGetErrorString(e_));          \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static const double MiB = 1024.0 * 1024.0;

static nvmlDevice_t g_dev;

    // bytes in use on the GPU. run this with nothing else on the GPU (service
    // stopped): the per-process query is unreliable on this driver, so the reading is
    // device-wide and equals ours only if we are alone on the card.
static double self_mib(void) {
    static bool once = false;
    if (!once) {
        unsigned int n = 0;
        const nvmlReturn_t r = nvmlDeviceGetComputeRunningProcesses(g_dev, &n, nullptr);
    fprintf(stderr, "# nvmlDeviceGetComputeRunningProcesses: %s (n=%u)\n", nvmlErrorString(r), n);
        once = true;
    }
    nvmlMemory_t m;
    if (nvmlDeviceGetMemoryInfo(g_dev, &m) != NVML_SUCCESS) {
        return -1;
    }
    return m.used / MiB;
}

static double free_mib(void) {
    size_t f = 0, t = 0;
    CK(cudaMemGetInfo(&f, &t));
    return f / MiB;
}

__global__ void nop(float * p) {
    if (p) {
        p[threadIdx.x] = 0.f;
    }
}

int main(int argc, char ** argv) {
    const int n_graphs = argc > 1 ? atoi(argv[1]) : 8;
    const int n_nodes  = argc > 2 ? atoi(argv[2]) : 200;
    const size_t alloc_gb = argc > 3 ? (size_t) atoi(argv[3]) : 12;

    if (nvmlInit() != NVML_SUCCESS) {
        fprintf(stderr, "nvmlInit failed\n");
        return 1;
    }
    if (nvmlDeviceGetHandleByIndex(0, &g_dev) != NVML_SUCCESS) {
        fprintf(stderr, "no GPU 0\n");
        return 1;
    }

    const double p0 = self_mib();
    CK(cudaFree(0));
    const double p1 = self_mib();
    printf("cuda context      : %8.1f MiB (self %.1f -> %.1f)\n", p1 - p0, p0, p1);

    cublasHandle_t h[8] = {};
    double prev = p1;
    for (int i = 0; i < 8; i++) {
        if (cublasCreate(&h[i]) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed at %d\n", i);
            return 1;
        }
        const double now = self_mib();
        printf("cublas handle %d    : %8.1f MiB (cumulative %.1f)\n", i + 1, now - prev, now - p1);
        prev = now;
    }
    const double p2 = self_mib();
    printf("8 cublas handles  : %8.1f MiB\n", p2 - p1);

    float * buf = nullptr;
    CK(cudaMalloc(&buf, 4096));
    const double p3 = self_mib();
    printf("cudaMalloc 4 KiB  : %8.1f MiB (allocation granularity)\n", p3 - p2);

    // big round trip: does the driver give the bytes back to the process ledger?
    float * big = nullptr;
    CK(cudaMalloc(&big, (size_t) alloc_gb * 1024 * 1024 * 1024));
    CK(cudaMemset(big, 0, (size_t) alloc_gb * 1024 * 1024 * 1024));
    CK(cudaDeviceSynchronize());
    const double p4 = self_mib();
    CK(cudaFree(big));
    CK(cudaDeviceSynchronize());
    const double p5 = self_mib();
    printf("cudaMalloc/Free %zu GiB: peak %.1f, after free %.1f, sticky %.1f MiB\n",
           alloc_gb, p4 - p3, p5 - p3, p5 - p3);

    cudaStream_t s = nullptr;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const double p6 = self_mib();
    for (int g = 0; g < n_graphs; g++) {
        cudaGraph_t graph = nullptr;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        for (int i = 0; i < n_nodes; i++) {
            nop<<<1, 32, 0, s>>>(buf);
        }
        CK(cudaStreamEndCapture(s, &graph));
        cudaGraphExec_t inst = nullptr;
        CK(cudaGraphInstantiate(&inst, graph, NULL, NULL, 0));
        CK(cudaGraphDestroy(graph));
        printf("graph %2d (%d nodes): self = %8.1f MiB\n", g, n_nodes, self_mib());
    }
    const double p7 = self_mib();
    printf("%d graphs x %d nodes: %8.1f MiB (%.2f MiB per graph)\n", n_graphs, n_nodes, p7 - p6,
           (p7 - p6) / n_graphs);
    printf("free (cudaMemGetInfo) at end: %.1f MiB\n", free_mib());

    CK(cudaFree(buf));
    return 0;
}
