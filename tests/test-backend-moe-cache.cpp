#include "ggml-backend-moe-cache.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char * message) {
    throw std::runtime_error(message);
}

void require(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

void write_all(int fd, const void * data, size_t size) {
    const uint8_t * current = static_cast<const uint8_t *>(data);
    while (size != 0) {
        const ssize_t written = write(fd, current, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        require(written > 0, "failed to write temporary expert source");
        current += written;
        size -= static_cast<size_t>(written);
    }
}

struct temp_source {
    std::string path;
    int         direct_fd = -1;

    ~temp_source() {
        if (direct_fd >= 0) {
            close(direct_fd);
        }
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

temp_source make_source(const std::vector<float> & weights) {
    char      name[] = "test-backend-moe-cache-XXXXXX";
    const int fd     = mkstemp(name);
    require(fd >= 0, "failed to create temporary expert source");
    write_all(fd, weights.data(), weights.size() * sizeof(float));
    require(fsync(fd) == 0, "failed to flush temporary expert source");
    require(close(fd) == 0, "failed to close temporary expert source");

    temp_source result;
    result.path      = name;
    result.direct_fd = open(name, O_RDONLY | O_DIRECT | O_CLOEXEC);
    require(result.direct_fd >= 0, "temporary filesystem does not support O_DIRECT");
    return result;
}

void replace_path(const std::string & path, size_t size) {
    require(unlink(path.c_str()) == 0, "failed to unlink temporary expert source");
    const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    require(fd >= 0, "failed to replace temporary expert source path");
    std::vector<uint8_t> zeros(size, 0);
    write_all(fd, zeros.data(), zeros.size());
    require(fsync(fd) == 0, "failed to flush replacement expert source");
    require(close(fd) == 0, "failed to close replacement expert source");
}

void compare_output(ggml_backend_t             backend,
                    ggml_tensor *              output,
                    const std::vector<float> & weights,
                    const std::vector<float> & input,
                    int                        expert,
                    int64_t                    k,
                    int64_t                    m) {
    std::vector<float> actual(static_cast<size_t>(m));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t row = 0; row < m; ++row) {
        float        expected = 0.0f;
        const size_t base     = (static_cast<size_t>(expert) * m + row) * k;
        for (int64_t column = 0; column < k; ++column) {
            expected += weights[base + column] * input[column];
        }
        const float tolerance = 2e-5f * std::max(1.0f, std::fabs(expected));
        if (std::fabs(actual[row] - expected) > tolerance) {
            std::fprintf(stderr, "expert %d row %lld: got %.9g, expected %.9g\n", expert, static_cast<long long>(row),
                         actual[row], expected);
            fail("cached MUL_MAT_ID output differs from the CPU reference");
        }
    }
    GGML_UNUSED(backend);
}

}  // namespace

int main() try {
#if !defined(__linux__) || !defined(GGML_MOE_CACHE)
    std::puts("MoE cache test skipped: feature is not built");
    return 0;
#else
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr) {
        std::puts("MoE cache test skipped: no GPU device");
        return 0;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg == nullptr || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
        std::puts("MoE cache test skipped: no CUDA device");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    require(backend != nullptr, "failed to initialize CUDA backend");
    using host_register_fn   = bool (*)(void *, size_t);
    using host_unregister_fn = bool (*)(void *);
    auto register_host       = reinterpret_cast<host_register_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_register_checked"));
    auto unregister_host = reinterpret_cast<host_unregister_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unregister_checked"));
    require(register_host != nullptr && unregister_host != nullptr, "checked CUDA pin hooks are absent");
    void * pin_probe = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(pin_probe != MAP_FAILED, "failed to allocate CUDA pin probe");
    const bool pin_supported = register_host(pin_probe, 4096);
    if (pin_supported) {
        require(unregister_host(pin_probe), "checked CUDA unpin probe failed");
        require(!unregister_host(pin_probe), "checked CUDA unpin accepted an unregistered range");
    }
    require(munmap(pin_probe, 4096) == 0, "failed to unmap CUDA pin probe");

    constexpr int64_t n_expert    = 67;
    constexpr int64_t k           = 250;
    constexpr int64_t m           = 3;
    constexpr size_t  expert_size = static_cast<size_t>(k * m) * sizeof(float);
    constexpr size_t  tensor_size = n_expert * expert_size;
    static_assert(expert_size == 3000, "test requires a non-aligned expert extent");

    std::vector<float> weights(tensor_size / sizeof(float));
    for (int expert = 0; expert < n_expert; ++expert) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t column = 0; column < k; ++column) {
                const size_t index = (static_cast<size_t>(expert) * m + row) * k + column;
                weights[index]     = 0.01f * (expert + 1) + 0.001f * row + 0.00001f * column;
            }
        }
    }
    std::vector<float> input(static_cast<size_t>(k));
    for (int64_t column = 0; column < k; ++column) {
        input[column] = 0.1f * static_cast<float>(column % 7 - 3);
    }

    temp_source file = make_source(weights);
    struct stat source_stat{};
    require(fstat(file.direct_fd, &source_stat) == 0, "failed to stat direct expert source");

    ggml_init_params context_params{
        /*.mem_size   =*/256 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(context_params);
    require(ctx != nullptr, "failed to initialize graph context");
    ggml_tensor * cached_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_expert);
    ggml_tensor * ids            = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * activations    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, 1);
    ggml_tensor * output         = ggml_mul_mat_id(ctx, cached_weights, activations, ids);
    ggml_cgraph * graph          = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(graph_buffer != nullptr, "failed to allocate CUDA graph tensors");
    ggml_backend_tensor_set(activations, input.data(), 0, input.size() * sizeof(float));

    ggml_tensor source_tensor = *cached_weights;
    source_tensor.buffer      = nullptr;
    source_tensor.data        = nullptr;

    ggml_moe_cache_source source{};
    source.tensor           = &source_tensor;
    source.tensor_name      = "test.experts";
    source.source_id        = 7;
    source.shard_index      = 0;
    source.tensor_offset    = 0;
    source.tensor_length    = tensor_size;
    source.shard_length     = tensor_size;
    source.source_device    = static_cast<uint64_t>(source_stat.st_dev);
    source.source_inode     = static_cast<uint64_t>(source_stat.st_ino);
    source.memory_alignment = 4096;
    source.offset_alignment = 4096;
    source.n_expert         = n_expert;
    source.expert_size      = expert_size;
    source.tensor_bytes     = tensor_size;
    source.fd               = file.direct_fd;

    ggml_moe_cache_config config{};
    config.mode                  = GGML_MOE_CACHE_MODE_AUTO;
    config.l1_bytes_per_device   = 2 * expert_size;
    config.l2_bytes              = 256ULL * 1024 * 1024;
    config.host_reserve_bytes    = 1024 * 1024;
    config.largest_expert_extent = expert_size;

    ggml_backend_t           backends[] = { backend };
    ggml_backend_moe_cache * cache      = ggml_backend_moe_cache_new(backends, 1);
    require(cache != nullptr, "failed to create MoE cache");

    ggml_moe_cache_source invalid_source = source;
    invalid_source.source_inode++;
    require(!ggml_backend_moe_cache_configure(cache, &config, &invalid_source, 1),
            "cache accepted a mismatched source inode");
    ggml_moe_cache_config reserve_config = config;
    reserve_config.host_reserve_bytes    = std::numeric_limits<size_t>::max();
    require(ggml_backend_moe_cache_configure(cache, &reserve_config, &source, 1),
            "failed to configure host-reserve rejection case");
    require(!ggml_backend_moe_cache_activate(cache), "cache crossed the configured host reserve");
    require(ggml_backend_moe_cache_consume_placement_invalidated(cache),
            "host-reserve rejection did not invalidate placement");

    ggml_moe_cache_config strict_config = config;
    strict_config.mode                  = GGML_MOE_CACHE_MODE_GENERATION;
    strict_config.strict                = true;
    strict_config.l2_bytes              = std::numeric_limits<size_t>::max();
    require(ggml_backend_moe_cache_configure(cache, &strict_config, &source, 1),
            "failed to configure strict budget rejection case");
    require(!ggml_backend_moe_cache_activate(cache), "strict cache accepted less than its requested L2 capacity");
    require(ggml_backend_moe_cache_consume_placement_invalidated(cache),
            "strict budget rejection did not invalidate placement");

    ggml_moe_cache_config automatic_config = config;
    automatic_config.l1_bytes_per_device   = 0;
    require(ggml_backend_moe_cache_configure(cache, &automatic_config, &source, 1),
            "failed to configure automatic MoE cache");
    require(!ggml_backend_moe_cache_activate(cache),
            "automatic cache accepted less than its minimum useful L1 capacity");
    require(ggml_backend_moe_cache_consume_placement_invalidated(cache),
            "automatic L1 rejection did not invalidate placement");
    require(ggml_backend_moe_cache_configure(cache, &config, &source, 1), "failed to configure MoE cache");

    require(close(file.direct_fd) == 0, "failed to close caller-owned direct descriptor");
    file.direct_fd = -1;
    replace_path(file.path, tensor_size);

    const bool activated = ggml_backend_moe_cache_activate(cache);
    if (!pin_supported) {
        require(!activated, "cache activated after checked CUDA registration was rejected");
        require(ggml_backend_moe_cache_consume_placement_invalidated(cache),
                "failed CUDA registration did not invalidate placement");
        ggml_backend_moe_cache_free(cache);
        ggml_backend_buffer_free(graph_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        std::puts("MoE cache integration skipped: CUDA read-only host registration is unsupported");
        return 0;
    }
    require(activated, "failed to activate MoE cache");
    ggml_backend_moe_cache_set_eligible(cache, true);

    auto run_expert = [&](int32_t expert) {
        const ggml_backend_moe_cache_result copy_result =
            ggml_backend_moe_cache_copy_experts(cache, backend, backend, &source_tensor, cached_weights, &expert, 1);
        require(copy_result == GGML_BACKEND_MOE_CACHE_HANDLED, "cache did not handle selected expert copy");
        ggml_backend_tensor_set(ids, &expert, 0, sizeof(expert));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA MUL_MAT_ID execution failed");
        compare_output(backend, output, weights, input, expert, k, m);
    };

    run_expert(0);
    require(ggml_backend_moe_cache_get_stats(cache).l3_read_count == 1, "first access did not read L3");
    run_expert(1);
    require(ggml_backend_moe_cache_get_stats(cache).l3_read_count == 2, "second expert did not read L3");
    run_expert(0);
    require(ggml_backend_moe_cache_get_stats(cache).l2_hits == 1, "evicted expert did not hit inclusive L2");
    run_expert(0);
    require(ggml_backend_moe_cache_get_stats(cache).l1_hits == 1, "resident expert did not hit L1");

    ggml_moe_cache_stats stats = ggml_backend_moe_cache_get_stats(cache);
    require(stats.l3_logical_bytes == 2 * (expert_size + std::min<size_t>(expert_size, 512)),
            "logical L3 accounting omitted MMQ padding");
    require(stats.l3_physical_bytes == 4096 + 8192, "physical L3 accounting omitted aligned read extents");
    require(stats.l1_evictions >= 2, "frequency admission did not evict constrained L1 entries");

    std::vector<int32_t> all_experts(static_cast<size_t>(n_expert));
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        all_experts[expert] = expert;
    }
    require(
        ggml_backend_moe_cache_copy_experts(cache, backend, backend, &source_tensor, cached_weights, all_experts.data(),
                                            all_experts.size()) == GGML_BACKEND_MOE_CACHE_HANDLED,
        "cache failed a read wave larger than the io_uring depth");
    int32_t final_expert = n_expert - 1;
    ggml_backend_tensor_set(ids, &final_expert, 0, sizeof(final_expert));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "CUDA MUL_MAT_ID execution after a large read wave failed");
    compare_output(backend, output, weights, input, final_expert, k, m);

    stats                    = ggml_backend_moe_cache_get_stats(cache);
    size_t expected_logical  = 0;
    size_t expected_physical = 0;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        const size_t logical_offset = static_cast<size_t>(expert) * expert_size;
        const size_t copy_size      = expert_size + (expert + 1 < n_expert ? std::min<size_t>(expert_size, 512) : 0);
        const size_t head           = logical_offset % 4096;
        expected_logical += copy_size;
        expected_physical += ((head + copy_size + 4095) / 4096) * 4096;
    }
    require(stats.l3_read_count == static_cast<uint64_t>(n_expert),
            "large read wave did not publish every expert exactly once");
    require(stats.l3_logical_bytes == expected_logical && stats.l3_physical_bytes == expected_physical,
            "large read wave accounting did not preserve aligned EOF reads");

    const uint64_t l1_hits_before_reset = stats.l1_hits;
    ggml_backend_moe_cache_reset(cache);
    run_expert(0);
    stats = ggml_backend_moe_cache_get_stats(cache);
    require(stats.l3_read_count == static_cast<uint64_t>(n_expert) && stats.l1_hits == l1_hits_before_reset + 1,
            "scheduler reset discarded ready cache entries");

    ggml_backend_moe_cache_reset_stats(cache);
    stats = ggml_backend_moe_cache_get_stats(cache);
    require(stats.l1_capacity == config.l1_bytes_per_device && stats.l2_capacity == config.l2_bytes,
            "counter reset discarded effective capacities");
    require(stats.l1_hits == 0 && stats.l2_hits == 0 && stats.l3_read_count == 0,
            "counter reset retained request counters");

    ggml_backend_moe_cache_free(cache);
    ggml_backend_buffer_free(graph_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts("MoE cache integration: OK");
    return 0;
#endif
} catch (const std::exception & exception) {
    std::fprintf(stderr, "MoE cache integration: %s\n", exception.what());
    return 1;
}
