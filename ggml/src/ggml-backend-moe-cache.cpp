#include "ggml-backend-moe-cache.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(GGML_MOE_CACHE)

#    include <fcntl.h>
#    include <liburing.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>

namespace {

constexpr size_t   MIB          = 1024 * 1024;
constexpr size_t   GIB          = 1024 * MIB;
constexpr size_t   L2_CHUNK_MIN = 256 * MIB;
constexpr unsigned RING_DEPTH   = 64;

using host_register_fn   = bool (*)(void *, size_t);
using host_unregister_fn = bool (*)(void *);

struct process_budget {
    std::mutex mutex;
    size_t     claimed = 0;
};

process_budget & budget_coordinator() {
    static process_budget budget;
    return budget;
}

bool add_checked(size_t a, size_t b, size_t & result) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

bool mul_checked(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool align_up_checked(size_t value, size_t alignment, size_t & result) {
    if (alignment == 0) {
        return false;
    }
    const size_t rem = value % alignment;
    return rem == 0 ? (result = value, true) : add_checked(value, alignment - rem, result);
}

size_t align_down(size_t value, size_t alignment) {
    return alignment == 0 ? value : value - value % alignment;
}

struct memory_info {
    size_t total     = 0;
    size_t available = 0;
};

memory_info read_memory_info() {
    memory_info result;
    FILE *      file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) {
        return result;
    }

    char               key[64];
    unsigned long long kib = 0;
    char               unit[16];
    while (std::fscanf(file, "%63s %llu %15s", key, &kib, unit) == 3) {
        size_t bytes = kib > std::numeric_limits<size_t>::max() / 1024 ? std::numeric_limits<size_t>::max() :
                                                                         static_cast<size_t>(kib) * 1024;
        if (std::strcmp(key, "MemTotal:") == 0) {
            result.total = bytes;
        } else if (std::strcmp(key, "MemAvailable:") == 0) {
            result.available = bytes;
        }
        if (result.total != 0 && result.available != 0) {
            break;
        }
    }
    std::fclose(file);
    return result;
}

size_t claim_host_bytes(size_t requested, size_t reserve) {
    process_budget &            coordinator = budget_coordinator();
    std::lock_guard<std::mutex> lock(coordinator.mutex);
    const memory_info           memory = read_memory_info();
    if (memory.available <= reserve || coordinator.claimed >= memory.available - reserve) {
        return 0;
    }
    const size_t safe    = memory.available - reserve - coordinator.claimed;
    const size_t claimed = std::min(requested, safe);
    coordinator.claimed += claimed;
    return claimed;
}

void release_host_bytes(size_t bytes) {
    process_budget &            coordinator = budget_coordinator();
    std::lock_guard<std::mutex> lock(coordinator.mutex);
    coordinator.claimed = bytes > coordinator.claimed ? 0 : coordinator.claimed - bytes;
}

struct cache_key {
    int      device = -1;
    uint32_t source = 0;
    int32_t  expert = -1;

    bool operator==(const cache_key & other) const {
        return device == other.device && source == other.source && expert == other.expert;
    }
};

struct cache_key_hash {
    size_t operator()(const cache_key & key) const {
        size_t value = static_cast<size_t>(key.source) * 0x9e3779b1u;
        value ^= static_cast<size_t>(key.expert + 1) * 0x85ebca6bu;
        value ^= static_cast<size_t>(key.device + 1) * 0xc2b2ae35u;
        return value;
    }
};

struct owned_source {
    ggml_moe_cache_source desc = {};
    std::string           name;
    int                   fd = -1;

    owned_source()                                 = default;
    owned_source(const owned_source &)             = delete;
    owned_source & operator=(const owned_source &) = delete;

    owned_source(owned_source && other) noexcept : desc(other.desc), name(std::move(other.name)), fd(other.fd) {
        other.fd = -1;
    }

    owned_source & operator=(owned_source && other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                close(fd);
            }
            desc     = other.desc;
            name     = std::move(other.name);
            fd       = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~owned_source() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

struct l2_chunk {
    uint8_t * data = nullptr;
    size_t    size = 0;
};

enum class l2_state {
    filling,
    ready,
    failed,
};

struct l2_entry {
    cache_key key;
    l2_state  state         = l2_state::filling;
    size_t    chunk         = 0;
    size_t    offset        = 0;
    size_t    physical_size = 0;
    size_t    data_offset   = 0;
    size_t    logical_size  = 0;
    uint64_t  last_use      = 0;
    uint32_t  pins          = 0;
};

enum class l1_state {
    copying,
    ready,
};

struct l1_entry {
    cache_key key;
    l1_state  state    = l1_state::copying;
    size_t    offset   = 0;
    size_t    size     = 0;
    uint64_t  last_use = 0;
    uint32_t  pins     = 0;
};

struct device_cache {
    ggml_backend_t        backend   = nullptr;
    ggml_backend_buffer_t buffer    = nullptr;
    ggml_backend_event_t  event     = nullptr;
    size_t                capacity  = 0;
    size_t                alignment = 1;
    bool                  pending   = false;
    std::list<l1_entry>   entries;
};

struct read_request {
    l2_entry *           entry      = nullptr;
    const owned_source * source     = nullptr;
    uint64_t             offset     = 0;
    size_t               size       = 0;
    uint64_t             generation = 0;
    bool                 cqe_seen   = false;
    int                  result     = 0;
};

struct access_record {
    cache_key            key;
    const owned_source * source        = nullptr;
    int32_t              expert        = 0;
    size_t               tensor_offset = 0;
    size_t               copy_size     = 0;
    l2_entry *           l2            = nullptr;
    l1_entry *           l1            = nullptr;
    bool                 copied        = false;
    bool                 l3_fill       = false;
};

void make_byte_tensor(ggml_tensor & tensor, ggml_backend_buffer_t buffer, void * data, size_t size) {
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type   = GGML_TYPE_I8;
    tensor.buffer = buffer;
    tensor.data   = data;
    tensor.ne[0]  = static_cast<int64_t>(size);
    tensor.ne[1]  = 1;
    tensor.ne[2]  = 1;
    tensor.ne[3]  = 1;
    tensor.nb[0]  = 1;
    tensor.nb[1]  = size;
    tensor.nb[2]  = size;
    tensor.nb[3]  = size;
}

void copy_device_range_async(ggml_backend_t        backend,
                             ggml_backend_buffer_t source_buffer,
                             void *                source,
                             ggml_backend_buffer_t destination_buffer,
                             void *                destination,
                             size_t                size) {
    ggml_tensor source_tensor;
    ggml_tensor destination_tensor;
    make_byte_tensor(source_tensor, source_buffer, source, size);
    make_byte_tensor(destination_tensor, destination_buffer, destination, size);
    ggml_backend_tensor_copy_async(backend, backend, &source_tensor, &destination_tensor);
}

}  // namespace

struct ggml_backend_moe_cache {
    ggml_moe_cache_config                                   config = {};
    std::vector<ggml_backend_t>                             backends;
    std::vector<owned_source>                               sources;
    std::vector<device_cache>                               devices;
    std::vector<l2_chunk>                                   chunks;
    std::list<l2_entry>                                     l2_entries;
    std::unordered_map<cache_key, uint64_t, cache_key_hash> frequencies;
    ggml_moe_cache_stats                                    stats                 = {};
    io_uring                                                ring                  = {};
    bool                                                    ring_ready            = false;
    bool                                                    configured            = false;
    bool                                                    active                = false;
    bool                                                    override_enabled      = false;
    bool                                                    eligible              = false;
    bool                                                    placement_invalidated = false;
    bool                                                    admissions            = false;
    size_t                                                  claimed_bytes         = 0;
    size_t                                                  host_reserve          = 0;
    size_t                                                  l2_alignment          = 1;
    uint64_t                                                generation            = 0;
    uint64_t                                                clock                 = 0;
    uint64_t                                                eligible_graphs       = 0;
    std::string                                             error;

    int backend_index(ggml_backend_t backend) const {
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].backend == backend) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    const owned_source * find_source(const ggml_tensor * tensor) const {
        for (const owned_source & source : sources) {
            if (source.desc.tensor == tensor ||
                (tensor != nullptr && source.desc.tensor != nullptr && source.desc.tensor->data == tensor->data)) {
                return &source;
            }
        }
        return nullptr;
    }

    l2_entry * find_l2(const cache_key & key) {
        for (l2_entry & entry : l2_entries) {
            if (entry.key == key && entry.state == l2_state::ready) {
                return &entry;
            }
        }
        return nullptr;
    }

    l1_entry * find_l1(const cache_key & key) {
        if (key.device < 0 || key.device >= static_cast<int>(devices.size())) {
            return nullptr;
        }
        for (l1_entry & entry : devices[key.device].entries) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    bool l1_contains(const cache_key & key) const {
        if (key.device < 0 || key.device >= static_cast<int>(devices.size())) {
            return false;
        }
        for (const l1_entry & entry : devices[key.device].entries) {
            if (entry.key == key) {
                return true;
            }
        }
        return false;
    }

    void set_error(const std::string & message) {
        error = message;
        GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
    }

    void sync_device(device_cache & device) {
        if (!device.pending) {
            return;
        }
        ggml_backend_event_synchronize(device.event);
        device.pending = false;
        for (l1_entry & entry : device.entries) {
            if (entry.state == l1_state::copying) {
                entry.state = l1_state::ready;
            }
            entry.pins = 0;
        }
        for (l2_entry & entry : l2_entries) {
            if (entry.key.device == backend_index(device.backend)) {
                entry.pins = 0;
            }
        }
    }

    void barrier() {
        for (device_cache & device : devices) {
            sync_device(device);
        }
    }

    void teardown_storage() {
        barrier();
        l2_entries.clear();
        for (device_cache & device : devices) {
            device.entries.clear();
            if (device.buffer != nullptr) {
                ggml_backend_buffer_free(device.buffer);
                device.buffer = nullptr;
            }
            if (device.event != nullptr) {
                ggml_backend_event_free(device.event);
                device.event = nullptr;
            }
        }
        devices.clear();

        host_unregister_fn unregister_host = nullptr;
        for (ggml_backend_t backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = dev == nullptr ? nullptr : ggml_backend_dev_backend_reg(dev);
            if (reg != nullptr) {
                unregister_host = reinterpret_cast<host_unregister_fn>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unregister_checked"));
                if (unregister_host != nullptr) {
                    break;
                }
            }
        }

        if (ring_ready) {
            io_uring_queue_exit(&ring);
            ring_ready = false;
        }

        for (l2_chunk & chunk : chunks) {
            if (chunk.data == nullptr) {
                continue;
            }
            std::memset(chunk.data, 0, chunk.size);
            const bool unregistered = unregister_host != nullptr && unregister_host(chunk.data);
            if (unregistered) {
                munmap(chunk.data, chunk.size);
            } else {
                GGML_LOG_ERROR("%s: quarantining %zu bytes after CUDA host unregistration failed\n", __func__,
                               chunk.size);
            }
            chunk.data = nullptr;
        }
        chunks.clear();

        release_host_bytes(claimed_bytes);
        claimed_bytes     = 0;
        stats.l1_capacity = 0;
        stats.l2_capacity = 0;
        active            = false;
        admissions        = false;
    }

    bool reserve_l2_range(size_t size, size_t alignment, size_t & chunk_index, size_t & offset) {
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            std::vector<std::pair<size_t, size_t>> intervals;
            for (const l2_entry & entry : l2_entries) {
                if (entry.chunk == ci && entry.state != l2_state::failed) {
                    intervals.emplace_back(entry.offset, entry.offset + entry.physical_size);
                }
            }
            std::sort(intervals.begin(), intervals.end());
            size_t cursor = 0;
            for (const auto & interval : intervals) {
                size_t aligned = 0;
                if (!align_up_checked(cursor, alignment, aligned)) {
                    return false;
                }
                if (aligned <= interval.first && size <= interval.first - aligned) {
                    chunk_index = ci;
                    offset      = aligned;
                    return true;
                }
                cursor = std::max(cursor, interval.second);
            }
            size_t aligned = 0;
            if (!align_up_checked(cursor, alignment, aligned)) {
                return false;
            }
            if (aligned <= chunks[ci].size && size <= chunks[ci].size - aligned) {
                chunk_index = ci;
                offset      = aligned;
                return true;
            }
        }
        return false;
    }

    bool evict_l1(int device_index) {
        device_cache & device           = devices[device_index];
        auto           victim           = device.entries.end();
        uint64_t       victim_frequency = std::numeric_limits<uint64_t>::max();
        for (auto it = device.entries.begin(); it != device.entries.end(); ++it) {
            if (it->pins != 0 || it->state != l1_state::ready) {
                continue;
            }
            const uint64_t frequency = frequencies[it->key];
            if (victim == device.entries.end() || frequency < victim_frequency ||
                (frequency == victim_frequency && it->last_use < victim->last_use)) {
                victim           = it;
                victim_frequency = frequency;
            }
        }
        if (victim == device.entries.end()) {
            return false;
        }
        device.entries.erase(victim);
        stats.l1_evictions++;
        return true;
    }

    bool evict_l2() {
        auto victim = l2_entries.end();
        for (auto it = l2_entries.begin(); it != l2_entries.end(); ++it) {
            if (it->state != l2_state::ready || it->pins != 0 || l1_contains(it->key)) {
                continue;
            }
            if (victim == l2_entries.end() || it->last_use < victim->last_use) {
                victim = it;
            }
        }
        if (victim == l2_entries.end()) {
            for (int device = 0; device < static_cast<int>(devices.size()); ++device) {
                if (evict_l1(device)) {
                    return evict_l2();
                }
            }
            return false;
        }
        l2_entries.erase(victim);
        stats.l2_evictions++;
        return true;
    }

    l2_entry * allocate_l2(const cache_key & key, size_t size, size_t alignment) {
        size_t chunk_index = 0;
        size_t offset      = 0;
        while (!reserve_l2_range(size, alignment, chunk_index, offset)) {
            if (!evict_l2()) {
                return nullptr;
            }
        }
        l2_entries.push_back({ key, l2_state::filling, chunk_index, offset, size, 0, 0, ++clock, 1 });
        return &l2_entries.back();
    }

    bool reserve_l1_range(device_cache & device, size_t size, size_t & offset) {
        std::vector<std::pair<size_t, size_t>> intervals;
        for (const l1_entry & entry : device.entries) {
            intervals.emplace_back(entry.offset, entry.offset + entry.size);
        }
        std::sort(intervals.begin(), intervals.end());
        size_t cursor = 0;
        for (const auto & interval : intervals) {
            size_t aligned = 0;
            if (!align_up_checked(cursor, device.alignment, aligned)) {
                return false;
            }
            if (aligned <= interval.first && size <= interval.first - aligned) {
                offset = aligned;
                return true;
            }
            cursor = std::max(cursor, interval.second);
        }
        size_t aligned = 0;
        if (!align_up_checked(cursor, device.alignment, aligned)) {
            return false;
        }
        if (aligned <= device.capacity && size <= device.capacity - aligned) {
            offset = aligned;
            return true;
        }
        return false;
    }

    l1_entry * admit_l1(const access_record & access, ggml_tensor * input_cpy) {
        device_cache & device = devices[access.key.device];
        if (device.capacity < access.copy_size || find_l1(access.key) != nullptr) {
            return nullptr;
        }

        const uint64_t score        = frequencies[access.key];
        auto           victim       = device.entries.end();
        uint64_t       victim_score = std::numeric_limits<uint64_t>::max();
        for (auto it = device.entries.begin(); it != device.entries.end(); ++it) {
            if (it->pins != 0 || it->state != l1_state::ready) {
                continue;
            }
            const uint64_t frequency = frequencies[it->key];
            if (victim == device.entries.end() || frequency < victim_score ||
                (frequency == victim_score && it->last_use < victim->last_use)) {
                victim       = it;
                victim_score = frequency;
            }
        }

        size_t offset = 0;
        while (!reserve_l1_range(device, access.copy_size, offset)) {
            if (victim == device.entries.end() || score < victim_score) {
                return nullptr;
            }
            device.entries.erase(victim);
            stats.l1_evictions++;
            victim       = device.entries.end();
            victim_score = std::numeric_limits<uint64_t>::max();
            for (auto it = device.entries.begin(); it != device.entries.end(); ++it) {
                if (it->pins == 0 && it->state == l1_state::ready) {
                    const uint64_t frequency = frequencies[it->key];
                    if (victim == device.entries.end() || frequency < victim_score ||
                        (frequency == victim_score && it->last_use < victim->last_use)) {
                        victim       = it;
                        victim_score = frequency;
                    }
                }
            }
        }

        device.entries.push_back({ access.key, l1_state::copying, offset, access.copy_size, ++clock, 1 });
        l1_entry & entry       = device.entries.back();
        uint8_t *  source      = static_cast<uint8_t *>(input_cpy->data) + access.tensor_offset;
        uint8_t *  destination = static_cast<uint8_t *>(ggml_backend_buffer_get_base(device.buffer)) + offset;
        copy_device_range_async(device.backend, input_cpy->buffer, source, device.buffer, destination,
                                access.copy_size);
        return &entry;
    }

    bool submit_reads(std::vector<read_request> & requests) {
        if (requests.empty()) {
            return true;
        }
        const int64_t start     = ggml_time_us();
        size_t        prepared  = 0;
        size_t        submitted = 0;
        size_t        seen      = 0;
        bool          valid     = true;

        auto reap = [&](size_t target) {
            while (seen < target) {
                io_uring_cqe * cqe = nullptr;
                int            rc;
                do {
                    rc = io_uring_wait_cqe(&ring, &cqe);
                } while (rc == -EINTR);
                if (rc < 0 || cqe == nullptr) {
                    set_error("io_uring completion wait failed");
                    return false;
                }

                read_request * request = static_cast<read_request *>(io_uring_cqe_get_data(cqe));
                if (request == nullptr || request < requests.data() || request >= requests.data() + requests.size() ||
                    request->cqe_seen) {
                    set_error("io_uring returned an invalid or duplicate completion");
                    valid = false;
                } else {
                    request->cqe_seen = true;
                    request->result   = cqe->res;
                    if (request->generation != generation) {
                        request->result = -ESTALE;
                        set_error("io_uring returned a completion for the wrong cache generation");
                        valid = false;
                    }
                }
                io_uring_cqe_seen(&ring, cqe);
                seen++;
            }
            return true;
        };

        bool submission_ok = true;
        while (prepared < requests.size()) {
            io_uring_sqe * sqe = io_uring_get_sqe(&ring);
            if (sqe == nullptr) {
                const int rc = io_uring_submit(&ring);
                if (rc == -EINTR) {
                    continue;
                }
                if (rc <= 0) {
                    set_error("io_uring submission failed while the queue was full");
                    submission_ok = false;
                    break;
                }
                submitted += static_cast<size_t>(rc);
                continue;
            }
            read_request & request     = requests[prepared++];
            uint8_t *      destination = chunks[request.entry->chunk].data + request.entry->offset;
            io_uring_prep_read(sqe, request.source->fd, destination, request.size, request.offset);
            io_uring_sqe_set_data(sqe, &request);
        }
        while (submission_ok && submitted < requests.size()) {
            const int rc = io_uring_submit(&ring);
            if (rc == -EINTR) {
                continue;
            }
            if (rc <= 0) {
                set_error("io_uring submitted only part of a read wave");
                submission_ok = false;
                break;
            }
            submitted += static_cast<size_t>(rc);
        }

        valid = valid && submission_ok;
        if (!reap(submitted)) {
            valid = false;
            if (ring_ready) {
                io_uring_queue_exit(&ring);
                ring_ready = false;
            }
        } else if (!submission_ok) {
            io_uring_queue_exit(&ring);
            ring_ready = false;
        }

        for (read_request & request : requests) {
            if (!request.cqe_seen || request.result < 0) {
                request.entry->state = l2_state::failed;
                valid                = false;
                continue;
            }
            const size_t result    = static_cast<size_t>(request.result);
            const bool   full      = result == request.size;
            const bool   valid_eof = result < request.size && request.offset <= request.source->desc.shard_length &&
                                     result == request.source->desc.shard_length - request.offset &&
                                     request.entry->data_offset <= result &&
                                     request.entry->logical_size <= result - request.entry->data_offset;
            if ((!full && !valid_eof) || request.entry->data_offset > result ||
                request.entry->logical_size > result - request.entry->data_offset) {
                request.entry->state = l2_state::failed;
                valid                = false;
                continue;
            }
            request.entry->state = l2_state::ready;
            stats.l3_read_count++;
            stats.l3_logical_bytes += request.entry->logical_size;
            stats.l3_physical_bytes += request.size;
        }
        stats.l3_wait_us += static_cast<uint64_t>(std::max<int64_t>(0, ggml_time_us() - start));
        if (!valid && error.empty()) {
            set_error("a direct expert read failed validation");
        }
        return valid;
    }

    void record_pending(device_cache & device) {
        ggml_backend_event_record(device.event, device.backend);
        device.pending = true;
    }

    void age_frequencies() {
        eligible_graphs++;
        if (eligible_graphs % 128 != 0) {
            return;
        }
        for (auto & item : frequencies) {
            item.second >>= 1;
        }
    }

    bool ensure_host_margin() {
        const memory_info memory = read_memory_info();
        if (memory.available == 0) {
            return true;
        }
        if (memory.available >= host_reserve) {
            admissions = active;
            return true;
        }
        barrier();
        while (!chunks.empty()) {
            const size_t tail   = chunks.size() - 1;
            const bool   pinned = std::any_of(l2_entries.begin(), l2_entries.end(), [tail](const l2_entry & entry) {
                return entry.chunk == tail && entry.pins != 0;
            });
            if (pinned) {
                break;
            }
            for (auto it = l2_entries.begin(); it != l2_entries.end();) {
                if (it->chunk != tail) {
                    ++it;
                    continue;
                }
                if (it->key.device >= 0 && it->key.device < static_cast<int>(devices.size())) {
                    device_cache & device = devices[it->key.device];
                    for (auto l1 = device.entries.begin(); l1 != device.entries.end();) {
                        if (l1->key == it->key) {
                            l1 = device.entries.erase(l1);
                            stats.l1_evictions++;
                        } else {
                            ++l1;
                        }
                    }
                }
                it = l2_entries.erase(it);
                stats.l2_evictions++;
            }
            l2_chunk chunk = chunks.back();
            chunks.pop_back();
            host_unregister_fn unregister_host = nullptr;
            for (ggml_backend_t backend : backends) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backend);
                ggml_backend_reg_t reg = dev == nullptr ? nullptr : ggml_backend_dev_backend_reg(dev);
                if (reg != nullptr) {
                    unregister_host = reinterpret_cast<host_unregister_fn>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unregister_checked"));
                    if (unregister_host != nullptr) {
                        break;
                    }
                }
            }
            std::memset(chunk.data, 0, chunk.size);
            if (unregister_host != nullptr && unregister_host(chunk.data)) {
                munmap(chunk.data, chunk.size);
            } else {
                GGML_LOG_ERROR("%s: quarantining a tail chunk after CUDA host unregistration failed\n", __func__);
            }
            claimed_bytes -= chunk.size;
            stats.l2_capacity -= chunk.size;
            release_host_bytes(chunk.size);
            if (read_memory_info().available >= host_reserve) {
                return true;
            }
        }
        admissions = false;
        return false;
    }

    void fail_runtime(const std::string & message) {
        set_error(message);
        for (device_cache & device : devices) {
            ggml_backend_synchronize(device.backend);
            device.pending = false;
            for (l1_entry & entry : device.entries) {
                entry.pins = 0;
            }
        }
        for (l2_entry & entry : l2_entries) {
            entry.pins = 0;
        }
        active                = false;
        admissions            = false;
        override_enabled      = false;
        placement_invalidated = true;
    }
};

static bool allocate_aligned_mapping(size_t size, size_t alignment, uint8_t *& result) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    alignment         = std::max(alignment, page);
    size_t total      = 0;
    if (!add_checked(size, alignment, total) || !align_up_checked(total, page, total)) {
        return false;
    }
    void * raw = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return false;
    }
    const uintptr_t base    = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t rem     = base % alignment;
    const uintptr_t aligned = rem == 0 ? base : base + alignment - rem;
    const size_t    prefix  = aligned - base;
    const size_t    suffix  = total - prefix - size;
    if (prefix != 0) {
        munmap(raw, prefix);
    }
    if (suffix != 0) {
        munmap(reinterpret_cast<void *>(aligned + size), suffix);
    }
    result = reinterpret_cast<uint8_t *>(aligned);
    return true;
}

ggml_backend_moe_cache * ggml_backend_moe_cache_new(ggml_backend_t * backends, int n_backends) {
    auto * cache = new ggml_backend_moe_cache;
    cache->backends.assign(backends, backends + n_backends);
    return cache;
}

void ggml_backend_moe_cache_free(ggml_backend_moe_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    cache->teardown_storage();
    delete cache;
}

void ggml_backend_moe_cache_reset(ggml_backend_moe_cache * cache) {
    if (cache != nullptr) {
        cache->barrier();
    }
}

bool ggml_backend_moe_cache_configure(ggml_backend_moe_cache *      cache,
                                      const ggml_moe_cache_config * config,
                                      const ggml_moe_cache_source * sources,
                                      size_t                        n_sources) {
    if (cache == nullptr || config == nullptr) {
        return false;
    }
    cache->teardown_storage();
    cache->sources.clear();
    cache->frequencies.clear();
    cache->stats = {};
    cache->error.clear();
    cache->placement_invalidated = false;
    cache->configured            = false;
    cache->override_enabled      = false;
    cache->config                = *config;

    if (config->mode == GGML_MOE_CACHE_MODE_OFF) {
        cache->configured = true;
        return true;
    }
    if (sources == nullptr || n_sources == 0) {
        cache->set_error("no direct expert sources were configured");
        return false;
    }

    cache->sources.reserve(n_sources);
    for (size_t i = 0; i < n_sources; ++i) {
        const ggml_moe_cache_source & source = sources[i];
        if (source.tensor == nullptr || source.tensor_name == nullptr || source.fd < 0 || source.n_expert <= 0 ||
            source.expert_size == 0 || source.tensor_bytes == 0 || source.memory_alignment == 0 ||
            source.offset_alignment == 0 || (source.memory_alignment & (source.memory_alignment - 1)) != 0 ||
            config->largest_expert_extent == 0 || source.tensor_offset > source.shard_length ||
            source.tensor_length > source.shard_length - source.tensor_offset) {
            cache->set_error("an expert source descriptor is incomplete or out of bounds");
            cache->sources.clear();
            return false;
        }
        size_t packed = 0;
        if (!mul_checked(source.expert_size, static_cast<size_t>(source.n_expert), packed) ||
            packed > source.tensor_length) {
            cache->set_error("an expert source has an invalid packed extent");
            cache->sources.clear();
            return false;
        }
        const int fd = fcntl(source.fd, F_DUPFD_CLOEXEC, 0);
        if (fd < 0) {
            cache->set_error("failed to duplicate a direct expert descriptor");
            cache->sources.clear();
            return false;
        }
        struct stat st    = {};
        const int   flags = fcntl(fd, F_GETFL);
        if (fstat(fd, &st) != 0 || flags < 0 || (flags & O_ACCMODE) != O_RDONLY || (flags & O_DIRECT) == 0 ||
            static_cast<uint64_t>(st.st_dev) != source.source_device ||
            static_cast<uint64_t>(st.st_ino) != source.source_inode ||
            static_cast<uint64_t>(st.st_size) != source.shard_length) {
            close(fd);
            cache->set_error("a direct expert descriptor no longer matches its mapped source");
            cache->sources.clear();
            return false;
        }
        owned_source owned;
        owned.desc = source;
        owned.name = source.tensor_name;
        owned.fd   = fd;
        cache->sources.emplace_back(std::move(owned));
    }

    cache->configured       = true;
    cache->override_enabled = true;
    return true;
}

bool ggml_backend_moe_cache_activate(ggml_backend_moe_cache * cache) {
    if (cache == nullptr || !cache->configured || cache->config.mode == GGML_MOE_CACHE_MODE_OFF) {
        return cache != nullptr && cache->configured;
    }
    cache->teardown_storage();
    cache->error.clear();
    cache->override_enabled = true;

    host_register_fn   register_host   = nullptr;
    host_unregister_fn unregister_host = nullptr;
    for (ggml_backend_t backend : cache->backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev == nullptr ? nullptr : ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
            continue;
        }
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.events) {
            cache->set_error("a selected CUDA device does not support events");
            cache->placement_invalidated = true;
            cache->override_enabled      = false;
            return false;
        }
        register_host = reinterpret_cast<host_register_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_register_checked"));
        unregister_host = reinterpret_cast<host_unregister_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unregister_checked"));
        if (register_host == nullptr || unregister_host == nullptr) {
            cache->set_error("the CUDA backend does not expose checked host registration");
            cache->placement_invalidated = true;
            cache->override_enabled      = false;
            return false;
        }
        device_cache device;
        device.backend   = backend;
        device.event     = ggml_backend_event_new(dev);
        device.alignment = ggml_backend_get_alignment(backend);
        if (device.event == nullptr) {
            cache->set_error("failed to allocate a cache-owned CUDA event");
            cache->placement_invalidated = true;
            cache->override_enabled      = false;
            cache->teardown_storage();
            return false;
        }
        cache->devices.emplace_back(std::move(device));
    }
    if (cache->devices.empty()) {
        cache->set_error("no CUDA backend is available for the MoE cache");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        return false;
    }

    cache->l2_alignment     = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_t largest_physical = cache->config.largest_expert_extent;
    for (const owned_source & source : cache->sources) {
        cache->l2_alignment = std::max(cache->l2_alignment, source.desc.memory_alignment);
        size_t padded       = 0;
        if (!add_checked(source.desc.expert_size, std::min<size_t>(source.desc.expert_size, 512), padded) ||
            !add_checked(padded, source.desc.offset_alignment - 1, padded) ||
            !align_up_checked(padded, source.desc.offset_alignment, padded)) {
            cache->set_error("expert direct-I/O extent overflows size_t");
            cache->placement_invalidated = true;
            cache->override_enabled      = false;
            cache->teardown_storage();
            return false;
        }
        largest_physical = std::max(largest_physical, padded);
    }
    size_t chunk_size = 0;
    if (!align_up_checked(std::max(L2_CHUNK_MIN, largest_physical), cache->l2_alignment, chunk_size)) {
        cache->set_error("L2 chunk size overflows size_t");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        cache->teardown_storage();
        return false;
    }
    size_t twice_largest_physical = 0;
    if (!mul_checked(largest_physical, 2, twice_largest_physical)) {
        cache->set_error("the minimum L2 capacity overflows size_t");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        cache->teardown_storage();
        return false;
    }
    const size_t min_l2 = std::max<size_t>(L2_CHUNK_MIN, twice_largest_physical);

    const memory_info memory = read_memory_info();
    cache->host_reserve      = cache->config.host_reserve_bytes != 0 ? cache->config.host_reserve_bytes :
                                                                       std::max<size_t>(8 * GIB, memory.total / 10);
    const size_t desired_l2 = cache->config.l2_bytes == 0 ? std::numeric_limits<size_t>::max() : cache->config.l2_bytes;
    const size_t claimed    = claim_host_bytes(desired_l2, cache->host_reserve);
    if (claimed < min_l2 || (cache->config.strict && cache->config.l2_bytes != 0 && claimed < cache->config.l2_bytes)) {
        release_host_bytes(claimed);
        cache->set_error("safe host memory is below the required L2 capacity");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        cache->teardown_storage();
        return false;
    }
    cache->claimed_bytes = claimed;

    size_t remaining = claimed;
    while (remaining >= largest_physical) {
        const memory_info current_memory = read_memory_info();
        size_t            current        = std::min(chunk_size, remaining);
        if (current_memory.available != 0) {
            if (current_memory.available <= cache->host_reserve) {
                break;
            }
            current = std::min(current, current_memory.available - cache->host_reserve);
        }
        current = align_down(current, cache->l2_alignment);
        if (current < largest_physical) {
            break;
        }
        uint8_t * data = nullptr;
        if (!allocate_aligned_mapping(current, cache->l2_alignment, data) || !register_host(data, current)) {
            if (data != nullptr) {
                munmap(data, current);
            }
            break;
        }
        cache->chunks.push_back({ data, current });
        cache->stats.l2_capacity += current;
        remaining -= current;
    }
    if (remaining != 0) {
        release_host_bytes(remaining);
        cache->claimed_bytes -= remaining;
    }
    if (cache->stats.l2_capacity < min_l2 ||
        (cache->config.strict && cache->config.l2_bytes != 0 && cache->stats.l2_capacity < cache->config.l2_bytes)) {
        cache->set_error("failed to allocate and pin the minimum useful L2 capacity");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        cache->teardown_storage();
        return false;
    }

    size_t cacheable_bytes = 0;
    for (const owned_source & source : cache->sources) {
        cacheable_bytes = source.desc.tensor_bytes > std::numeric_limits<size_t>::max() - cacheable_bytes ?
                              std::numeric_limits<size_t>::max() :
                              cacheable_bytes + source.desc.tensor_bytes;
    }
    size_t remaining_inclusive = cache->stats.l2_capacity;
    size_t minimum_l1          = 0;
    if (cache->config.l1_bytes_per_device == 0) {
        size_t minimum_by_expert = 0;
        if (!mul_checked(cache->config.largest_expert_extent, 8, minimum_by_expert)) {
            cache->set_error("the minimum automatic L1 capacity overflows size_t");
            cache->placement_invalidated = true;
            cache->override_enabled      = false;
            cache->teardown_storage();
            return false;
        }
        minimum_l1 = std::max(GIB, minimum_by_expert);
    }
    for (device_cache & device : cache->devices) {
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(device.backend), &free, &total);
        GGML_UNUSED(total);
        size_t desired =
            cache->config.l1_bytes_per_device == 0 ? free : std::min(free, cache->config.l1_bytes_per_device);
        desired = std::min(desired, std::min(cacheable_bytes, remaining_inclusive));
        desired = align_down(desired, std::max(cache->config.largest_expert_extent, device.alignment));
        if (desired < minimum_l1) {
            continue;
        }
        const size_t required_l1 = align_down(cache->config.l1_bytes_per_device,
                                              std::max(cache->config.largest_expert_extent, device.alignment));
        if (cache->config.strict && cache->config.l1_bytes_per_device != 0 && desired < required_l1) {
            cache->set_error("post-reservation memory is below the requested per-device L1 capacity");
            cache->placement_invalidated = true;
            cache->override_enabled      = false;
            cache->teardown_storage();
            return false;
        }
        while (desired >= std::max(cache->config.largest_expert_extent, minimum_l1) && desired != 0) {
            device.buffer = ggml_backend_alloc_buffer(device.backend, desired);
            if (device.buffer != nullptr) {
                break;
            }
            if (cache->config.strict) {
                break;
            }
            desired = align_down(desired / 2, std::max(cache->config.largest_expert_extent, device.alignment));
        }
        if (device.buffer == nullptr) {
            if (cache->config.strict) {
                cache->set_error("failed to allocate the requested L1 buffer");
                cache->placement_invalidated = true;
                cache->override_enabled      = false;
                cache->teardown_storage();
                return false;
            }
            continue;
        }
        device.capacity = desired;
        remaining_inclusive -= desired;
        cache->stats.l1_capacity += desired;
    }
    if (cache->stats.l1_capacity == 0) {
        cache->set_error("no usable L1 capacity remains after graph reservation");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        cache->teardown_storage();
        return false;
    }

    const int ring_result = io_uring_queue_init(RING_DEPTH, &cache->ring, 0);
    if (ring_result < 0) {
        cache->set_error("failed to initialize io_uring");
        cache->placement_invalidated = true;
        cache->override_enabled      = false;
        cache->teardown_storage();
        return false;
    }
    cache->ring_ready = true;
    cache->active     = true;
    cache->admissions = true;
    GGML_LOG_INFO("%s: requested L1=%zu MiB/device L2=%zu MiB, effective L1=%zu MiB L2=%zu MiB, reserve=%zu MiB\n",
                  __func__, cache->config.l1_bytes_per_device / MIB, cache->config.l2_bytes / MIB,
                  cache->stats.l1_capacity / MIB, cache->stats.l2_capacity / MIB, cache->host_reserve / MIB);
    return true;
}

void ggml_backend_moe_cache_set_eligible(ggml_backend_moe_cache * cache, bool eligible) {
    if (cache != nullptr) {
        cache->eligible = eligible;
    }
}

bool ggml_backend_moe_cache_should_offload(const ggml_backend_moe_cache * cache,
                                           const ggml_tensor *            node,
                                           const ggml_tensor *            weight,
                                           ggml_backend_t                 backend) {
    if (cache == nullptr || !cache->override_enabled || !cache->eligible || node == nullptr || weight == nullptr ||
        node->op != GGML_OP_MUL_MAT_ID || cache->find_source(weight) == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev == nullptr ? nullptr : ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
}

enum ggml_backend_moe_cache_result ggml_backend_moe_cache_copy_experts(ggml_backend_moe_cache * cache,
                                                                       ggml_backend_t           input_backend,
                                                                       ggml_backend_t           split_backend,
                                                                       const ggml_tensor *      input,
                                                                       ggml_tensor *            input_cpy,
                                                                       const int32_t *          ids,
                                                                       size_t                   n_ids) {
    GGML_UNUSED(input_backend);
    if (cache == nullptr || !cache->eligible || !cache->override_enabled || cache->find_source(input) == nullptr) {
        return GGML_BACKEND_MOE_CACHE_NOT_HANDLED;
    }
    if (!cache->active || !cache->ring_ready) {
        cache->fail_runtime("the MoE cache was selected before successful activation");
        return GGML_BACKEND_MOE_CACHE_FAILED;
    }
    if (ids == nullptr || n_ids == 0) {
        cache->fail_runtime("an eligible MUL_MAT_ID graph has no selected experts");
        return GGML_BACKEND_MOE_CACHE_FAILED;
    }

    const int            device_index = cache->backend_index(split_backend);
    const owned_source * source       = cache->find_source(input);
    if (device_index < 0 || source == nullptr || input->ne[2] != source->desc.n_expert ||
        input->nb[2] != source->desc.expert_size) {
        cache->fail_runtime("an eligible MUL_MAT_ID tensor does not match its source descriptor");
        return GGML_BACKEND_MOE_CACHE_FAILED;
    }
    cache->barrier();
    cache->ensure_host_margin();
    cache->age_frequencies();
    cache->generation++;

    std::vector<int32_t> experts(ids, ids + n_ids);
    std::sort(experts.begin(), experts.end());
    experts.erase(std::unique(experts.begin(), experts.end()), experts.end());

    std::vector<access_record> accesses;
    accesses.reserve(experts.size());
    for (int32_t expert : experts) {
        if (expert < 0 || expert >= source->desc.n_expert) {
            cache->fail_runtime("a selected expert ID is out of range");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }
        size_t tensor_offset = 0;
        if (!mul_checked(static_cast<size_t>(expert), source->desc.expert_size, tensor_offset)) {
            cache->fail_runtime("an expert offset overflows size_t");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }
        const size_t padding = expert < source->desc.n_expert - 1 ? std::min<size_t>(source->desc.expert_size, 512) : 0;
        size_t       copy_size = 0;
        if (!add_checked(source->desc.expert_size, padding, copy_size) || tensor_offset > source->desc.tensor_length ||
            copy_size > source->desc.tensor_length - tensor_offset) {
            cache->fail_runtime("an expert copy including MMQ padding exceeds the source tensor");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }

        access_record access;
        access.key           = { device_index, source->desc.source_id, expert };
        access.source        = source;
        access.expert        = expert;
        access.tensor_offset = tensor_offset;
        access.copy_size     = copy_size;
        access.l1            = cache->find_l1(access.key);
        access.l2            = cache->find_l2(access.key);
        cache->frequencies[access.key]++;
        accesses.push_back(access);
    }

    device_cache &            device = cache->devices[device_index];
    std::vector<read_request> requests;
    requests.reserve(std::min<size_t>(RING_DEPTH, accesses.size()));

    auto enqueue_ready = [&]() {
        for (access_record & access : accesses) {
            if (access.copied) {
                continue;
            }
            if (access.l1 != nullptr && access.l1->state == l1_state::ready) {
                access.l1->pins++;
                access.l1->last_use  = ++cache->clock;
                l2_entry * inclusive = cache->find_l2(access.key);
                if (inclusive == nullptr) {
                    cache->fail_runtime("an L1 entry is missing its inclusive L2 entry");
                    return false;
                }
                inclusive->pins++;
                inclusive->last_use = cache->clock;
                uint8_t * source_data =
                    static_cast<uint8_t *>(ggml_backend_buffer_get_base(device.buffer)) + access.l1->offset;
                uint8_t * destination = static_cast<uint8_t *>(input_cpy->data) + access.tensor_offset;
                copy_device_range_async(device.backend, device.buffer, source_data, input_cpy->buffer, destination,
                                        access.copy_size);
                cache->stats.l1_hits++;
                access.copied = true;
                continue;
            }
            if (access.l2 != nullptr && access.l2->state == l2_state::ready) {
                access.l2->pins++;
                access.l2->last_use = ++cache->clock;
                const uint8_t * source_data =
                    cache->chunks[access.l2->chunk].data + access.l2->offset + access.l2->data_offset;
                ggml_backend_tensor_set_async(split_backend, input_cpy, source_data, access.tensor_offset,
                                              access.copy_size);
                if (!access.l3_fill) {
                    cache->stats.l2_hits++;
                }
                access.copied = true;
            }
        }
        return true;
    };

    if (!enqueue_ready()) {
        return GGML_BACKEND_MOE_CACHE_FAILED;
    }

    for (access_record & access : accesses) {
        if (access.copied) {
            continue;
        }
        if (!cache->admissions) {
            cache->fail_runtime("host reserve stopped L3 admissions");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }

        if (access.tensor_offset > std::numeric_limits<uint64_t>::max() - source->desc.tensor_offset) {
            cache->fail_runtime("a direct expert file offset overflows uint64_t");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }
        const uint64_t logical_offset   = source->desc.tensor_offset + access.tensor_offset;
        const size_t   offset_alignment = source->desc.offset_alignment;
        const uint64_t physical_offset  = logical_offset - logical_offset % offset_alignment;
        const size_t   head             = static_cast<size_t>(logical_offset - physical_offset);
        size_t         physical_size    = 0;
        if (!add_checked(head, access.copy_size, physical_size) ||
            !align_up_checked(physical_size, offset_alignment, physical_size)) {
            cache->fail_runtime("a direct expert read extent overflows size_t");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }
        if (physical_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            cache->fail_runtime("a direct expert read is too large for io_uring completion accounting");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }

        l2_entry * entry = cache->allocate_l2(access.key, physical_size, source->desc.memory_alignment);
        if (entry == nullptr) {
            if (!requests.empty()) {
                if (!cache->submit_reads(requests)) {
                    cache->fail_runtime(cache->error);
                    return GGML_BACKEND_MOE_CACHE_FAILED;
                }
                requests.clear();
            }
            if (!enqueue_ready()) {
                return GGML_BACKEND_MOE_CACHE_FAILED;
            }
            cache->record_pending(device);
            cache->sync_device(device);
            entry = cache->allocate_l2(access.key, physical_size, source->desc.memory_alignment);
        }
        if (entry == nullptr) {
            cache->fail_runtime("no L2 wave can hold a required aligned expert extent");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }
        entry->data_offset  = head;
        entry->logical_size = access.copy_size;
        access.l2           = entry;
        access.l3_fill      = true;
        requests.push_back({ entry, source, physical_offset, physical_size, cache->generation, false, 0 });

        if (requests.size() == RING_DEPTH) {
            if (!cache->submit_reads(requests)) {
                cache->fail_runtime(cache->error);
                return GGML_BACKEND_MOE_CACHE_FAILED;
            }
            requests.clear();
            if (!enqueue_ready()) {
                return GGML_BACKEND_MOE_CACHE_FAILED;
            }
        }
    }

    if (!cache->submit_reads(requests)) {
        cache->fail_runtime(cache->error);
        return GGML_BACKEND_MOE_CACHE_FAILED;
    }
    if (!enqueue_ready()) {
        return GGML_BACKEND_MOE_CACHE_FAILED;
    }

    for (access_record & access : accesses) {
        if (!access.copied) {
            cache->fail_runtime("an expert access was not copied into the CUDA input tensor");
            return GGML_BACKEND_MOE_CACHE_FAILED;
        }
        cache->admit_l1(access, input_cpy);
    }
    cache->record_pending(device);
    return GGML_BACKEND_MOE_CACHE_HANDLED;
}

ggml_moe_cache_stats ggml_backend_moe_cache_get_stats(const ggml_backend_moe_cache * cache) {
    return cache == nullptr ? ggml_moe_cache_stats{} : cache->stats;
}

void ggml_backend_moe_cache_reset_stats(ggml_backend_moe_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    const size_t l1_capacity = cache->stats.l1_capacity;
    const size_t l2_capacity = cache->stats.l2_capacity;
    cache->stats             = {};
    cache->stats.l1_capacity = l1_capacity;
    cache->stats.l2_capacity = l2_capacity;
}

bool ggml_backend_moe_cache_consume_placement_invalidated(ggml_backend_moe_cache * cache) {
    if (cache == nullptr) {
        return false;
    }
    const bool result            = cache->placement_invalidated;
    cache->placement_invalidated = false;
    return result;
}

const char * ggml_backend_moe_cache_get_error(const ggml_backend_moe_cache * cache) {
    return cache == nullptr || cache->error.empty() ? nullptr : cache->error.c_str();
}

#else

struct ggml_backend_moe_cache {
    ggml_moe_cache_config config                = {};
    ggml_moe_cache_stats  stats                 = {};
    bool                  placement_invalidated = false;
    std::string           error;
};

ggml_backend_moe_cache * ggml_backend_moe_cache_new(ggml_backend_t *, int) {
    return new ggml_backend_moe_cache;
}

void ggml_backend_moe_cache_free(ggml_backend_moe_cache * cache) {
    delete cache;
}

void ggml_backend_moe_cache_reset(ggml_backend_moe_cache *) {}

bool ggml_backend_moe_cache_configure(ggml_backend_moe_cache *      cache,
                                      const ggml_moe_cache_config * config,
                                      const ggml_moe_cache_source *,
                                      size_t) {
    if (cache == nullptr || config == nullptr) {
        return false;
    }
    cache->config = *config;
    if (config->mode != GGML_MOE_CACHE_MODE_OFF) {
        cache->error = "GGML_MOE_CACHE is not available in this build";
        return false;
    }
    return true;
}

bool ggml_backend_moe_cache_activate(ggml_backend_moe_cache * cache) {
    return cache != nullptr && cache->config.mode == GGML_MOE_CACHE_MODE_OFF;
}

void ggml_backend_moe_cache_set_eligible(ggml_backend_moe_cache *, bool) {}

bool ggml_backend_moe_cache_should_offload(const ggml_backend_moe_cache *,
                                           const ggml_tensor *,
                                           const ggml_tensor *,
                                           ggml_backend_t) {
    return false;
}

enum ggml_backend_moe_cache_result ggml_backend_moe_cache_copy_experts(ggml_backend_moe_cache *,
                                                                       ggml_backend_t,
                                                                       ggml_backend_t,
                                                                       const ggml_tensor *,
                                                                       ggml_tensor *,
                                                                       const int32_t *,
                                                                       size_t) {
    return GGML_BACKEND_MOE_CACHE_NOT_HANDLED;
}

ggml_moe_cache_stats ggml_backend_moe_cache_get_stats(const ggml_backend_moe_cache * cache) {
    return cache == nullptr ? ggml_moe_cache_stats{} : cache->stats;
}

void ggml_backend_moe_cache_reset_stats(ggml_backend_moe_cache * cache) {
    if (cache != nullptr) {
        cache->stats = {};
    }
}

bool ggml_backend_moe_cache_consume_placement_invalidated(ggml_backend_moe_cache * cache) {
    if (cache == nullptr) {
        return false;
    }
    const bool result            = cache->placement_invalidated;
    cache->placement_invalidated = false;
    return result;
}

const char * ggml_backend_moe_cache_get_error(const ggml_backend_moe_cache * cache) {
    return cache == nullptr || cache->error.empty() ? nullptr : cache->error.c_str();
}

#endif
