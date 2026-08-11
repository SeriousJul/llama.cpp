#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

enum llama_file_access {
    LLAMA_FILE_ACCESS_BUFFERED,
    LLAMA_FILE_ACCESS_DIRECT_PREFERRED,
    LLAMA_FILE_ACCESS_DIRECT_STRICT,
};

enum llama_mmap_policy {
    LLAMA_MMAP_POLICY_STOCK,
    LLAMA_MMAP_POLICY_DEMAND,
};

struct llama_file {
    llama_file(const char * fname, const char * mode, enum llama_file_access access = LLAMA_FILE_ACCESS_BUFFERED);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
    size_t direct_memory_alignment() const;
    size_t direct_offset_alignment() const;
  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, enum llama_mmap_policy policy = LLAMA_MMAP_POLICY_STOCK, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;
    bool   contains(const void * ptr, size_t len) const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
