#pragma once

#include "ggml-backend.h"

struct ggml_backend_moe_cache;

enum ggml_backend_moe_cache_result {
    GGML_BACKEND_MOE_CACHE_NOT_HANDLED = 0,
    GGML_BACKEND_MOE_CACHE_HANDLED     = 1,
    GGML_BACKEND_MOE_CACHE_FAILED      = 2,
};

ggml_backend_moe_cache * ggml_backend_moe_cache_new(ggml_backend_t * backends, int n_backends);
void                     ggml_backend_moe_cache_free(ggml_backend_moe_cache * cache);
void                     ggml_backend_moe_cache_reset(ggml_backend_moe_cache * cache);

bool                               ggml_backend_moe_cache_configure(ggml_backend_moe_cache *      cache,
                                                                    const ggml_moe_cache_config * config,
                                                                    const ggml_moe_cache_source * sources,
                                                                    size_t                        n_sources);
bool                               ggml_backend_moe_cache_activate(ggml_backend_moe_cache * cache);
void                               ggml_backend_moe_cache_set_eligible(ggml_backend_moe_cache * cache, bool eligible);
bool                               ggml_backend_moe_cache_should_offload(const ggml_backend_moe_cache * cache,
                                                                         const ggml_tensor *            node,
                                                                         const ggml_tensor *            weight,
                                                                         ggml_backend_t                 backend);
enum ggml_backend_moe_cache_result ggml_backend_moe_cache_copy_experts(ggml_backend_moe_cache * cache,
                                                                       ggml_backend_t           input_backend,
                                                                       ggml_backend_t           split_backend,
                                                                       const ggml_tensor *      input,
                                                                       ggml_tensor *            input_cpy,
                                                                       const int32_t *          ids,
                                                                       size_t                   n_ids);

ggml_moe_cache_stats ggml_backend_moe_cache_get_stats(const ggml_backend_moe_cache * cache);
void                 ggml_backend_moe_cache_reset_stats(ggml_backend_moe_cache * cache);
bool                 ggml_backend_moe_cache_consume_placement_invalidated(ggml_backend_moe_cache * cache);
const char *         ggml_backend_moe_cache_get_error(const ggml_backend_moe_cache * cache);
