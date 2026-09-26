# 011: Page hints for the CUDA UVM path

Status: done (2026-09-26, verdict: no. Hints alone cannot cross the overflow
cliff; see Results. Prototype preserved as
`specs/artifacts/011-sweep/011-prefetch-prototype.patch`; the tree keeps only a
1-line bug fix it exposed.)
Depends on: 001 (shares), no code dependency on 012 or 013
Scope: `ggml/src/ggml-cuda/`, NVIDIA path, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`.

## Origin

Analysis of `RaymondHuang210129/llama.cpp-adaptive-kv-streaming` (branch
`feature/kv-stream-phase-arena`, 2026-09). That fork beats stock Unified Memory at long
context by *not* using Unified Memory: it manages residency itself because the stock UVM
path thrashes. Reading our tree, the reason is one missing API call set: the NVIDIA UVM
path allocates and stops.

Verified in-tree:

- `ggml/src/ggml-cuda/ggml-cuda.cu:144-158`: under `GGML_CUDA_ENABLE_UNIFIED_MEMORY`,
  `ggml_cuda_malloc` calls `cudaMallocManaged`. That is the only CUDA API touched.
  There is no `cudaMemAdvise` and no `cudaMemPrefetchAsync` anywhere in the CUDA
  backend (only the AMD-only `hipMemAdviseSetCoarseGrain` hint, line 150).
- Consequence: every page fault is demand-driven through the driver. A layer that reads
  weights or KV whose pages were evicted stalls the kernel until migration finishes,
  one 4 KB to 2 MB page at a time, with no overlap with compute.

The idea: the access pattern is fully known to the host at launch time. Before a kernel
that reads tensor X on device d, the pages of X are almost always needed, and after
inference settles, most of them should live in VRAM. Both facts are exactly what
`cudaMemAdvise` and `cudaMemPrefetchAsync` are for. This spec adds the hints; it does
not add a new memory subsystem.

## Problem Statement (measured 2026-09-26)

Harness: `build-004` (`4d500ed6b`, b11146), model `Qwen3.8-27B-UD-IQ4_XS`
(14.24 GB file), `-ngl 99 -fa on -b 2048 -ub 512 -n 128 --ignore-eos -st -fit off`.
Prompts are synthetic filler, token counts verified with `llama-tokenize --stdin
--show-count`: P1 = 117,801 tokens, P2 = 209,001 tokens (5.58 bytes/token).
`ON` = `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, `OFF` = unset. Full rows in
`specs/artifacts/011-sweep/window*.log`.

| kv | ctx | prompt | OFF pp | OFF tg | ON pp | ON tg |
|---|---|---|---|---|---|---|
| q8_0 | 160000 | P1 117.8K | 2111.0 | 38.5 | 2090.9 | 38.4 |
| q8_0 | 262144 | P2 209K | 1611.8 | 31.1 | 1607.2 | 31.1 |
| f16 | 163840 | P1 | OOM (compute pp buffers, 295.7 MiB refused) | - | 2151.4 | 21.4 |
| f16 | 262144 | P1 | OOM (KV 16384 MiB refused) | - | 2122.0 | 37.9 |
| f16 | 262144 | P2 | OOM | - | **1112.2** | **0.6** |

What the table says:

1. **When managed allocations fit VRAM, UVM is free.** q8_0 rows: ON within 0.3%
   of OFF at pp and tg, at both depths. The earlier window-1 ABBA matrix
   (combined pp160K+tg256, OFF 1687.7/1696.1, ON 1672.9/1672.1 t/s) said the same.
   The driver migrates the whole working set to device pages and stops faulting.
   The env var only routes buffer allocations (`ggml-cuda.cu:884 -> :141-145`);
   pool scratch uses the VMM pool (`:679-686`) regardless.
2. **The cliff is the target.** P2 f16 spills ~3.4 GiB of a ~27.5 GiB footprint
   (model 12726 + KV ~13932 + compute 861 MiB vs 24084 MiB device): decode
   collapses 31.1 -> 0.6 t/s, a 52x fall, and pp drops 1611 -> 1112. That 0.6 t/s
   is what stock UVM hands the user who wants f16 KV at 209K on 24 GB. The fork's
   entire reason to exist is this regime (its 16 GB card lives in it permanently).
3. Demand-driven migration alone recovers almost nothing once any part of the
   steady-state working set is out of VRAM: 3.4 GiB of 24 GB spilled costs 98%
   of decode. Decode touches every KV byte every token, so the LRU has no locality
   to exploit: it is a pure PCIe-latency-bound fault storm, one migration at a
   time, zero compute overlap.

Hypothesis for the design, sharpened by the measurement: `cudaMemPrefetchAsync`
issued per KV/weight buffer on the compute stream one layer ahead converts that
fault storm into pipelined DMA. The gap between 0.6 t/s (driver-managed faults)
and ~31 t/s (the same bytes when resident) is the headroom hints can chase; the
fork demonstrates the shape of the answer at 10-15 t/s with hand-managed rings.

### Phase 1 result (2026-09-26, window 5): advise-at-allocation FAILED

`cudaMemAdvise(SetAccessedBy)` applied to every managed buffer at allocation
(build-011, same-binary A/B via `GGML_CUDA_UVM_HINTS`, cliff row f16 P2 c262K):

| | CTRL x2 | HINT |
|---|---|---|
| pp t/s | 1110.1, 1109.0 | **142.8** |
| tg t/s | 0.6, 0.6 | **1.7** |

Decode improves 2.8x as predicted, but prefill falls 7.8x: persistent device
mappings make the first-touch GPU writes into host-resident KV pages during pp
expensive in a way the lazy default is not. Net full-request time regresses ~6x.

### Phase 2 result (windows 6-14): lookahead prefetch FAILED differently

`cudaMemPrefetchAsync(src, used_prefix_bytes, stream)` issued W=64 nodes ahead in
the node loop, NVIDIA-only, gated three ways (env opt-out, managed-bytes > 0.9 x
VRAM regime detector, decode-shape detector `ggml_nrows(FA q) <= 1024`). CUDA
capture rejects prefetch nodes, so the mode also drops graphs for decode graphs
only. Measured on the cliff row (f16 P2 c262K):

| variant | pp | tg |
|---|---|---|
| CTRL | 1104-1127 | 0.6-0.8 |
| pre-pass, unclamped (v2a) | timeout >25 min | n/a |
| lookahead (v2g final) | 972 | **0.2** |
| fitting regime (q8 c160K) | 2092 vs 2079 | 38.2 vs 38.2 (parity, gate off) |
| ops suite under UVM | green (rc=0) | |

Why decode got *worse*: the cliff working set (27.5 GiB) exceeds VRAM (24.1 GiB),
so every step's prefetch of "the next layer's KV prefix + weights" evicts bytes
the current step still needs, and `cudaMemPrefetchAsync` over already-resident
ranges is not free (per-page table work on the host path). Prefetch with no
eviction budget is a ping-pong generator: the fork's whole point is that it *is*
an eviction-budget subsystem (resident pool + ring + layer pacing), and no hint
can emulate that.

Bonus row (window 011c, f16 at ctx 160000 = the *near-cliff*, spill only ~0.4
GiB): pp 133 t/s, tg 1.2 t/s. Near-cliff is worse than the 3.4 GiB cliff, not
better: max residency means maximum LRU churn, every page is hot and every miss
ejects something hot. Prefetch W=8 did not save it either.
Measurement rule born the hard way in 011c: a q8 fitting-regime row measured
116/1.2 t/s immediately after two f16 cliff rows, and 2082/38.2 when re-run
alone. Managed host pages from prior runs (27 GiB each) had not returned to the
free pool fast enough; the q8 run was really executing at the cliff. Cliff rows
must not share a window with fitting-regime controls; re-run the control last,
or in its own process window after the driver settles.

### Verdict

The 50x gap between 0.6 and 31 t/s at the cliff is real, but it is not reachable
from the hint surface: bulk DMA without residency control moves the fault storm
into a migration storm. Anything that wins here owns the page budget, i.e. 013.
The salvageable outputs:

1. A real bug fix kept in-tree: `cuda_device_info::total_vram` was declared but
   never populated by `ggml_cuda_init`, so it read 0 (this silently inverted an
   early threshold and cost an hour to find). One line, in the device loop where
   `device_vram` is computed. Candidate for a small standalone upstream PR.
2. Measured characterization of the UVM regimes (tables above), now in-file
   instead of assumed.
3. `011-prefetch-prototype.patch` records the failed variants and the working
   parts found along the way (decode-shape test via `ggml_nrows(FA q)`, the
   `n_kv_max` prefix clamp for FA srcs, capture-illegality of prefetch nodes),
   a starting point if 013 ever builds the real budget owner.

## Design

Two hook points, both small:

1. **Advise at allocation.** In `ggml_cuda_malloc` (managed branch), after
   `cudaMallocManaged`, record the allocation. When a tensor is assigned to a CUDA
   device buffer at load time, call
   `cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device)` and
   `cudaMemAdviseSetAccessedBy` for the using devices. For weights on a partially
   offloaded model this pins what fits; the remainder stays migratable.
2. **Prefetch ahead of compute.** In `ggml_cuda_compute_forward`
   (`ggml-cuda.cu:2062`), before dispatching an op whose src tensor is managed, issue
   `cudaMemPrefetchAsync(ptr, size, device, stream)` for that tensor on the same
   stream. The natural placement is once per tensor per graph node set, not per kernel,
   so CUDA graph replay is unaffected: prefetched pages stay resident between replays.

Deliberately out: eviction policies beyond defaults, per-page hot-set tracking, a
managed-allocator replacement. If hints alone do not close the gap, this spec stops and
013 carries the idea.

## Acceptance Criteria

- [x] Baselines in this file before code lands (2026-09-26).
- Long-ctx decode (tg at the largest ctx that survives today): hinted UVM strictly
  better than unhinted UVM by more than run noise, paired ABBA per the README rule.
- Non-UVM runs unchanged: when the env is unset, the binary behavior and allocation
  path must be byte-identical in effect. Verify by md5-free code path check plus one
  control A/B.
- `test-backend-ops -b CUDA0` green with UVM on (this is currently not even exercised
  in CI on this machine; add a run, not a test file).
- No new CLI flags. Env-gated behavior stays under the same env that already enables
  managed memory.

## Rollback and Risk

- Consumer Windows drivers page UVM through WDDM with much worse migration granularity;
  the fork records a write-combined pinned-host fix (its commit `05ba2068`). Test Linux
  first; treat Windows as a separate report.
- `cudaMemPrefetchAsync` on a huge tensor per layer can cost more than it saves when
  everything already fits in VRAM. The prefetch must be conditional on the allocation
  actually being migratable (advice differs for resident vs overflow).
- Interaction with CUDA graphs: prefetched state is not captured; replay relies on pages
  still being resident. That is safe (a miss still migrates), but the win may shrink
  after idle eviction. Measure replay-heavy decode specifically.

## Out of Scope

- Weight streaming by block (012 territory), KV residency rings (013), ROCm path,
  anything in `src/` or `common/`.
