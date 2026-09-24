# 005: Quantized KV read directly by the MMA flash-attention kernel (sm89)

Status: done (2026-09-24, verdict: no change). The removable part measured at
3.1%, and the two config directions that the counters suggested both fail.
Label: closed
Depends on: 001 (P2/D2 kernel breakdown), 002 (verify-step attention numbers)
Scope: sm89 only, q8_0 KV only (add q4_0 in a second step if it is free).

## Measurement first (2026-09-23): the removable part is 3%, not 12-15%

The clean way to price the conversion pass is to run the same prefill twice, once
with an f16 KV cache (no conversion, kernel reads f16) and once with q8_0 (kernel
reads the converted f16 copy). The attention kernel does the same work in both, so
the difference *is* the conversion, including its memory-system side effects.

9B IQ4_XS, pp65536, `-fa on`, ubatch 512, one rep, nsys totals:

| | f16 KV | q8_0 KV |
|---|---|---|
| pass, wall | 7.52 s | 7.70 s (**+2.4%**) |
| captured kernel total | 13836 ms | 14165 ms |
| `flash_attn_ext_f16` | 2981.7 ms, 21.5% | 2972.7 ms, 21.0% |
| `dequantize_block_q8_0_f16` | absent | 444.9 ms, 3.1% |
| `mul_mat_q` | 7329.9 ms | 7202.1 ms |
| `gated_delta_net_cuda` | 1356.9 ms | 1331.3 ms |

Two facts come out of this and both matter for the design:

1. The removable cost is the dequant pass, and it is **3.1%** of a 65k prefill, not
   the 12-15% this spec claimed. It is also already efficient: 4096 instances at
   108.6 us, each moving 67 MB in and 134 MB out, which is 1.84 TB/s effective.
2. The attention kernel is 21% here and 35.5% in 001's 131k profile, and it is
   **not** DRAM-bandwidth bound. Its unique KV traffic per pass is about 274 GB in
   2.98 s, which is 9% of the 1008 GB/s roof, and its effective rate against the
   bytes it must actually fetch is bounded by L2, since the eight query blocks of a
   ubatch re-read the same 33.5 MB per-head slice. Halving those bytes would leave
   roughly 9% of the roof, from a kernel already sitting at 21% of wall time.

So the read-width half of this spec's premise is weak, and the loader change carries
a specific risk: the quantized path cannot use `cp.async` into the operand tile, so
it would trade a halved byte count for losing the async staging that a latency-bound
kernel may be relying on. That trade has to be measured, not assumed.

## What is still worth doing here, in order

1. Price the tile/occupancy question on `flash_attn_ext_f16` with counters before
   any loader work, the way 004 and 006 did. Command in Further Notes. If it comes
   back LSU- or issue-bound like MMQ, the 21-35% of wall in this kernel is reachable
   by a config-table change for sm89, which is a far better prize than 3.1%.
2. Then decide the loader on its own merits: 3.1% on a 65k prefill, plus about 536
   MB of scratch per attention node at this context length, which on a 24 GB card is
   what forces `-ub 512` instead of a larger ubatch. The VRAM argument may be worth
   more than the 3.1% in the production 160k config, where the scratch is 1.3 GB.
3. The cheap interim option: `launch_fattn` converts `ggml_nelements(K)` with no
   bound on rows. Passing the already-known `n_kv_max` and converting only
   `[0, n_kv_max)` is a few lines. It does not help a fresh prefill, where
   n_kv_max is the whole prefix, but it caps the cost for the verify step and for
   any configuration where the cache is longer than the active window.

## Counters on flash_attn_ext_f16 (2026-09-24), `/tmp/ncu005.txt`

Profiled instance: `flash_attn_ext_f16<256, 256, 16, 4, ...>`, grid (256,1,1),
block (32,4,1), 2.73 ms, the tail ubatch of a 65k prefill.

| metric | value | reading |
|---|---|---|
| **Tensor (FP) pipe** | **57.7%** | the top pipe; this kernel is already on tensor cores |
| Compute (SM) throughput | 57.7% | same |
| L2 throughput | 63.0% | co-limiter |
| **DRAM throughput** | **14.6%** | 144 GB/s, nothing |
| L2 hit rate | 95.7% | the KV re-reads are L2 hits, as predicted |
| **active warps per scheduler** | **1.64 of 12** | 8 warps per SM, 2 CTAs |
| issued warps per scheduler | 0.23, one instruction every 4.3 cycles | |
| dominant stall | 4.0 of 7.11 cycles, 56%, math-pipe throttle | the tensor pipe is oversubscribed at too few warps |
| ncu `Est. Local Speedup` | 37% | from occupancy and stall alone |

The config row this shape lands on explains the low warp count: the Ampere MMA table
row `(256, 256, 64, 128, 2, 32, 128, 128, 128, 2, true)` gives the DK=256 prefill
tile, which both the 9B (ncols1 16 x ncols2 4) and the 27B (8 x 8) use, 128 threads
per block and an occupancy target of 2. That is 4 warps per CTA and 8 warps per SM,
which is where 1.64 active warps per scheduler comes from. `nstages_target` on this
row is already 2, so async staging is on; an earlier draft of this section claimed
otherwise and that was a misread of the table.

Working the shared-memory formula in `ggml_cuda_flash_attn_ext_mma_f16_case` out for
this row, with `Q_in_reg = true` so the Q tile is not allocated:

| term | size |
|---|---|
| KV, 2 stages: `nbatch_fa * (stride_K + stride_V) * 4` | 32.0 KB |
| mask: `ncols1 * (nbatch_fa/2 + 4) * 4` | 1.3 KB |
| combine: `nwarps * 16 * (nbatch_combine + 4) * 4` | **33.8 KB** |
| per CTA, `max(combine, KV + mask)` | 33.8 KB |
| 2 CTAs per SM | 67.6 KB of the 100 KB |
| 3 CTAs per SM | 101.4 KB, does not fit |

So the combine buffer, not the KV tile, is what pins this kernel at 2 CTAs per SM,
and it is sized off `nthreads`: 4 warps x 16 columns x 132 half2. Two ways out, and
they are the experiment:

1. shrink `nbatch_combine` 128 to 64, which makes the KV tile the limiter at 33.3 KB.
   Predicted weak: 3 CTAs still needs 3 x 34.3 KB with the driver reserve.
2. go to `nthreads` 256 with `occupancy` 2 and `nbatch_combine` 64. The combine term
   grows with warps per CTA but not proportionally to the extra work each thread
   gives up, so per CTA it is about 34.8 KB, 2 CTAs fit, and the machine gets 16
   warps per SM instead of 8. This is the one that should move the 37%.

### Verdict on the original 005 design: do not build it

The kernel is tensor-pipe and L2 bound at 14.6% of DRAM. Reading q8_0 instead of
f16 would halve L2 traffic, which helps a co-limiter, but the quantized path cannot
`cp.async` the operand tile because the values must be converted on the way in, and
this kernel is running with only 1.64 warps per scheduler relying on whatever
staging it does have. Trading async staging for bytes, in a kernel that is not
bandwidth-bound, is likely net negative. The part that is certainly removable, the
dequant pass, is 3.1%.

### What to do instead

Retune the MMA FA config rows for sm89, which is a data table plus possibly one
extra instantiation, against a kernel that is 21% of a 65k prefill and 35.5% of a
131k one, and whose own counters ask for 37%. Candidates, in the order the
arithmetic favours:

1. `nstages_target` 1 to 2 on `(256, 256, 64)`, which costs 16 KB of shared memory
   per CTA and is almost certainly why the row is set to 1: 2 CTAs x 2 stages
   approaches Ada's 100 KB per SM. Measure whether staging wins over occupancy.
2. `occupancy` 2 to 1 with `nstages` 2 and more warps per CTA (`nthreads` 128 to
   256), which trades CTAs per SM for warps per SM at the same shared-memory budget.
3. `nbatch_fa` 32 to 64, which halves the softmax rescale and accumulator-scaling
   work per KV tile, the part that competes with the mma for issue slots.
4. `ncols1` and `ncols2` themselves, since they set both the KV re-read count and
   the tile shape, which is the re-scope of the deferred 001 item C.

Acceptance for the whole set: the `flash_attn_ext_f16` share of a 65k prefill drops
while `test-backend-ops -o FLASH_ATTN_EXT` stays green, measured paired ABBA, and
the tensor-pipe percentage rises at the same occupancy.

### The config sweep, run and lost

Balanced 3-run comparison at pp65536 on the 9B with q8_0 KV and `-fa on`, the case
where this kernel is 21% of the wall:

| variant | pp65536 | vs baseline |
|---|---|---|
| baseline, `(256, 256, 64, 128, 2, 32, 128, 128, 128, 2)` | 8136.4 t/s | - |
| `nbatch_combine` 128 -> 64 | 8116.5 t/s | -0.24%, and behind the baseline in all three rounds |
| `nthreads` 128 -> 256 with `occupancy` 2 | 6188.9 t/s | **-23.9%** |

The reason the wide-block variant collapses is in the register file, not shared
memory. `cuobjdump -res-usage` on the instantiated FA kernels shows 182 to 212
registers per thread with a non-zero stack. Two CTAs of 128 threads at 212 registers
is 54.3 K of the 64 K per SM, which fits. Two CTAs of 256 threads at the same
footprint is 108 K, which does not, so `__launch_bounds__(256, 2)` forces the
compiler down to 128 registers per thread and the kernel spills. That is the 24%.

So the honest statement of the constraint is: **the DK=256 attention tile cannot go
above 8 warps per SM without shrinking the per-thread register footprint**, and no
row in the table can buy that. `Q_in_reg = true` is what holds the Q fragments
live; giving it up, or reducing the per-thread tile along DK, is a kernel design
study with a real chance of losing, not a tuning change.

### Verdict

1. Do not build the q8_0 loader. Ceiling 3.1% on a 65k prefill, in a kernel that is
   tensor-pipe and L2 bound rather than DRAM bound, and it costs the `cp.async`
   staging that 1.64 warps per scheduler depend on.
2. The table retune that the counters suggested is closed by measurement in both
   directions available to it.
3. `flash_attn_ext_f16` stays the largest unexploited item on this stack, 21% of a
   65k prefill and 35.5% of a 131k one, with a named blocker: 212 registers per
   thread. If it is ever opened again it should be as a tile-design study, with the
   register budget as the first constraint, and not as a dispatch or table change.
4. The one small change that survives from this spec's original list is unrelated to
   quantized KV: `launch_fattn` converts `ggml_nelements(K)` with no row bound,
   where `n_kv_max` is already passed to the node. It does not help a fresh prefill
   and was not implemented here.

The working-tree change from this investigation was reverted; the tree is back at
the two committed kernels and `test-backend-ops -o FLASH_ATTN_EXT` is unaffected.

## Problem Statement

When the KV cache is quantized, the flash-attention path that Ada uses for
prompt processing and for spec-decode verify steps does not read the cache. It
first builds a full f16 copy of the layer's K and V, on every attention call,
and then reads that copy. The copy is pure overhead: it is thrown away and
rebuilt for the next ubatch, growing with the context.

Where this is set in code:

- `ggml/src/ggml-cuda/fattn.cu`, in `ggml_cuda_flash_attn_ext_get_alloc_size`:
  for `BEST_FATTN_KERNEL_TILE` and `BEST_FATTN_KERNEL_MMA_F16` the backend sets
  `need_f16_K = need_f16_V = true` with no condition on the KV type. Ada always
  takes MMA_F16 for prefill and for verify batch > 2.
- `ggml/src/ggml-cuda/fattn-common.cuh`, in `launch_fattn`: `to_fp16` /
  `to_fp16_nc` over `ggml_nelements(K)` and `ggml_nelements(V)`, once per node
  execution. Not per new token: per call.
- `ggml/src/ggml-cuda/fattn-common.cuh`, in
  `ggml_cuda_flash_attn_ext_get_f16_extra_data`: the scratch for that copy is
  carved out of the attention node's own buffer allocation, so it is VRAM the
  model cannot use for anything else.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, in `flash_attn_ext_f16_load_tile`: the
  tile loader signature takes `const half2 * KV`. There is no quantized source
  in the MMA kernel at all.
- the vec kernel already solves this problem: `ggml/src/ggml-cuda/fattn-vec.cuh`
  is templated on `type_K` / `type_V` and consumes q8_0 from DRAM through the
  `vec_dot` path, and `fattn.cu` compiles those instances from
  `GGML_CUDA_FA_QUANTS`.

Measured cost, from 001 and 002:

| what | number | source |
|---|---|---|
| dequant q8_0 -> f16, pp131071 | 11.2 s of 402 s = 2.8% | 001, D2 table |
| dequant instances | 49152 = 2 per (pass, ubatch, attn layer) | 001 |
| fattn mma-f16, pp131071 | 142.7 s = 35.5%, 5.8 ms per instance | 001, D2 table |
| effective KV read rate in that kernel | ~47 GB/s of unique bytes | 001 finding |
| per attn layer per ubatch at the 131k tail | 2 x 447 us dequant + 11.5 ms mma | 001 |
| verify-step attention at 131k, batch 2-8 | dequant 14.2 ms + mma 10.1 ms = 24.3 ms per step | 002 |

Three consequences follow. The 273 MB of q8_0 becomes 546 MB of f16 that the
kernel then reads, so the attention kernel moves 2x the bytes it needs. The
write-then-read stream of 546 MB per layer per ubatch passes through the 72 MB
L2 and displaces what is useful there, which is the best available explanation
for the 47 GB/s figure. And the copy costs about 536 MB of VRAM at 131k context,
on a 24 GB card that already holds 15.5 GB of weights plus 4.4 GB of KV.

## Solution

Give the MMA kernel quantized tile loaders, the same way the vec kernel has
them. A KV tile is read from the cache as q8_0, converted to f16 on its way into
shared memory, and the tensor-core math and the shared-memory layout do not
change. The per-call `to_fp16` pass and its scratch buffer go away.

From the operator's point of view: faster long prompts and faster spec-decode
verify steps at long context, less VRAM used by the attention scratch, and the
same q8_0 cache type as today. From the maintainer's point of view: this closes
the gap the vec kernel has always covered, on one arch and one KV type, and it
removes an allocation from the backend's per-node sizing.

## User Stories

1. As a llama-server operator, I want prefill at 100k+ context to skip the f16
   KV copy, so that my time to first token drops by roughly a tenth.
2. As a llama-server operator, I want the spec-decode verify step to stop paying
   14 ms of conversion per step, so that accepted-token bursts feel shorter.
3. As a llama-server operator, I want the ~536 MB of attention scratch back, so
   that I can raise context or ubatch instead of buying a second GPU.
4. As a llama-server operator, I want q8_0 KV to be as fast as f16 KV in the
   attention kernel rather than slower, so that my cache-type choice stops being
   a speed/size tradeoff.
5. As a user with one 24 GB card, I want the whole 27B plus 131k of KV to fit
   with room to spare, so that the server does not spill layers to the CPU.
6. As a llama.cpp contributor, I want the change limited to a template parameter
   on the existing tile loader, so that no attention math is rewritten.
7. As a llama.cpp contributor, I want the new instances driven by the same
   `GGML_CUDA_FA_QUANTS` list the vec kernel uses, so that a build without q8_0
   MMA support behaves exactly as it does today.
8. As a llama.cpp contributor, I want the f16 fallback path to stay, so that a
   KV type with no MMA instance still runs (slowly, and with the existing
   warning) instead of aborting.
9. As a reviewer, I want DK=256 to be the case that is proven first, since 256
   divides the q8_0 block of 32 exactly, so that the tile loader does not need
   ragged-row handling.
10. As a reviewer, I want the `V_is_K_view` case handled, so that MLA-style
    caches where V is a view of K keep working with one conversion, as they do
    today.
11. As a person running the 9B dev model, I want a per-layer kernel A/B from one
    nsys capture, so that the change is judged on kernel sums and not on wall
    noise from the CPU speculator.
12. As a maintainer of the vec path, I want the Ada dispatch rule in
    `ggml_cuda_get_best_fattn_kernel` left as it is, so that 002's falsified
    "relax the `<= 2` gate" experiment is not quietly re-attempted here.

## Implementation Decisions

- Add `type_K` / `type_V` template parameters to the MMA kernel and to
  `flash_attn_ext_f16_load_tile`. For f16 the code path must be bit-identical to
  today, including the `cp.async` staging, so the quantized variants may not
  change the f16 instantiation's shared-memory sizing.
- The quantized variant stages raw q8_0 bytes with `cp.async` and converts to
  f16 inside shared memory, because a `cp.async` destination must be shared
  memory and the mma operand layout cannot be produced straight from global.
  Where the second stage buffer does not fit at DK=256 on a 100 KB SM, that
  instantiation runs with `nstages=1` and the fact is recorded in the PR body;
  a synchronous register-then-store conversion is the fallback that always fits.
- A q8_0 row of DK=256 is 8 blocks x 34 B = 272 B, and `FATTN_KQ_STRIDE` tiling,
  the swizzle pattern, and the `half2` operand tiles are unchanged. The
  conversion writes the same bytes the `to_fp16` pass used to write, only
  sooner and without leaving the SM.
- Ship q8_0 only. q4_0 has a per-block scale and a min term in some layouts and
  is a follow-up. Do not enable the whole quant matrix in one change.
- `ggml_cuda_flash_attn_ext_get_alloc_size` stops requesting the f16 scratch for
  the node types that now have a native loader. Keep the code that computes it,
  since the fallback path still needs it.
- `ggml_cuda_get_best_fattn_kernel` does not change. Whatever kernel it picks
  today keeps being picked; only the work inside it gets cheaper. This keeps 002
  closed.
- Sparse-gather and multi-stage loading stay mutually exclusive, as today.
- No change to the vec kernel, and no change to the KV cache layout.

## Test Seam and Testing Decisions

One seam, and it needs one parameter change, not a new harness:
`tests/test-backend-ops.cpp`, `test_flash_attn_ext`, which already takes
`type_KV` for K and V and already runs q8_0 through the vec path.

- The existing enumeration gate restricts quantized KV to head sizes 64 and 72
  (`if (type_KV != GGML_TYPE_F16 && hsk != 64 && hsk != 72) continue;`), so
  DK=256 with q8_0 is not covered by anything today. Relaxing that gate for
  q8_0 at hsk=256 is the whole test-side change, and it is also the acceptance
  gate for this spec: the case must pass on CPU, on the vec kernel, and on the
  new MMA instance, with identical tolerances.
- Good tests here compare the op output against the CPU reference within the
  tolerance the file already defines for that type pair. They do not assert
  which kernel ran, which stages ran, or that a scratch buffer exists.
- Cover, in this order: DK=DV=256 with a GQA ratio of 6 (the production shape,
  q-heads as `nr2`), batch > 2 so the MMA path is the one under test, kv lengths
  both below and above one `FATTN_KQ_STRIDE` tile, a permuted V, `V` as a view of
  `K`, and a mask with a hole (prompt-cache style) so the padded-region handling
  is exercised.
- Prior art in the same file: the mixed-type rows near the end of the FA
  enumeration (`test_flash_attn_ext(64, 64, 4, ... GGML_TYPE_Q8_0,
  GGML_TYPE_Q4_0)` and friends), and the `v_is_view_of_k` cases for MLA.
- Add no new file to `tests/`.

## Acceptance Criteria

Measured per 001/002 method: service stopped, one standalone nsys capture per
library, kernel sums inside the eval window sliced by the `print_timing` line.
Not wall t/s, which 002 shows is dominated by the CPU ngram speculator.

- pp131071: the `dequantize` rows leave the profile entirely (baseline 11.2 s of
  402 s), and `fattn mma-f16` total drops by >= 20% from its 142.7 s.
- pp131071 wall: >= 8% faster than the 1905 t/s baseline in 001.
- pp32768: >= 3% faster than 2705 t/s, so the change pays at mid context too.
- Verify step at 131k, per attn layer: the 1.5 ms (0.41 + 0.46 dequant + 0.62
  mma) becomes <= 0.9 ms, i.e. the 24.3 ms per step is at or under 15 ms.
- Peak VRAM during a 131k prefill drops by >= 350 MB.
- `test-backend-ops -o FLASH_ATTN_EXT -b cuda` green, including the new 256/q8_0
  cases, 5 consecutive runs.
- Output equality against the baseline library on a fixed 8k-token prompt: same
  sampled tokens for the first 64 generated tokens with temp 0.
- The f16-KV configuration is within 1% of its own baseline (the f16
  instantiation must not regress while the quantized one is added).

## Rollback and Risk

- Risk 1: the conversion inside the SM costs more ALU than the copy saved, which
  would show up as a faster dequant-kernel-free profile but a slower total. The
  A/B in the acceptance list catches it; the change is then closed as no-op.
- Risk 2: shared memory does not fit two stages at DK=256, so the quantized
  instantiation loses `cp.async` pipelining and gets slower. Decide per
  instantiation: 1 stage plus a 272 B row copy beats 2 stages plus a 544 B row
  copy only if it is measured.
- Risk 3: alignment. The q8_0 row stride in the unified cache is not a multiple
  of 16 B for every head count, and `cp.async` needs 4/8/16 B units. Where the
  stride fails, use the non-async conversion path for that instantiation rather
  than changing the cache layout.
- Risk 4: compile time and binary size, since this multiplies the MMA
  instantiations. Keep the new instances behind the `GGML_CUDA_FA_QUANTS` list
  so a default build pays for q8_0 only.
- Rollback: the `need_f16_K/V = true` assignment is one place in
  `ggml_cuda_flash_attn_ext_get_alloc_size`. Restoring it returns the old path,
  with the copy kernel and the scratch both still in the tree.

## Out of Scope

- Relaxing the `Q->ne[1] <= 2` Ada dispatch gate to route verify batches onto the
  vec kernel, and any `cols_per_block = 8` vec work. Both are falsified by 002.
- A GQA-packed vec kernel that reads one KV chunk per layer instead of per
  q-group. 002 measures that as under 1% of felt wall on this stack.
- An fp8 or fp4 KV cache type, and any new ggml type.
- Blackwell, Hopper, AMD, and the tile kernel (it keeps `need_f16_K/V = true`).
- Chunking the SSM prefill kernel: see 006.

## Further Notes

Evidence pointers (line numbers as of `bddf8263c`, upstream/master):

- unconditional f16 requirement: `ggml/src/ggml-cuda/fattn.cu`,
  `ggml_cuda_flash_attn_ext_get_alloc_size`
- per-call conversion: `ggml/src/ggml-cuda/fattn-common.cuh`, `launch_fattn`
- scratch sizing inside the node allocation: same file,
  `ggml_cuda_flash_attn_ext_get_f16_extra_data`
- f16-only tile loader: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`,
  `flash_attn_ext_f16_load_tile`
- working precedent for quantized KV: `ggml/src/ggml-cuda/fattn-vec.cuh`
  (`type_K`, `type_V`, `vec_dot_KQ`, `Q_q8_1`)
- Ada dispatch: `ggml/src/ggml-cuda/fattn.cu`,
  `ggml_cuda_get_best_fattn_kernel`
- build-time KV type matrix: `ggml/CMakeLists.txt`, `GGML_CUDA_FA_QUANTS`
  (this machine: `f16-f16;q8_0-q8_0;q4_0-q4_0`)
- 002 verify-step table and the ngram-map noise finding: `specs/002-...md`
