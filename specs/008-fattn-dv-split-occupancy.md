# 008: Split DV across CTAs to buy attention occupancy (sm89)

Status: active (ready-for-agent)
Label: ready-for-agent
Depends on: 005 (closed as no change; it produced the measurements this spec is built on), 001
Scope: sm89, DK = DV = 256, prefill and verify-batch shapes.

## Note on the register gate, 2026-09-24

The gate is `cuobjdump -res-usage` on the split instantiation, and it cannot be
closed by arithmetic on the declarations, which is worth recording because it is
tempting to try.

The declarations say `VKQ_C[DV / (2 * T_C_VKQ::J)]` = 16 tiles of 4 half2 = 64
registers per thread, `KQ_C[nbatch_fa / (np * T_C_KQ::J)]` = 2 tiles of 8 floats =
16, and `Q_B[DKQ / (2 * T_B_KQ::J)]` = 16 tiles of 4 half2 = 64 for the Q fragments.
Summed with the rest that should be well over the 255 measured, and the one time this
reasoning was tested against a build it was wrong by 3.5x: flipping `Q_in_reg` on this
row moved the measured count from 255 to 237, so the Q fragments cost about 18
registers, not the 64 the declaration implies. The compiler is reusing Q's registers
for other live state, which is invisible to a paper budget.

So the gate is a real compile of a real split, and the honest prior is that it is a
coin flip: 64 for VKQ, 16 for KQ, tens for operands and addressing, against a 128
ceiling for 2 CTAs of 256 threads. Do not close or open this spec on arithmetic.

## Problem Statement

`flash_attn_ext_f16` is 21.0% of a 65k prefill and 35.5% of a 131k one on this
stack, and it is running with 8 warps per SM where the machine has warp slots for
48. 005 established the reason with counters and a register map, and it is neither
bandwidth nor the config table.

Measured on the shipping build, `cuobjdump -res-usage` against
`libggml-cuda.so`:

| instantiation | shape | registers per thread |
|---|---|---|
| `flash_attn_ext_f16<256,256,16,4,...>` | 9B prefill, GQA 4 | **255** |
| `flash_attn_ext_f16<256,256,8,8,...>` | 27B prefill, GQA 6 | **255** |
| lowest of every DK=256 instantiation | ncols=32 shapes | 231 |

All with a 16-byte stack frame, `LOCAL:0`. 255 is the hardware ceiling.

The occupancy arithmetic, 65536 registers per Ada SM:

| target | registers per thread allowed |
|---|---|
| 2 CTAs x 128 threads (today) | 256, so 255 fits with 256 of slack |
| 3 CTAs x 128 threads | **170** |
| 2 CTAs x 256 threads | **128** |

Nothing in the DK=256 space reaches 170, and the wide-block variant was measured,
not assumed: `nthreads` 128 to 256 with `occupancy` 2 costs **-23.9%** on pp65536
(6188.9 against 8136.4 t/s, three balanced runs, no overlap), because the compiler
must then fit in 128 registers and spills. `nbatch_combine` 128 to 64 was measured
at -0.24%, behind the baseline in all three rounds.

`Q_in_reg` was also tested directly, by flipping it on the production row and
reading the register count: 255 to 237 for `<256,256,16,4>`, and it adds
`ncols * (DKQ/2 + 4) * 4` = 33.8 KB of shared memory per CTA, which puts 2 CTAs
over the 100 KB budget. It buys 18 registers and costs the one thing that could
have paid for them. Not a lever.

From 005's counters, what the low occupancy is actually costing:

| metric | value |
|---|---|
| Tensor (FP) pipe | 57.7% |
| L2 throughput | 63.0% |
| DRAM throughput | 14.6% |
| L2 hit rate | 95.7% |
| active warps per scheduler | 1.64 of 12 |
| dominant stall | 56% math-pipe throttle |
| ncu `Est. Local Speedup` | 37% |

Both dominant resources are around 60% and the kernel is register-saturated at two
CTAs per SM. That is a tile-shape problem.

## Why the accumulator is the register cost

The VKQ accumulator holds `ncols x DV` floats per CTA. At the production shape that
is 64 x 256 = 16384 floats, which across 128 threads is **128 registers per
thread**, half the entire budget, and it is invariant to how the work is
otherwise sliced. Q in registers adds about 18 measured. The rest is operand
fragments and address arithmetic.

So the only term big enough to cut is the accumulator, and the way to cut it
without cutting the tile shape is to stop one CTA from being responsible for all
256 output columns.

## Design: DV split across two CTAs

Each CTA handles `DV/2` of the value and output columns for the same query and key
tile, and the grid gains a dimension for the half.

Register arithmetic at the production shape, ncols=64:

| term | today | with DV split, `nthreads` 256 |
|---|---|---|
| VKQ accumulator, `ncols x (DV/2)` floats | 128/thread | **32**/thread |
| Q fragments, `Q_in_reg` | ~18/thread | ~9/thread |
| operands, KQ accumulators, addressing | ~109 | ~90 |
| estimated total | 255 | **~130** |

Target is 128 for 2 CTAs of 256 threads, which is the same 16 warps per SM that 4
CTAs of 128 threads would give, at half the CTAs. If the estimate lands above 128,
`nthreads` 256 with `occupancy` 1 is pointless and the split has to be along
`ncols` instead, which is the fallback, not the plan.

Shared memory at the split, with `nbatch_fa` 32 and `nbatch_combine` 64:

| term | today | with DV split, 8 warps |
|---|---|---|
| KV tile | 32.0 KB | 24.0 KB, V is half width |
| combine, `nwarps x 16 x (nbatch_combine+4) x 4` | 33.8 KB | 17.4 KB |
| mask | 1.3 KB | 1.3 KB |
| per CTA | 33.8 KB | ~25 KB |
| 2 CTAs | 67.6 KB | ~50 KB, fits |

What it costs:

- K is fetched by both halves, so the K read doubles. At a 95.7% L2 hit rate and
  14.6% DRAM that is the cheapest thing this kernel can double, and V does not
  double at all.
- The stream-K and fixup paths have to carry the extra grid dimension, and so does
  the sparse-gather path, or those two must be excluded from the split at first.
- The output store is naturally split, since each CTA owns a disjoint DV half.
- The softmax denominators and row maxima stay per query row and are computed
  identically in both halves, so the numerics of the KQ pass do not change; the
  PV accumulators are disjoint, so nothing has to be combined across the two CTAs.
  That is the property that makes this a grid-and-stride change rather than a
  reduction change.

## Test Seam and Testing Decisions

Unchanged from 005, and it already covers this shape: `tests/test-backend-ops.cpp`,
`test_flash_attn_ext`, which parameterises `hsk`, `hsv`, head count, kv length,
batch and KV type. A DV split is invisible at this seam, which is exactly what
makes it the right one: the output must match the CPU reference with the current
tolerances, including the cases where `hsv != hsk` and where V is a view of K.

- The grid change must be exercised where the fixup path triggers, so the
  stream-K cases at `nb` (batch) 8 and 75 with kv 512 and 1024 matter more than
  new synthetic shapes.
- `mask` and `sinks` cases stay as they are; the split must not care about them.
- No new test file, no new op, nothing in `src/`.

## Acceptance Criteria

Baseline numbers to beat, all measured on this machine: pp65536 9B with q8_0 KV and
`-fa on` at 8136 t/s, where `flash_attn_ext_f16` is 2972 ms of a 14165 ms captured
pass; and 001's 27B pp131071, where the kernel is 35.5% of 402 s.

- Register count for the split instantiation at or below 128 per thread, read from
  `cuobjdump -res-usage` before any benchmarking. If it is not, the design stops
  there and the result is recorded, because the whole case rests on it.
- `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0` fully green, 5 consecutive runs.
- Active warps per scheduler rises from 1.64 toward 3 or more, with the tensor pipe
  above 65%, from the same ncu section set used in 005.
- pp65536 on the 9B: at least +5% in a balanced multi-run comparison, and the FA
  share of a captured pass down from 21.0%.
- 27B pp131071: at least +8%, since the kernel is 35.5% there. This is the case
  the spec exists for.
- Decode and short prefill within 1%: an extra grid dimension must not cost
  anything when the KV is short, which means the split has to be declined by the
  host for small `n_kv`, the same way `use_sparse` is declined today.
- Perplexity on the 9B and the 27B unchanged to the printed precision, on the same
  corpus and chunk count used for 007.

## Rollback and Risk

- The estimate that matters is the non-accumulator register cost, about 99 of the
  255 today, and it is inferred from `Q_in_reg` (255 to 237) rather than measured
  per fragment. If the real number after the split is 150 instead of 130, the
  target shape is unreachable and this spec closes as a measurement. Cheap to find
  out: the register count is known from a compile, before any benchmarking.
- Halving V's tile width per CTA shortens the mma run per KV tile and doubles the
  number of CTAs walking the same K, so the L2 could become the limiter instead of
  the tensor pipe. The counters will say so directly.
- Stream-K, the fixup kernel and the sparse gather each need the new grid
  dimension. The first cut should refuse the split for those three configurations
  and take the win on plain dense prefill and verify batches, which is where
  001 says the time is.
- This changes a hot kernel that every attention model uses. The `n_kv` and shape
  conditions must be narrow: DK = DV = 256, no ALiBi, no sinks, no softcap, and
  nothing that already has a dedicated tuned row in the table.

## Out of Scope

- Reading quantized KV in the MMA kernel, and anything from 005's original design.
  005 measured that ceiling at 3.1%.
- Any change to the vec or tile kernels, to the MMQ GEMM (004), or to the SSM
  kernel (006).
- fp8 or fp4 attention math, and any new type.
- The `n_kv_max` bound on the leftover conversion, which 005 noted as a small
  independent change.
- Hopper and Blackwell, where wgmma and TMA change the tradeoff entirely.

## Further Notes

Measurement provenance:

- register map: `cuobjdump -res-usage build-004/bin/libggml-cuda.so.0.25.0`, parsed
  per mangled instantiation, DK=256 rows, shipping build `b56c071f5464b683e6c09fa94e6a268b`
- counters, stall mix, warps per scheduler: `/tmp/ncu005.txt`, spec 005
- conversion pricing, f16 against q8_0 KV: spec 005, `/tmp/005/kv_*.sqlite`
- config sweep that failed: spec 005, `/tmp/005/sweep.csv`
- `Q_in_reg` probe: register counts read from a throwaway build, flag reverted, no
  perf run taken because the shared-memory arithmetic alone rules it out
- shares at 65k and 131k: this spec's tables and `specs/001-baseline.md`
