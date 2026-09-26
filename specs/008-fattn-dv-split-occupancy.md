# 008: Split DV across CTAs to buy attention occupancy (sm89)

Status: done (2026-09-26, verdict: no change). All three gates close it, on measurement and without
a benchmark run: criterion 1 fails on registers, the accumulator turned out not to be the register
cost, and the one spill-free occupancy step costs 3x the KV bytes against an L2 already at 63%.
See "Gates, measured 2026-09-26"
Label: research
Depends on: 005 (closed as no change; it produced the measurements this spec is built on), 001
Scope: sm89, DK = DV = 256, prefill and verify-batch shapes.

## Gates, measured 2026-09-26

Two throwaway probes, no benchmarking and no kernel change. Both restore the tree when they
finish, and the register one is re-runnable per shape:
`bash specs/artifacts/008-reg-gate.sh [DV] [nthreads] [occupancy] [ncols ...]`.

### Registers

`008-reg-gate.sh [DV] [nthreads] [occupancy] [ncols ...]` inserts probe rows for a DKQ=256 kernel
owning a slice of DV (the register shape of a split CTA) into `fattn-mma-f16.cuh`, generates a
matching instantiation into /tmp (kept out of the tree so it cannot drift from the rows it tests),
compiles it with build-008's own nvcc flags, and reads `cuobjdump -res-usage`. Nothing links and
nothing runs, so each cell costs one file compile. The archived output is `008-reg-gate.txt`; read
occupancy 1 as "what the kernel wants" (the budget cannot bind there) and occupancy >1 as "what
launch bounds force", with STACK as where the difference goes.

| registers per thread | budget | half DV, ncols=64 | half DV, ncols=32 | full DV, ncols=64 | full DV, ncols=32 |
|---|---|---|---|---|---|
| 128 threads, occupancy 2 | 256 | 219, STACK 16 | 193, STACK 16 | 255, STACK 16 (ships) | 231, STACK 16 |
| 128 threads, occupancy 3 | **170** | 168, **STACK 64** | **168, STACK 16** | 255, over | 231, over |
| 128 threads, occupancy 4 | 128 | 128, **STACK 272** | 128, **STACK 128** | - | - |
| 256 threads, occupancy 1 | 256 | 192, STACK 16 | 196, STACK 16 | - | - |
| 256 threads, occupancy 2 | **128** | 128, **STACK 112** | 128, **STACK 176** | 255, over | 231, over |

Criterion 1 fails for the design as frozen: a half-DV CTA at the production tile wants **192**
registers at 256 threads, and the budget for 2 CTAs is 128. Forcing it pays in local memory,
STACK 16 -> 112, which is the same trade that measured -23.9% on the `nthreads` 128 -> 256 run.
"Over budget" on a full-DV row means nvcc left the count alone rather than squeezing it, so the
shape cannot hold those warps; the half-DV rows do get squeezed, and STACK is the bill.

### The accumulator is not the register cost

Splitting further was the obvious next question, so it was measured. One config row and one
instantiation per compile, because two rows on the same `(DKQ, DV, ncols)` key make the first one
win silently - that is how an earlier version of this probe reported registers for a shape it never
compiled. `008-reg-gate.txt` is the whole matrix.

| registers per thread, 128 threads | budget 256 (occ 2) | budget 170 (occ 3) | budget 128 (occ 4) |
|---|---|---|---|
| full DV, ncols=64 (ships) | 255, STACK 16 | over | - |
| half DV, ncols=64 | 219, STACK 16 | 168, STACK 64 | 128, STACK 272 |
| quarter DV, ncols=64 | 187, STACK 16 | 168, STACK 32 | - |
| full DV, ncols=32 | 231, STACK 16 | over | - |
| half DV, ncols=32 | 193, STACK 16 | **168, STACK 16** | 128, STACK 128 |
| quarter DV, ncols=32 | 178, STACK 16 | **168, STACK 16** | - |

Two things follow.

1. **The accumulator is roughly a third of the production row, not half of it.** Quartering DV
   (128 -> 32 columns per CTA, nominally 128 -> 32 registers of accumulator) moves the measured
   count 255 -> 187 at ncols=64 and 231 -> 178 at ncols=32. At ncols=32 a further quartering buys
   nothing at all: occupancy 3 measures 168 for half DV and 168 for quarter DV. There is a
   per-thread floor near 168 that does not contain the accumulator, so "why the accumulator is the
   register cost" is measured false, and a plan that buys occupancy by shrinking the accumulator
   hits that floor before it hits its target.
2. **The floor sits between exactly two occupancy budgets.** 168 clears 170 (3 CTAs of 4 warps)
   with STACK 16 and cannot clear 128 (4 CTAs of 4 warps, or 2 CTAs of 8), where the compiler pays
   the difference in stack. So the register side allows one step up from today's 8 warps, and which
   CTA counts shared memory will actually grant is the next section.

So the reachable ceiling for this whole idea is 8 warps -> 12 warps, and getting there costs the
tile narrowing as well as the split.

### Shared memory

`008-smem-occupancy.cu` asks the other half: does shared memory even allow the CTA count a shape
wants? It replays the launcher's own `nbytes_shared_total` expression and calls the same
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` that `launch_fattn` calls, with a dummy kernel
whose only relevant property is its thread count, so the register file cannot interfere. It exists
because an arithmetic pass over that same expression got these numbers wrong in the permissive
direction: the Q term is `ncols * (DKQ/2 + 4) * 4` and does not contain DV, but the combine term
is `nwarps * cols_per_warp * (nbatch_combine + 4) * 4`, which grows with `nwarps` and shrinks with
`nbatch_combine`, and at 8 warps it is the binding term, not Q.

    device: 102400 B shared per SM, 101376 B max per CTA, 65536 registers per SM, 128 SMs
    ships        ncols=64 DV=256 n128 nstages=2   33792 B/CTA -> 2 CTA/SM =  8 warps (budget 256)
    DV split     ncols=64 DV=128 n256 nstages=2   34816 B/CTA -> 2 CTA/SM = 16 warps (budget 128)
    DV split     ncols=64 DV=128 n256 nstages=1   34816 B/CTA -> 2 CTA/SM = 16 warps (budget 128)
    DV split     ncols=64 DV=128 n128 nstages=2   33792 B/CTA -> 2 CTA/SM =  8 warps (budget 256)
    DV split nmw ncols=32 DV=128 n128 nstages=2   25216 B/CTA -> 3 CTA/SM = 12 warps (budget 170)
    DV split nmw ncols=32 DV=128 n128 nstages=1   17408 B/CTA -> 5 CTA/SM = 20 warps (budget 102)
    DV quarter   ncols=64 DV=64  n256 nstages=2   33792 B/CTA -> 2 CTA/SM = 16 warps (budget 128)
    narrow       ncols=32 DV=256 n128 nstages=1   33792 B/CTA -> 2 CTA/SM =  8 warps (budget 256)
    narrow       ncols=16 DV=256 n128 nstages=1   33792 B/CTA -> 2 CTA/SM =  8 warps (budget 256)

### What the two gates leave

1. **Criterion 1 fails as frozen, and shared memory is not the sole reason.** At the production tile with
   half DV and 256 threads there is room for 2 CTAs (34816 B of the 102400 B per SM), which is the
   16 warps the design was after. The register file refuses: 192 wanted against a 128 budget, paid
   for in local memory (STACK 16 -> 112, and the shipping kernel spills zero requests today, so that
   is all new traffic). My first read of this section claimed the opposite, that Q pinned the CTA and
   8 warps was the ceiling; that was arithmetic on one term of `nbytes_shared_total` while the max()
   had other terms, and the device API says otherwise. Note the shipping row is limited by registers
   *and* shared memory at 2 CTAs at the same time (both Block Limits read 2 in the ncu dump below);
   the 256-thread design is the one that escapes the shared-memory limit while still failing registers.
2. **The spec's headline gain, 16 warps, is not reachable cleanly by any shape.** The candidates
   that could hold it all pay in stack: half DV at 256 threads and occupancy 2 (STACK 112 at
   ncols=64, 176 at ncols=32), quarter DV at 256 threads and occupancy 2 (STACK 48), and 4 CTAs at
   128 threads (STACK 128 to 272). Nothing between 8 warps and 12 warps is spill-free.
3. **12 warps is reachable clean**: half DV, ncols=32, nthreads=128, occupancy=3, at REG 168 of a
   170 budget, STACK 16, 25216 B per CTA. That is the fallback the design section named ("the split
   has to be along `ncols` instead, which is the fallback, not the plan"), and it is +50% warps
   rather than the +100% this spec was written for.
4. **The accumulator is not the register cost, so the mechanism is weaker than the design
   assumed.** Quartering DV buys 32 registers at ncols=64 and exactly zero at ncols=32, against a
   nominal accumulator of 128 and 64. There is a per-thread floor near 168 that does not move with
   the accumulator; see the table in the previous section.
5. **No config-row-only route exists.** Full DV measures 255 at ncols=64 and 231 at ncols=32, at
   128 threads and at 256, and the number does not move between occupancy 2 (budget 256) and
   occupancy 3 (budget 170) - nvcc leaves the count alone rather than squeezing it, so "over budget"
   here means the shape cannot hold those warps. Halving the tile is worth 24 registers and the
   170 budget is 61 away. The occupancy cannot be bought from the table.
6. **What the surviving shapes cost, in L2 traffic.** Splitting DV n ways makes n CTAs read the same
   K while each reads only its own `V/n` slice, so per KV chunk the bytes go from `K + V` to
   `n*K + V`; halving the query tile multiplies that by 2 again.

   | shape | K reads | V reads | total KV bytes vs today | L2 at 63.0% x that |
   |---|---|---|---|---|
   | 12 warps: DV/2 + ncols/2 | 4x | 2x | 3.0x | 189% |
   | 16 warps: DV/4, ncols=64 | 4x | 1x | 2.5x | 158% |

   Both overshoot the pipe 005 measured at 63.0% while DRAM sat at 14.6% and the L2 hit rate at
   95.7%: the traffic is already carried by L2 and there is no DRAM headroom to trade against it.
   This column is arithmetic on read counts, not a measurement, and one ncu section set on a single
   `flash_attn_ext_f16` call (L2 throughput plus the Block Limit columns) would settle it in one
   run. But it points the same way as the floor in item 4 and as 009, where reaching occupancy on
   MMQ measured -20.9%: on this card these kernels are not short of warps.

### Baseline occupancy and bandwidth, from the archived 005 ncu dump

The block limits this spec needs were already captured in `specs/artifacts/tmp-ncu005.txt` on the
shipping row `flash_attn_ext_f16<256, 256, 16, 4, 0, 0, 0>`, grid (256,1,1) x block (32,4,1), CC 8.9.
No new profiling run was needed, and two of 008's assumptions change under it.

    Registers Per Thread                     255
    Shared Memory Configuration Size       102.40 Kbyte
    Driver Shared Memory Per Block           1.02 Kbyte/block
    Dynamic Shared Memory Per Block         34.05 Kbyte/block   <- confirms the 33792 B computed above
    Waves Per SM                                 1
    Block Limit Registers                        2
    Block Limit Shared Mem                       2
    Block Limit Warps                           12
    Theoretical Active Warps per SM              8   (16.67% occupancy)
    Achieved Active Warps per SM               6.54   (13.63%)
    Active Warps Per Scheduler                1.64
    Local Memory Spilling Requests               0
    Mem Busy                                  63.00 %
    Max Bandwidth                             62.82 %
    L2 Cache Throughput                       63.00 %
    L2 Hit Rate                               95.72 %
    DRAM Throughput                           14.62 %
    Duration                                   2.73 ms

1. **Both limits bind at 2 today.** Block Limit Registers 2 and Block Limit Shared Mem 2, so the
   shipping shape is pinned by the register file and by shared memory at the same time. ncu's own
   wording: "theoretical occupancy (16.7%) is limited by the number of required registers, and the
   required amount of shared memory".
2. **`launch_fattn` launches exactly one wave.** Under stream-K the grid is
   `min(max_blocks_per_sm * nsm, work)`, which is why Waves Per SM is 1 and the grid is 256 = 2 x 128
   SMs. So raising CTAs per SM does convert to real concurrency: at 3 CTAs the launcher would run 384
   CTAs over the same total work, with a shorter KV range each. The occupancy premise is structurally
   sound; the gates are what it has to pass to get there.
3. **Today the kernel spills nothing.** Local Memory Spilling Requests 0, with a 16 B frame per
   thread. So the frame column in the register tables is not traffic today, and it measures the size
   of the hazard rather than the cost: the frozen design's STACK 112 on a 128-thread CTA is 112 x 128
   x 256 = ~3.5 MB of live state parked in local memory per wave, in a kernel that currently parks
   none. Whether it is actually touched in the inner loop is exactly what that same ncu metric would
   say for a built variant.
4. **The bandwidth price of both surviving shapes does not fit.** At DKQ = DV = 256 the two streams
   are equal width, so splitting DV n ways takes a query tile from `K + V` to `n*K + V`, which is
   `(n+1)/2`. Halving the query tile is a separate 2x, because each tile re-reads the KV it attends
   over. Measured against the 63.00% L2 duty cycle, and giving each shape the generous best case
   where duration falls exactly in proportion to the warps it adds:

   | shape | per tile | tiles | traffic vs today | warps | best-case duration | duty needed |
   |---|---|---|---|---|---|---|
   | ships today: ncols=64, full DV | K + V | 1x | 1.0 | 8 | 1.00 | 63.0% |
   | 12 warps: ncols=32, DV/2 | 2K + V | 2x | 3.0 | 12 | 0.67 | 282% |
   | 16 warps: ncols=64, DV/4 | 4K + V | 1x | 2.5 | 16 | 0.50 | 315% |

   Both need more than twice the L2 the kernel is already spending, and the credit for extra warps is
   generous: 009 measured what reaching occupancy actually bought on MMQ at -20.9%, and ncu's own
   read of this kernel is "Compute and Memory are well-balanced: to reduce runtime, both computation
   and memory traffic must be reduced". Adding warps reduces neither.

### Verdict

Closed as no change, without writing the split. Three independent reasons, any one of which is
enough:

- **Criterion 1.** The frozen shape needs 192 registers where 2 CTAs of 256 threads allow 128, and
  launch bounds do not produce a fit, they produce a 96-byte-per-thread frame on a kernel that
  spills nothing today.
- **The mechanism.** The accumulator is not the register cost. There is a ~168 per-thread floor
  independent of DV, so the split removes the smallest part of the problem and the floor decides
  which occupancy is reachable. The design's central table ("128/thread -> 32/thread") is measured
  false.
- **The price.** The only spill-free step above today's 8 warps is 12, and it needs 3.0x the KV
  bytes against a memory system ncu describes as "well-balanced", where the fix for both is fewer
  bytes and less compute, not more warps.

What a future attention spec should take from this file: the shipping row is limited by registers
and shared memory *both* at 2 CTAs (ncu Block Limits 2 and 2), `launch_fattn` runs exactly one wave,
and 34.05 KB of dynamic shared memory per block is already committed at DKQ=DV=256, so any shape
that wants 3 CTAs has to shrink the Q tile, the KV tile and the combine tile together, not one of
them. Splitting DV further does not open a door either: quarter DV compiles, and it is clean only
where half DV was already clean (ncols=32 at occupancy 3 measures 168 registers and STACK 16 for
both), while the shapes that use the smaller slice to reach 128 registers pay a 48 to 160 byte
frame.

### stream-K cannot be declined on sm89

The Risk section assumed a first cut could "refuse the split for stream-K, the fixup kernel and
the sparse gather". On this machine that is not a scoping option, it is a switch-off:
`ggml_cuda_flash_attn_ext_mma_f16_case` always calls `launch_fattn` with `stream_k = true`
(`fattn-mma-f16.cuh:2112`), and `should_use_stream_k` returns true for every NVIDIA cc at or above
Ada (`fattn-common.cuh:1144`) before it looks at tile efficiency. So the KV range is always
partitioned over `blockIdx.x`, and a DV half has to live alongside that partitioning rather than
replace it.

The seam for doing that is narrow. `dst_tmp_meta` is `blocks_num.x * ncols * (2 + DV/2)` float2
(`fattn-common.cuh:1176`), where the leading 2 float2 hold S and the row max and the rest is the
accumulator slice. Both halves compute identical S and max, so the slot can become
`2*2 + DV/2` float2 per column, each half owning its own meta pair and its own `DV/4` floats, and
`flash_attn_stream_k_fixup_uniform` / `_general` gain one stride to read. No cross-CTA reduction is
added; the halves stay disjoint, which was the property the design section relied on.


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
