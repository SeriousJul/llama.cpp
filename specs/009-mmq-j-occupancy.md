# 009: J as the occupancy knob for the prefill GEMM (sm89)

Status: done (2026-09-24, verdict: no change). Gate 0 passed on paper and the
experiment lost badly, which closes the occupancy route into the biggest kernel and
corrects the reasoning that made it look cheap.
Label: closed
Depends on: 004 (closed as no change; it produced the counter profile this spec
interprets), 001 (shares), 008 (the same register-ceiling story on the other big
kernel)
Scope: sm89, quantized GEMM at prefill batch sizes.

## Problem Statement

`mul_mat_q` is 50.8% of a 65k prefill on the 9B and about 42% of the 27B's
131k pass, which makes it the largest single item on this stack. 004 established
with counters that it is stalled, not saturated:

| metric, `mul_mat_q<23, 128, 0>`, IQ4_XS, ubatch 512 | value |
|---|---|
| LSU pipe | 49.0%, the top pipe |
| tensor pipe | 38.3% |
| DRAM | 23.4%, L2 hit 91.5% |
| issue slots busy | 45.3% |
| achieved occupancy | 16.67%, 8 warps of 48 slots |
| Block Limit Registers | 1 |
| Block Limit Shared Mem | 1 |
| registers per thread | **254** |
| dynamic shared memory per block | 57.86 KB |

Both limits say one CTA per SM, so 004's follow-up note (004b) pointed at the
operand path. What 004 did not look at is that there is already a knob in the
config table that changes the register footprint, and it is the one that decides
the tile shape: `J`, the number of activation columns a block handles.

## The register map, measured

`cuobjdump -res-usage` over the shipping library gives 704 `mul_mat_q`
instantiations. For IQ4_XS, non-fallback rows, SASS entries only (the PTX entries
in the same dump report a meaningless REG of 24 and have to be filtered out):

| J | registers/thread | CTAs of 256 threads that fit in 64K |
|---|---|---|
| 8 | 92 | 2 |
| 16 | 102 | 2 |
| 24 | 120 | 2 |
| 32 | 126 | 2 |
| 40 | 128 | 2 |
| 48 | 130 | 1, at the boundary |
| 64 | 168 | 1 |
| 80 | 168 | 1 |
| 96 | 222 | 1 |
| 112 | 252 | 1 |
| 128, what batch 512 actually uses | **254** | 1 |

A least-squares fit over the instantiations gives

```
REG ~= 88.0 + 0.435 x J
```

The fp32 accumulator alone costs `I x J / (nwarps x warp_size)` = `128 x J / 256`
= 0.5 registers per unit of J. The measured slope is 0.435, so essentially the
whole J-dependence *is* the accumulator, and everything else, operand fragments,
the dequant temporaries, address arithmetic, the y-side, is a fixed cost of about
88 registers per thread.

That gives the arithmetic for what 004 wanted: two CTAs of 256 threads need
`65536 / 512` = 128 registers, which the table reaches at `J <= 40`. Today's
batch-512 dispatch takes `J = 128`, which is the single worst row in the table for
occupancy, and it buys that with 254 of 255 registers.

## Why this trade is cheap on this machine, specifically

Smaller J means more column blocks, and each column block re-reads the same weight
tiles, so weight traffic per GEMM scales as 1/J. That is normally the reason not to
do this. It is exactly what 004 measured away:

- 1.85x the weight bytes cost 3.5% of prefill time (Q8_0 against IQ4_XS on the
  same shapes), and MMQ runs at 12 to 18% of the DRAM roof with L2 hitting 91.5%
- fewer global load instructions changed nothing (-0.58%), so the fetch path is
  not the constraint

So paying in weight bytes to buy warps is a trade where the currency being spent
is the one this kernel has in surplus. Doubling the traffic to halve J moves the
demand from 12% toward 24% of DRAM and takes the machine from 8 to 16 warps per
SM, which is what the 49% LSU and 45% issue figures say is missing.

## Gate 0, run on paper, passes on two rows

Shared memory per block is `mmq_get_nbytes_shared`:

```
nbs_ids + nbs_x + PAD(nbs_y, nthreads*4)
  nbs_ids = J * 4          scales with J
  nbs_x   = I * sram_stride * 4 = 128 * 70 * 4 = 35.0 KiB   J-invariant
  nbs_y   = J * sizeof(block_q8_1_mmq) = J * 144 B           scales with J
```

Anchoring the J-invariant part on 004's measured 57.86 KiB at J=128, which implies
39.4 KiB fixed plus 18.5 KiB of J-dependent term, and putting it against the
measured register counts, the two limits together give:

| J | registers | CTAs on registers | smem per CTA | CTAs on smem | both allow 2 |
|---|---|---|---|---|---|
| 128, today | 254 | 1 | 57.9 KiB | 1 | no |
| 96 | 222 | 1 | 52.9 KiB | 1 | no |
| 64 | 168 | 1 | 48.6 KiB | 2 | no |
| **48** | 130 | 1, misses the 128 ceiling by 2 | 46.6 KiB | 2 | no |
| **40** | **128** | 2 | 45.6 KiB | 2 | **yes** |
| 32 | 126 | 2 | 44.5 KiB | 2 | **yes** |
| 24 | 120 | 2 | 43.4 KiB | 2 | yes |

So the experiment has exactly two serious candidates, J=40 and J=32, and the register
file is the binding constraint, not shared memory. Note how close that is: the current
row is 254 registers and the ceiling for 2 CTAs of 256 threads is 128, which is the
measured count at J=40 to within zero registers.

## Result: the trade is not cheap, and the reason is instructive

`mul_mat_q_switch_J` picks the J that minimises the tile count, so it always lands
on the largest J that fits shared memory, 128 at batch 512. The experiment capped
that loop, which is a one-token change, and measured three balanced rounds of each
test on the 9B with flash attention on and a q8_0 KV cache:

| cap on J | registers | CTAs per SM | pp65536 | pp4096 | tg256 |
|---|---|---|---|---|---|
| 128, shipping | 254 | 1 | 8548.2 t/s | 11310.4 t/s | 151.7 |
| 64 | 168 | 1 | 7734.8 (-9.52%) | 9944.5 (-12.08%) | 151.7 (+0.02%) |
| **40** | 128 | **2** | **6761.2 (-20.91%)** | **8395.6 (-25.77%)** | 151.7 (+0.00%) |

Monotone and large. The occupancy target was reached, the J=40 row measures exactly
128 registers as predicted, and the result is a quarter of the prefill slower.

### Why the Gate 0 arithmetic was not enough

The argument was: 004 measured that 1.85x the weight bytes costs 3.5%, so buying
warps with weight traffic should be nearly free. That conflated two different
things.

- 004's Q8_0 against IQ4_XS test changed the **bytes carried per weight**, at an
  identical number of tiles, an identical number of barrier round-trips, and an
  identical number of dequant invocations.
- Capping J multiplies the **number of passes over the same weight tiles**. Each
  extra column block re-runs the whole K loop: the same number of global load
  instructions per tile, the same LUT dequant, the same scattered shared-memory
  stores, the same `__syncthreads()` sequence. The only thing that gets cheaper per
  unit of that work is the DRAM byte count, which is the one thing 004 already
  proved is not the constraint.

So the cost being multiplied is precisely the per-tile instruction work, and the
benefit being bought, more warps to cover it, is overwhelmed by there being 3.2x
more of it. Decode is untouched, as designed, since batch 1 to 8 goes to MMVQ.

### What this establishes for 004b

This is a negative result with a positive content, and it is the third one in a row
after 004's three probes and 005's config sweep: every route into `mul_mat_q` that
does not reduce the **per-tile instruction count** has now been measured and lost.

- more bytes per tile, cheap (004 probe 1)
- fewer global loads per tile, no effect (004 probe 2)
- more math per tile, saturates at batch 512 (004 probe 3)
- more tiles to buy warps, much worse (this spec)

What is left is the thing all four point at and none of them can fix: the dequant
plus scattered shared-memory store per tile, and the `mma` operand reads that
surround it. 004b, `ldmatrix` operand delivery and fragment reuse so fewer non-mma
instructions sit between tensor operations, is now the only unexplored route into
the largest kernel on this stack. It is a rewrite of `mmq-vec-dot.cuh` and the
tile layout, and 004's fixed-cost measurement of about 88 registers per thread of
operand machinery is the size of the job.

## Design

No new kernel. One step, since Gate 0 cleared:

1. Pin J for the prefill batch to 40 and to 32, measure against the current 128 at
   pp65536 and pp4096 on the 9B, paired. The batch-to-J choice lives in the same MMQ
   layer as the config table, and there is precedent for per-architecture tuning of
   exactly this kind of crossover, which is how the MMVQ-to-MMQ boundary was tuned
   for sm_70.
2. If either wins, make it a sm89 mapping in the table rather than a pin, and
   re-measure the whole matrix, because the winning J is a function of `ne11` and the
   mapping must hold across the batch sizes prefill actually uses.

Accept the result only if the tensor pipe utilization rises, which is the confirmation
that the extra warps reached the tensor cores rather than just the L2. If occupancy
doubles and the tensor pipe does not move, the kernel is LSU-bound in a way that warps
cannot fix, and 004b, the operand rewrite, is the only route left.

## Test Seam and Testing Decisions

The same one as 004, and it needs nothing new: `tests/test-backend-ops.cpp`
`test_mul_mat` on quantized `src0`, which covers the tile shapes and the padded
tails, and it is a pure configuration change so the op results must be *identical*,
not merely within tolerance. J affects the reduction order across the K loop, so
verify against the CPU reference at the existing tolerances and additionally check
greedy model output against the baseline library on a fixed prompt.

Performance is judged outside the seam: paired ABBA at pp65536 (where this kernel is
50.8%) and pp4096, plus one ncu pass on `mul_mat_q` to confirm occupancy and pipe
utilization moved as predicted. Never judge it from an nsys total, for the reason
recorded in the README notes.

## Acceptance Criteria

Not pursued: the experiment was run against the first two and failed them badly.
Recorded so the numbers survive:

- Gate 0, no GPU: passed. J=40 and J=32 are the only rows where the register and
  shared-memory limits both allow 2 CTAs per SM. Registers were the binding limit,
  and J=40 hits exactly 128.
- Occupancy did reach 2 CTAs per SM and the result was -20.9% on pp65536, so
  `Block Limit Registers` improving is not evidence of a win, and only wall time
  settles this class of change.
- The libraries are archived: baseline `b56c071f`, cap 64 `0ba19c7c`, cap 40
  `456bef59`, with the balanced raw data in `/tmp/009/sweep.csv`.

## Rollback and Risk

- The fixed cost is 88 registers per thread even at J=8, so there is a floor below
  which no J reaches 3 CTAs. This is a two-CTAs-or-nothing experiment.
- Shared memory is the co-equal limit and 004 measured it at exactly 1 block per SM
  too. If `tile_x` dominates and does not shrink with J, the register win is dead on
  arrival. This is Gate 0 for exactly that reason.
- Smaller J multiplies the stream-K work, and 004 showed the stream-K fixup kernel
  is already at 1.3% of SM busy on a partial wave; the fixup count grows with the
  block count, so the candidate set must include the fixup cost, not just the main
  loop.
- The MoE `MUL_MAT_ID` path and the `fallback` rows have their own J handling and
  are out of scope, so the change has to be narrow enough not to move them.
- If the answer is "J=128 is right after all", that is a useful result: it closes
  the last cheap route into the biggest kernel and leaves only 004b, the operand
  rewrite.

## Out of Scope

- The operand-delivery rewrite from 004b: `ldmatrix`, fragment reuse, cutting the
  fixed 88 registers. That is the only thing that changes the floor, and it is a
  kernel project, not this.
- `cp.async` pipelining of the weight tiles (004, closed).
- The attention kernels (005, closed; 008, open).
- The SSM kernel (006).
- Any arch other than sm89, and any change to the decode MMVQ path.

## Further Notes

- register map and fit: `cuobjdump -res-usage build-004/bin/libggml-cuda.so.0.25.0`,
  704 MMQ instantiations, SASS rows only
- pipe, occupancy and the two block limits: `/tmp/ncu004.txt`, reproduced in 004
- bytes-do-not-matter and load-count probes: 004, Probe 1 and Probe 2
- shares: `specs/001-baseline.md`, and this session's 9B capture at pp65536 in
  `/tmp/005/kv_q8_0.sqlite`
