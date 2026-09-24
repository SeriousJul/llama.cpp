# 006: Chunked gated-delta-net prefill kernel (sm89)

Status: closed (2026-09-24), verdict: no change to the chunked direction. Increments 1 and 2 stay
committed locally as 42d24e195 (SSM kernel x1.86, 27B pp4096 +7.3 %) and are the only shipped gain.
Stages 1 and 2 were built and are recorded below, then taken out of the tree: the chunked kernel is
preserved as `specs/artifacts/006-chunked-scaffold.patch`, which applies cleanly to HEAD, and
`build-004` now produces a `libggml-cuda.so` bit-identical to HEAD. Stage 3 was not written, for the
reason in "Why this closes".
Label: closed
Depends on: 001 (kernel shares), the qwen35 model shape
Scope: sm89 only. Recurrent kernel stays for decode and for K > 1.

## Probe results (2026-09-23): register prefetch of the token inputs, wall-neutral

First experiment on the path to this spec, before touching the recurrence: stage
token t+1's k/q/v/g/beta into registers before consuming token t, so the serial
chain never waits on a global load. Arithmetic, order, and lane mapping all
unchanged. Patch: 59 insertions, 21 deletions in the CUDA kernel, saved at
`/tmp/004/gdn-prefetch.patch`, reverted from the tree.

Setup: `build-004`, 9B IQ4_XS, `-b 2048 -ub 512`, paired A/B with ABBA order
within each round, md5-verified library swap, SM clock sampled per run
(`/tmp/004/ab.sh`). A = baseline, B = prefetch.

| | A mean | B mean | delta |
|---|---|---|---|
| pp4096 (3 rounds) | 10661.6 t/s | 10656.7 t/s | **-0.05%** |
| tg256 (2 rounds) | 153.16 t/s | 153.21 t/s | +0.03% |

Kernel level, from two nsys captures: `gated_delta_net_cuda` went 347.9 ->
320.8 us per instance, but *every* kernel in the same pair dropped about 4.7%,
so the drift-free reading is that this kernel improved roughly 3% and nothing
else did. 3% of 17% is 0.5% of prefill, which is consistent with the wall being
unchanged.

The unpaired measurement taken an hour earlier said +6.3% on pp1024 and +5.1% on
pp4096. That was entirely clock drift: the same baseline library measured 10140
t/s before and 10661 t/s after, and the SM clock went from the 2.45 GHz the ncu
report recorded to 2700-2715 MHz. See the README workflow notes; all A/B from
here on is paired.

### What this rules out

The token-loop global loads are not the wall. So the first structural hypothesis
for this spec, "the recurrence is waiting on its inputs", is dead, and a chunked
kernel is not justified by load latency. It may still be justified by the
reasoning below, which is now the open question.

Cycle budget for the current kernel, measured vs modelled:

- measured: 347.9 us per (layer, ubatch) for 512 tokens = 680 ns/token =
  ~1840 SM cycles/token at 2.7 GHz
- shape: grid (H=32 heads, 1 seq, 32 column-blocks), block (32,4) = 128 threads,
  so ~8 CTAs per SM = 32 of 48 warp slots, `__launch_bounds__` min 2 blocks
- per warp per token: 8 four-byte global loads (k and q, `rows_per_lane=4`
  strided), 1 broadcast v, 1 g, 1 beta, two dependent butterfly reductions of 5
  shuffles each, about 16 FMA
- modelled dependent chain: roughly 300-400 cycles/token
- modelled L1 wavefront cost: 32 warps x 8 wavefronts = ~256 cycles/token/SM

So the kernel sits 4.6x above its own dependency chain and 7x above its L1
request cost, and hiding the loads changes nothing. Something else is the wall:
the shuffle/MIO pipe, the issue slots of 128-thread blocks, or the store
pattern (only lane 0 writes each output column, so the per-token output write is
1/32 utilized). Counters settle it.

### Facts established for the design, all verified this session

- production dispatch is the **scalar-gate** path: `kda = (src_g->ne[0] == S_v)`
  is false for qwen35, because the graph builds `g` as `[1, H, n_tokens, n_seqs]`
- prefill is **K = 1**, so `keep_rs_t` is false and no per-token state snapshots
  are required. A chunked kernel can therefore be gated on
  `!KDA && !keep_rs_t` and leave 3 of the 4 template combinations untouched
- shapes on the 9B dev model: `ssm.state_size = 128`, `ssm.group_count = 16`,
  `ssm.inner_size = 4096` so H_v = 32 with v_repeat = 2, and
  `full_attention_interval = 4`. On the 27B: 48 of 64 layers are SSM
- the state is stored transposed, `M[col][i] = S[i][col]`, and the output tensor
  packs the attention scores followed by the K state slots
- correctness seam is stronger than 006 assumed: `test-backend-ops
  -o GATED_DELTA_NET -b CUDA0` passes 36/36 on the baseline and already covers
  head sizes 16/32/64/128, `kda` both ways, K = 1..4, `v_repeat`, permuted
  layouts, multi-sequence, and PP-64/256/512/1024
- the current kernel compiles with `LOCAL:0 STACK:0` and 56 registers for the
  S_v=128 scalar-gate instantiation, so there is no hidden spill to explain the
  1840 cycles/token either

### Gate for the next step

Do not write the chunked kernel until the counters name the wall. The counters
are now in, and they are below.

## Counter results (2026-09-23, route A on gated_delta_net_cuda), `/tmp/ncu006.txt`

Kernel `gated_delta_net_cuda<128, 0, 0>`, grid (32, 1, 32), block (32, 4),
314 us per launch, 512 tokens.

| metric | value | reading |
|---|---|---|
| **LSU pipe utilization** | **91.9%** | the wall |
| L1/TEX throughput | 93.2% | same wall, other side |
| `Mem Pipes Busy` | 91.9% | |
| ALU, highest non-memory pipe | 24.1% | fp32 math is nearly idle |
| tensor pipe | 0% | this kernel uses none |
| DRAM throughput | 6.2% | 61 GB/s |
| L2 hit rate | 96.2% | the data is already resident |
| L1/TEX hit rate | 82.9% | |
| active warps per scheduler | 7.79 of 12 | occupancy is adequate |
| achieved occupancy | 64.8% (31.1 warps per SM) | theoretical 83.3% |
| issued warps per scheduler | 0.45 | the pipe is full, issue is not the wall |
| stalls | 36.7% long-scoreboard (L1TEX), 31.0% short-scoreboard (MIO) | both are queueing behind that pipe |
| registers per thread | 47, no spills | |
| waves per SM | 0.80 | grid is under one wave |
| executed instructions | 176.6 M = 84 per token per warp | |

ncu's own summary: LSU is the highest-utilized pipeline at 91.9% and "the
overall pipeline utilization appears to be caused by frequent, low-latency
instructions", with the workload above 80% of available performance so work has
to move off that unit.

### Reading it

Per warp per token the kernel issues roughly 22 memory-pipe instructions: 8
strided 4-byte loads for k and q, 3 broadcast loads (v, g, beta), 1 store from
lane 0, and 10 `SHFL` for the two dependent butterflies. 4096 warps x 512 tokens
x 22 is about 46 M LSU instructions, and at ~2 busy cycles each that is 92 M of
the ~100 M LSU cycles available in a 314 us launch. That closes against the
91.9% measurement, so the model of the kernel is right: it is bound by the
**count of memory-pipe instructions**, and the two stall types are the queue in
front of that pipe rather than distance to DRAM. That is exactly why the prefetch
probe moved nothing.

Two conclusions, and both change this spec:

1. **The serial chain is not the problem.** 31 warps per SM are resident and the
   LSU is saturated; the dependency chain is already hidden. The original framing
   here ("512 serial steps starve the machine") is wrong as stated.
2. **Amortizing loads alone caps low.** Giving each warp C columns divides the
   k/q and misc loads by C, but the butterflies are per column: 10 `SHFL` per
   column-token stays 10 per column-token. At C=4 the LSU instruction count per
   column-token goes ~22 to ~14, so the ceiling on that restructure is about
   1.6x, and it costs 4x the state registers per thread and drops the grid from
   0.8 waves to far less.

### The corrected case for chunking

Chunking is still the right answer, because it removes the LSU *work class*
rather than hiding its latency. Per (head, token) the current kernel performs two
128-row reductions against the state, one per column, each with its own
butterfly. Over a chunk of C tokens those become one 128 x C matrix product per
side, so each loaded k and q element feeds C accumulations instead of one, and
the row reductions collapse into register-resident accumulators instead of
per-token shuffles.

The floor arithmetic says the prize is large. Per (head, token) the state work is
3 x 128 x 128 = 49K FMA, so one (layer, ubatch) is 805 MFMA = 1.6 GFLOP. At the
Ada fp32 roof (128 SM x 128 lanes x 2 x 2.5 GHz = 82 TFLOP/s) that is a ~20 us
floor against 314 us measured. Reaching a quarter of the fp32 roof, which needs
no tensor cores and no change of numeric type, is already 314 to ~80 us: about
4x on 17.2% of prefill, so roughly 12% off pp on the 9B.

Design target: tile the recurrence over (rows x chunk-of-tokens) so every k and q
value is used C times from a register, and drive
`sm__inst_executed_pipe_lsu.sum` per (head, token, column) down from ~22. fp32
SIMT first; tensor cores and the full WY/UT triangular-solve formulation only if
the LSU work is not removable a simpler way.

### One more counter pass before code

To say which half of those 22 instructions to attack first, split the LSU total
between global loads, stores, and shuffles. Command in Further Notes.

## Landed increment 1 (2026-09-23): contiguous row ownership, vectorized staging

**+3.21% on pp4096, paired and reproducible. Kept in the tree.**

What changed: when `S_v == 4 * warp_size` (128 on Ada, the production shape), each
lane now owns 4 *contiguous* state rows instead of 4 rows spaced a warp apart.
That makes k, q and the state itself move as 16 B accesses. Everything else keeps
the original strided mapping through an `if constexpr`, so no other head size or
backend path changes shape.

- the lane-to-row map is one helper used by the state prologue, the k/q staging,
  the KDA gate reads, the snapshot writes and the epilogue, so the mapping cannot
  disagree between them
- the wrapper asserts the q/k row strides stay multiples of 4 floats, on top of
  the 128 B alignment the CUDA allocator already guarantees
- SASS for `gated_delta_net_cuda<128,0,0>` after the change: 3 `LDG.E.128`, 3
  `LDG.E`, 1 `STG.E.128`, 1 `STG.E`, 10 `SHFL`. Per token per warp the load and
  store instructions went 12 -> 6. The 10 shuffles are untouched, which is the
  point of the next increment.
- 48 registers, `LOCAL:0 STACK:0`, `test-backend-ops -o GATED_DELTA_NET -b CUDA0`
  36/36, including the kda and K>1 cases and PP-512/PP-1024

Perf, paired ABBA, 3 rounds, md5-verified library swap, SM clock sampled per run
(`/tmp/004/ab_vec.sh`, A = `cc79ddf3...`, C = `5bc8180c...`):

| | mean | spread |
|---|---|---|
| A baseline pp4096 | 10706.5 t/s | 10682.0 to 10746.3 |
| C vectorized pp4096 | **11050.3 t/s** | 11029.3 to 11078.9 |
| delta | **+3.21%** | every C run above every A run, clocks identical |
| tg256 | A 153.10, C 153.28 | +0.12%, decode neutral |

Against the LSU model this is the expected magnitude: the instruction mix went
from 22 to 16 per token per warp, so the ceiling was 1.375x on this kernel and
the observed gain is about 1.23x, which puts `gated_delta_net` near 14.5% of
prefill instead of 17.2%.

Note the contrast with the prefetch probe above, which was 22 instructions before
and after and gained nothing: cutting LSU *count* works, hiding LSU *latency*
does not. That is the confirmation the diagnosis was right.

## Landed increment 2 (2026-09-23): multi-column warps, the big one

**9B pp4096 +6.22%, 27B pp4096 +7.26%, decode +0.04%, 36/36 op tests green.**
Each warp now owns COLS state columns instead of one. k and q are read once per
warp per token regardless of how many columns it serves, so the biggest item in
the LSU pipe amortizes by COLS, while the butterflies stay per column.

- `col0 = (blockIdx.z * blockDim.y + threadIdx.y) * COLS`, and `grid.z` shrinks by
  COLS, so the column tiling stays exact
- with COLS == 4 the 4 columns are contiguous, so the per-token output write
  becomes one `STG.E.128` from lane 0 instead of four 4 B stores, and the state
  prologue and epilogue are one `float4` per column
- for KDA the per-row gate is hoisted out of the column loop (`eg[r]`), which it
  must be, since the gate is shared by all columns of a head; the value is the
  same one the original code recomputed twice per row per token
- the effective column count in the kernel is `cols_e = kVec ? COLS : 1`, and the
  launcher mirrors that condition (`S_v == 128 && warp_size == 32`), so a Wave64
  or non-128 build compiles and launches the original single-column shape
- `COLS` is a `#define` at the top of the file; for a PR it should become a plain
  constant, since the sweep below shows 4 is not a value that needs runtime choice

SASS for the production instantiation, per token per warp: 2 `LDG.E.128` for k
and q, 4 narrow loads for v, 2 for g and beta, 1 `STG.E.128`, 40 `SHFL`. That is
12.25 LSU instructions per column-token against 16 after increment 1 and 22 at
baseline. 64 registers, `LOCAL:0`.

### COLS sweep, paired ABBA against the same baseline, 9B pp4096

| COLS | pp4096 t/s | vs baseline | LSU per column-token | modelled ceiling |
|---|---|---|---|---|
| 1 (baseline) | 10693 | - | 22 | - |
| 1 + vectorized rows (increment 1) | 11050 | +3.21% | 16 | 1.375x |
| 2 | 11326 | +6.33% | 14.5 | 1.52x |
| **4** | **11358** | **+6.22%** | **12.25** | **1.80x** |
| 8 | 11160 | +4.88% | 11.6 | 1.90x |

2 and 4 are within noise of each other and 8 is clearly worse: past 4 the cost of
4x fewer warps (grid.z shrinks with COLS) and the state registers, 16 to 32 to 64
per thread, outweigh the remaining amortization. The measured gain also stops
tracking the instruction-count model above COLS 2, which says the LSU is no longer
the only pipe in the way at that point.

### 27B confirmation, the production shape

Final artifact after the launcher/kernel condition was unified: lib md5
`52e20b02f06114f14cb91049317fe640`, reproduces the same numbers in a fresh paired
run, 9B pp4096 +6.19% (A [10653..10727], F [11319..11389]), 9B tg256 +0.10%, 27B
pp4096 +7.31% (A [3077.9..3091.9], F [3307.9..3311.1]), 36/36 op tests.

Paired ABBA, 3 rounds, pp4096, `-fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512`,
md5-verified swap, clock sampled per run:

| | mean | spread |
|---|---|---|
| A baseline | 3081.4 t/s | 3073.7 to 3091.3 |
| COLS=4 | **3305.0 t/s** | 3298.2 to 3311.7 |
| delta | **+7.26%** | clean, every run separated |

nsys attribution on the same shape, one rep per capture, so shares are
not clock-dependent:

| | baseline | COLS=4 |
|---|---|---|
| `gated_delta_net` per call | 500.4 us | **268.8 us** = x1.86 |
| `gated_delta_net` share of prefill | 14.85% | 8.56% |
| `mul_mat_q` share (untouched control) | 64.40% | 69.24% |

The share arithmetic implies +7.38% on pp from the GDN change alone against the
+7.26% measured, so the attribution closes. Decode on the 9B is +0.04%, which is
noise: at n_tokens 1 the loop body is entered once and the column tiling only
changes the shape of the state prologue and epilogue.

### What is left in this kernel

At COLS=4 the LSU budget per column-token is 12.25 instructions of which 10 are
shuffles. That is now the whole remaining cost, and it is why COLS=8 regressed
rather than continuing to help: the loads have almost nothing left to give.

- shortening the butterfly needs fewer lanes per column, which costs state
  registers per thread and blocks per SM at the same time, so it is a trade, not a
  win
- removing the butterfly entirely is the tensor-core chunk form in the design
  below: `mma` does the row reduction in the tensor pipe, which the profile shows
  is currently at exactly zero

The kernel is now at 268.8 us per (SSM layer, ubatch) against the ~20 us fp32
floor, so 13x of headroom is still on the table, but it is no longer an LSU
instruction-count problem. Anything past this point is the chunked restructure.

## Next candidate: the shuffle width, closed as negative by the instruction mix

The idea was to shorten the butterfly by giving each column fewer lanes, which trades
`SHFL` for `LDG` and looked like a 1.2 to 1.6x win on the LSU budget. The static mix
of the shipping kernel says the trade now runs the other way:

| group | count | share |
|---|---|---|
| FFMA, FADD, FMUL | 117 | 37.1% |
| IADD3, LEA, IMAD, MOV, SHF | 108 | 34.3% |
| **SHFL** | **40** | **12.7%** |
| **LDG** | **12** | **3.8%** |
| total | 315 | |

The loads are already down to a third of the shuffle count, because increments 1 and
2 put the k and q reads into one 16-byte access amortized across four columns. Going
from 32 lanes per column to 16 halves the shuffles, 40 to 32, which saves 8 of the 52
LSU-side instructions, and it doubles the loads, 12 to 24, which costs 12. Net LSU is
worse by about 8%, before counting the extra address arithmetic. Going to 8 lanes per
column is worse again, and it also needs 16 state registers per thread instead of 4.

So the layout is at a local optimum, and the reason is the same one 010 found for MMQ:
the reduction is done by moving data between lanes, which is LSU work, and the only
way out is to stop doing reductions in the lane dimension at all. That is the chunked
tensor-core kernel, where `mma` reduces inside the tensor pipe. The increment line on
this kernel is finished.

### The pre-measurement note, kept for the record

Written before the instruction mix above was taken. Its reasoning was right about the
trade being two-sided and wrong about the direction, which is why the section above
measured it instead of building it.

After increment 2 the LSU budget per column-token is 12.25 instructions, of which
10 are shuffles. The loads have almost nothing left to give, so the only SIMT
lever remaining is the reduction width, and the COLS sweep already showed what it
costs: past COLS=4 the shrinking grid hurts more than the amortization helps.

- 16 lanes per column, 8 rows per lane: 4 butterfly steps per dot, so 8 `SHFL`
  per column-token instead of 10
- 8 lanes per column, 16 rows per lane: 3 steps per dot, 6 `SHFL`

Costs, both of which bit: state registers per thread go 4 to 8 to 16, and the
grid shrinks by the same factor, from the 0.8 waves it already has. At 32 warps
per SM and a saturated pipe the loss of blocks may be affordable, but this needs
a measurement per step, not a leap. Expected value if it works at the 8-lane
setting: LSU 16 -> ~10, so up to 1.6x on the kernel and 4-5% more on pp.

This is also the point where fp32 SIMT stops paying: the state rows per thread is
what buys the shorter butterfly, and it is bought from the register file, which
is what caps the blocks per SM. The chunked tensor-core form in the design below
removes the row reduction from the LSU pipe instead of shortening it.

## Chunked form: algebra validated, and the cost model forks the design

### The derivation, checked in host code

Throwaway validator at `specs/artifacts/006-chunk_check.cpp` (plain C++, not in the build):
one head, S_k = S_v = 128, fp32, implements the recurrent form exactly as the CUDA
kernel does and the chunked form below, and compares them. Inputs match what
qwen35 actually feeds the op: q and k L2-normalized by `build_gdn_l2_norm`
(`src/models/qwen35.cpp`), so |q| = |k| = 1, beta through a sigmoid, gate
`g = exp(-|n| * 0.05)` in (0, 1).

Per chunk of C tokens, with S0 the state at chunk start and gam_t = prod_{s<=t} g_s:

```
T[t][s]  = beta_t (k_t . k_s) gam_t/gam_s          for s < t, strictly lower
rhs_t    = beta_t v_t - beta_t gam_t (S0^T k_t)
delta    = (I + T)^-1 rhs                          unit lower triangular solve
out_t    = scale [ gam_t (S0^T q_t) + sum_{s<=t} (q_t . k_s)(gam_t/gam_s) delta_s ]
S_end    = gam_C S0 + sum_s (gam_C/gam_s) k_s (x) delta_s
```

Results, 4096 tokens through 64 chunks, three seeds and three chunk sizes:

| C | max abs output error | max abs state error | max |out| |
|---|---|---|---|
| 32 | 4.5e-8 | 3.0e-7 | 0.0749 |
| 64 | 4.3e-8 | 3.6e-7 | 0.0749 |
| 128 | 4.5e-8 | 3.0e-7 | 0.0749 |

That is fp32 rounding agreement, so the derivation, the decay bookkeeping and the
unit-lower-triangular solve are right, and chunk size is not chosen by numerics
over this range. C=64 is the pick for the smem and register budget below.

Note the earlier failure of this test before the inputs were fixed: with
unnormalized k, `k_t . k_s` is about 32 instead of 1, `(I + T)^-1` explodes, and
both forms overflow to inf. The rule is only stable because the model normalizes
k, which is also why `T` is well-conditioned in practice. Any future change that
drops the normalization breaks this kernel as much as it breaks the recurrent one.

### The cost model, which is the bad news

Counting the matmuls the chunked form needs per (head, chunk) at C=64, dk=dv=128:

| term | FLOP |
|---|---|
| KS = K S0 | 2.10 M |
| QS = Q S0 | 2.10 M |
| K K^T (strictly lower half only) | 0.52 M |
| Q K^T (lower plus diagonal) | 0.79 M |
| triangular solve over dv columns | 0.52 M |
| (I+T)^-1 applied to rhs | 1.05 M |
| decay-weighted intra-chunk term into out | 1.05 M |
| state update K^T delta | 2.10 M |
| total | **10.2 M** per head-chunk |

That is 2.7 GFLOP per (SSM layer, ubatch of 512) on the 9B, against the 1.6 GFLOP
the recurrent form does: the chunked form buys parallelism by spending **1.67x
more FLOPs**.

Put those against the roofs with the current kernel as the reference, 268.8 us per
(27B layer, ubatch) on 1.6 GFLOP, which is 6 GFLOP/s:

| implementation | FLOP needed | roof | at 35% efficiency | gain over today |
|---|---|---|---|---|
| chunked, fp32 SIMT | 2.7 GFLOP | 82 TFLOP/s | 94 us | 2.9x |
| chunked, f16 mma fp32 accum | 2.7 GFLOP | 165 TFLOP/s | 47 us | 5.7x |

So the fork is real and it is a decision, not an implementation detail:

- **fp32 SIMT chunked** keeps numerics where they are (validated above at 4e-8)
  and buys about 2.5-3x. It is a large new kernel for that, and it is the same
  order as what increments 1 and 2 already got with 144 changed lines.
- **f16 operand chunked** is where the 5-8x lives, because the tensor pipe is the
  only thing above the fp32 roof. It needs the four state matmuls (KS, QS, the
  inverse application, and the state update) in `mma.sync` with f16 operands and
  fp32 accumulate, and it puts roughly 1e-3 relative error on those terms, which
  is a perplexity question, not a kernel question.

### Numerics gate for f16 operands: closed, go

`specs/artifacts/006-chunk_f16.cpp` runs the chunked form with every operand of the four big
matmuls rounded to half and the accumulation in fp32, which is what
`mma.sync ... f16.f16.f32` does, against the fp32 recurrent reference. The state is
rounded to half between chunks, since holding it half precision is the point.

First result was garbage (87% error) and was my own bug: the half-rounding helper
scaled the value by a factor of two and used the wrong quantization grid. Fixed, the
answer is stable and it is good:

| per-token decay | state retained across one 64-token chunk | out err | state err after 64 chunks |
|---|---|---|---|
| 0.670 (fast) | 7.6e-12 | 0.04% | 0.05% |
| 0.961 | 0.077 | 0.04-0.05% | 0.04-0.05% |
| 0.996 | 0.774 | 0.05% | 0.04% |
| 0.9996 (slowest) | 0.975 | 0.05-0.06% | 0.04-0.05% |

Relative error does not grow as the gate slows, which is the accumulation question
that mattered: at 97.5% state retention per chunk over 64 chunks the error is still
~5e-4 relative, the same as at fast decay. The delta rule projects the state rather
than compounding it, so operand error does not integrate.

For scale, IQ4_XS weights already carry roughly 1e-2 relative error and a q8_0 KV
cache more than that again. 5e-4 on the SSM path is below the noise floor the model
runs in, so the f16 tensor-core version looks safe on numerics grounds and the
perplexity run is a confirmation, not a gamble.

### Decision

Build the tensor-core chunked kernel. The fp32 SIMT variant is not worth its size
(2.9x at 35% efficiency against 5.7x with mma, for the same new kernel), so the
plan is:

1. one kernel, block per (head, sequence), looping over 64-token chunks, state
   resident in shared memory as f16 (128 x 128 x 2 B = 32 KB, which fits where the
   fp32 state at 64 KB would not)
2. the four big terms as `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
   KS, QS, the inverse applied to rhs, and the state update
3. `T` and `Q K^T` over the 64-token chunk also as mma, since they are 128-deep
   dot products of unit-norm vectors, and the strictly-lower masking is a predicate
   on the accumulator, not extra work
4. the unit-lower triangular solve as fp32 SIMT over the 64 x 64 block, done once
   per chunk because it is dv independent, 0.52 M of the 10.2 M FLOP
5. selected by `!KDA && !keep_rs_t && n_tokens >= 64 && S_v == 128`; every other
   shape keeps the recurrent kernel that increments 1 and 2 just made 1.86x faster

First measurable milestone is correctness: the structure, the smem layout and the
solve, checked by `test-backend-ops -o GATED_DELTA_NET`, which already covers
head_size 128, both gate kinds, K 1 to 4, and PP up to 1024. Perf is only read
after that, because a chunked kernel that is wrong is also a chunked kernel that is
fast.

## Shared-memory gate, measured 2026-09-24: the obvious tile plan does not fit

Before writing the kernel I priced its shared memory with a stub in
`specs/artifacts/006-tile_budget.cu`, one block per (head, sequence) holding the state and the
chunk tiles, and ptxas answered directly:

```
ptxas error: Entry function uses too much shared data (0x12080 bytes, 0xc000 max)
```

Three things come out of that single line:

1. The layout as first sketched, state f16 32 KB plus Q, K and V tiles at 16 KB each
   plus the 64 x 64 `T`, is about 104 KB. Ada allows 99 KB per block by opt-in and
   48 KB statically, so this must be dynamic shared memory with
   `cudaFuncSetAttribute`, the same pattern MMQ and the flash-attention kernels
   already use, and even then it does not fit. It needs aliasing.
2. What aliasing buys: Q and K are consumed at different times so they can share one
   buffer, and `T` can live in V's space once V has been folded into `rhs`. That lands
   around 72 to 80 KB per block, which is **one block per SM**, at 8 warps with a
   256-thread block. Not fatal, FA already runs the tensor pipe at 57.7% with 8 warps
   per SM, but it means the design has no slack and every later idea that wants shared
   memory competes with a single resident block.
3. The device numbers, from `cudaGetDeviceProperties` on this card, turn the choice
   from an opinion into a table. sm_89 gives 48 KiB static, 99 KiB per block after
   `cudaFuncSetAttribute`, and **100 KiB per SM**, which is what actually decides how
   many blocks are resident. Layout is state 32 KiB plus two 16-bit operand tiles of
   `C x 128` plus `max(V, T)`:

| C | shared per block | blocks per SM | warps per SM at 256 threads | serial chunks for 512 tokens |
|---|---|---|---|---|
| 16 | 44 KiB | **2** | **16** | 32 |
| 32 | 56 KiB | 1 | 8 | 16 |
| 64 | 80 KiB | 1 | 8 | 8 |

   A correction to my own earlier note here: I leaned to `C = 32` for the headroom,
   and headroom is not what it buys. 56 KiB still fits only one block per SM, because
   two would need 112 KiB, so it takes the mma-efficiency penalty of a small chunk and
   the occupancy penalty of a resident single block at the same time. On these numbers
   the real choice is `C = 16`, the only size where a second block fits and where the
   layout also drops under the 48 KiB static ceiling so no opt-in call is needed at
   all, or `C = 64`, which maximizes the tile shape but pins one block per SM. Nothing
   in between is interesting.

   That is the whole trade in one line: occupancy on this kernel can only be bought
   with `C = 16`, and `C = 16` pays for it in serial chunks, 32 of them per ubatch
   against 8, and in smaller matmul tiles. Stage 3 should measure both, and the
   numerics work already says chunk size is free from 16 to 128, so there is no
   quality reason to prefer either.

Note the distinction from 008, where a *register* budget estimated from declarations
was wrong by 3.5x: this is shared memory, which is declared arithmetic, and ptxas
checked it. The register side of the same question still needs a real compile of the
real kernel, and that stays the gate on stage 3.

## Stage 1 landed (2026-09-24): the scaffold, flag off by default

`gated_delta_net_chunked_cuda<S_v, C, LANES>` in `ggml/src/ggml-cuda/gated_delta_net.cu`, plus its
launcher, behind `GGML_CUDA_GDN_CHUNKED` which defaults to 0. It is built only when the flag is set,
and no caller selects it at either setting. What it contains is the geometry, the shared-memory
layout with its aliasing, the input staging, and the chunk loop; the algebra is the gap the comments
name, in the order stage 2 fills it. No numerical claim.

Gate results, all from `specs/artifacts/006-gate.sh`, which is rerunnable:

| check | result |
|---|---|
| arch 89, flag off | 0 chunked symbols in the object; device SASS and resource usage identical to the committed tree, 9585 lines byte for byte |
| arch 89, flag on | builds clean; every *shipped* kernel's device code still identical to the committed tree; `test-backend-ops -o GATED_DELTA_NET -b CUDA0` 36/36, which turned out to be vacuous for this kernel: see the stage 2 section |
| Wave64 | see the caveat below; builds clean at 64 lanes, flag off and on, with master as a control in the same run |
| registers, from the compile | all four instantiations (C = 16/64 x LANES = 32/64) at `REG:31 STACK:0 LOCAL:0`. This is a floor, not a budget: the matmul loops that own the registers are not there yet |
| shared memory, from the compile | `SHARED:0` because the tiles are one dynamic request; the size is `ggml_cuda_gdn_chunked_smem<S_v, C, DV_TILE>::bytes`, pinned by static_assert so the layout cannot drift out from under the gate table without a compile error |

Baseline library for the later paired runs: `build-004` at flag off, `libggml-cuda.so.0.25.0` md5
`878c6cb4ae574c7b0e92087ba9bcce33`, printed by step 7 of `specs/artifacts/006-gate.sh`. Treat that
number as perishable. It moves whenever a line is added above the wrapper: the flag-off object and
the same file at HEAD differ in 26 bytes out of 187264, which are ten 2-byte `__LINE__` immediates
that `GGML_ASSERT` bakes into the wrapper below the insertion point, each shifted by exactly the
number of lines added, plus nvcc's temp-file name in `.strtab`. Byte identity of the flag-off build
is therefore unreachable for anything inserted above the wrapper, and only because of assert line
numbers; device code identity is reachable, and that is what the gate asserts.

### The smem gate, now measured on the device

`specs/artifacts/006-launch-smem.cu` sets the attribute, launches, and asks the driver for resident
blocks, so the gate table is no longer arithmetic alone:

| C | smem | launch | blocks per SM, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
|---|---|---|---|
| 16 | 45056 B | ok | **2** |
| 32 | 57344 B | ok | 1 |
| 64 | 81920 B | ok | 1 |

Device read back as 48 KiB static, 99 KiB opt-in, 100 KiB per SM, as the gate recorded. C = 32 is
confirmed pointless. The C = 16 versus C = 64 choice stays open for stage 3.

### What stage 1 found: the grid is the binding constraint, not shared memory

The design is one block per (head, sequence) with the chunk loop serial, so the grid is `H_v x n_seqs`.
On the 9B that is 32 x 1 = **32 CTAs on a 128 SM card**, read off the GGUF (`ssm.inner_size` 4096 /
`ssm.state_size` 128 = 32 value heads, `full_attention_interval` 4). Blocks-per-SM capacity is then
not the limit at any C: the kernel is grid-limited to a quarter of the machine, 8 warps resident on
32 SMs and 96 SMs idle. The recurrent kernel it replaces runs 256 CTAs of 4 warps.

Cost, on the 9B shape at C = 64 (2.62 GFLOP per layer-ubatch, `avail` = 165 TFLOP/s x CTAs/128):

| state columns per block | CTAs | total FLOP | avail TFLOP/s | floor at 100 % | at 35 % | at 20 % |
|---|---|---|---|---|---|---|
| 128 (the frozen shape) | 32 | 2.62 G | 41 | 63.5 us | 181 us | 317 us |
| 64 | 64 | 3.09 G (+18 %) | 82 | 37.4 us | 107 us | 187 us |
| 32 | 128 | 4.02 G (+54 %) | 165 | 24.4 us | **70 us** | 122 us |
| 16 | 256 | 5.90 G (+125 %) | 165 | 35.7 us | 102 us | 179 us |

35 % is the efficiency the cost model above already used for the mma form. Read against the
268.8 us the shipping kernel takes per (layer, ubatch), the frozen shape tops out near 1.5x, so the
4x acceptance target is out of reach at one block per (head, sequence) no matter what C is. The extra
slices pay for their own duplicated FLOPs: 1.83 M of the 10.23 M per head-chunk is dv-independent
(`K K^T`, `Q K^T` and the solve), so a slice recomputes it, and the 32-column row is the optimum
because it is the first one that fills all 128 SMs.

That is the Risk 2 fallback 006 already named: split the state columns across `grid.z`. The smem side
of it is favourable too, because the state tile shrinks with the slice while the k and q tiles do
not (they are `dk` wide, and `dk` is not split):

| columns per block | C | state | k + q | V or T | total | blocks per SM capacity | CTAs |
|---|---|---|---|---|---|---|---|
| 128 | 64 | 32 KiB | 32 KiB | 16 KiB | 80 KiB | 1 | 32 |
| 64 | 64 | 16 KiB | 32 KiB | 16 KiB | 64 KiB | 1 | 64 |
| 32 | 64 | 8 KiB | 32 KiB | 16 KiB | 56 KiB | 1 | 128 |
| 32 | 16 | 8 KiB | 8 KiB | 1 KiB | 17 KiB | 5 | 128 |

So at C = 64 the split buys CTAs, not residency: 1 block per SM either way, but the third row is the
first one that reaches all 128 SMs. Column capacity only turns into extra warps at a smaller C, where
the k and q tiles shrink with it. Stage 3 should sweep columns alongside C, not C alone.

### The Wave64 check, and what it does not prove

This box has no ROCm (`/opt/rocm` absent, no hipcc, no `hip/hip_runtime.h`), so a genuine gfx9 build
cannot be run here. `specs/artifacts/006-wave64-compile.sh` compiles the file against a copied header
tree with `ggml_cuda_get_physical_warp_size()` pinned to 64, which is the value a wave64 build hands
every kernel in the file, and it controls itself by compiling the committed tree the same way. Both
chunk sizes at both lane widths then get a real ptxas pass. What it does not cover: the HIP-only
branches elsewhere in `common.cuh`, and anything that would only show up under the AMD assembler. A
CI HIP build is still the proof for those.

## Stage 2 landed (2026-09-24): the fp32 SIMT chunk body, and the seam was lying

The algebra is written and green behind the flag: `gam`, `KS`, `QS`, `rhs`, `T`, `delta` by forward
substitution, the output and the state update, as plain fp32 SIMT matmuls over the resident state.
The wrapper now selects the chunked path when the flag is on, for `!kda && !keep_rs && n_tokens >= C
&& S_v % DV_TILE == 0`. Built-in defaults: C = 16, `DV_TILE` = 32, 256 threads.

| gate | result |
|---|---|
| op suite through the chunked path | 40/40, 8 consecutive runs inside `006-gate.sh` plus 45 more by hand: zero failures |
| registers, from the compile | `S_v=128, C=16, LANES=32`: REG 46, STACK 0, LOCAL 0; same at LANES 64 |
| flag off | device code still byte-identical to the committed tree, 0 chunked symbols |
| Wave64 | still builds, flag off and on |
| wider suite | `GATED_DELTA_NET,SSM_SCAN,CPY` 302/302 |
| real model | 9B pp1024 completes with the path live |

Directional only, and not a measurement: pp1024 came out 9199 t/s with the flag on against
11542 +- 539 with it off, from single unpaired runs at different `-r`. Stage 2 is not the
deliverable and its speed is not a result; perf is read in stage 4.

### The seam was vacuous, and how that was caught

Stage 1's 36/36 proved nothing. The only `head_size = 128` cases with a long sequence live in
`make_test_cases_perf()`, which is a timing list and not a correctness check; in the checked eval
set, the `head_size = 128` cases carry 1 token and the 64..256-token cases carry `head_size = 64`.
The earlier claim that the seam covers the production shape was true of the two lists separately and
false of their intersection. It was caught by counting launches with a `fprintf` in the launcher,
not by reading the registration list, and 4 parameters were added to the eval set at d = 128 to
cover 64, 65 (ragged), 512 (32 chunks) and `n_seqs = 2`, plus `v_repeat = 2`, which is the
production broadcast. Anyone re-running this gate should keep trusting the launch count over the
case list.

### f16 tiles sit exactly on the seam's tolerance: a flake, not a race

The first stage-2 draft used the f16 tiles the mma design calls for, and the suite failed
intermittently, about one full run in fifteen: `ERR = 1.09e-7` against `max_nmse_err = 1e-7`, on the
65-token d = 128 case. `compute-sanitizer --tool racecheck` reported 0 hazards, which is what
pointed at tolerance rather than a data race. The mechanism, confirmed later in `tests/test-backend-
ops.cpp`: `init_tensor_uniform` seeds from `std::random_device` per process, so the inputs are a
fresh draw every time the binary starts. What I first wrote here (that the stream advances with the
number of cases run ahead of the failing one) was wrong, and the control that showed it was
pointless: the same case, in the same binary, re-run in a second process, moved from 9.2e-8 to
9.6e-8. A single green suite run is one sample, not a proof.

The arithmetic says it was no coincidence: `sqrt(1e-7) = 3.2e-4`, which is f16 epsilon. An f16
operand path sits on this seam's boundary by construction. Stage 2 therefore uses fp32 tiles, which
is also what "plain fp32 SIMT matmuls" asks for, and the failure rate went to zero in 53 runs.

**What this forces on stage 3.** An f16-operand mma form meets the same wall, so the plan's gate ("op
tests green at the existing tolerances") is not satisfiable as written. The tree already holds the
precedent for handling it: `test_ssm_scan` overrides `max_nmse_err()` to 2e-7 because its SSD path
uses fp16 intermediates, and 5e-4 relative on the SSM path is below the noise floor an IQ4_XS model
runs in. See the next subsection for the measured split between the two rounding sources, because it
determines the layout, not just the tolerance.

## Stage 3 opened (2026-09-24): the SIMT form measured, and it loses

Two things were measured before any mma line was written, and both changed the plan.

### The f16 operand layout, measured and then withdrawn

The f16 tile version (state and the k / q operands in f16, accumulators and the solve in fp32) was
built, run per case one process at a time, and measured: across the shapes the chunked path can take
the error lands at 8.0e-8 to 9.9e-8 NMSE, with the 1.09e-7 tail seen earlier, against the seam's
1e-7. That is 80 % to 99 % of the budget on every draw, which is why it failed one run in fifteen:
it is not a marginal case, it is the whole distribution sitting on the bar.

It was then reverted out of the tree, and the tolerance override was reverted with it. The reason is
the flag: with `GGML_CUDA_GDN_CHUNKED` off, which is the shipped state, those same cases run the
recurrent fp32 kernel and measure 4e-15, so a 2e-7 bar would be masking nothing that the strict bar
would have caught, while quietly weakening a shared test for a configuration that is currently slower
than what it is meant to replace. The measurement is kept here instead, and stage 3 applies the f16
tiles together with the mma operands and the documented tolerance, in one change that is worth the
loosening.

The dtype switch itself is three numbers and two helpers, so it is not lost: `state` and `kq` in
`ggml_cuda_gdn_chunked_smem` become `sizeof(half)`, `s_state` / `s_k` / `s_q` become `half *`, the
tile stores go through a `store4` that packs two `half2` with `__floats2half2_rn`, the loads through
a `load4` that unpacks with `__low2float` / `__high2float`, and the state update rounds once on the
way back into the resident tile. `delta`, `out`, the C x C scratch and the gate stay fp32.

### Paired ABBA: the chunked SIMT form is slower than the shipped kernel

Two rounds, `A B C C B A` per round so the baseline brackets both variants inside each round, library
md5-verified before every run, clocks sampled per run and pinned at 2700 MHz throughout, 9B IQ4_XS,
`-b 2048 -ub 512`, `pp4096` and `tg256` as separate rows:

| config | mean pp4096 | vs baseline | mean tg256 |
|---|---|---|---|
| A: flag off, recurrent kernel | 11430 t/s | - | 151.6 t/s |
| B: chunked, C = 16, fp32 tiles | 10282 t/s | **-10.0 %** | 152.0 t/s |
| C: chunked, C = 64, fp32 tiles | 9115 t/s | **-20.2 %** | 153.7 t/s |

Every A run beat every B run and every B run beat every C run, so the ordering is not noise. Decode
moves by less than the spread of the baseline, as it must: `n_tokens >= C` keeps it on the recurrent
kernel.

What this settles:

1. The 2.9x the cost model offered for a chunked fp32 SIMT form is not reachable by writing SIMT
   matmuls over the state. The chunked form spends 1.67x more FLOPs than the recurrent one, and in
   SIMT those FLOPs are paid for with the same LSU and issue pressure that increments 1 and 2 were
   tuned against. The extra work only becomes cheap in the tensor pipe, so stage 3's mma swap is not
   an optimization on top of a working kernel: it is the thing that decides whether this direction
   exists at all.
2. Do not carry this C ordering into stage 3. C = 16 beating C = 64 by 13 % is an occupancy result:
   with fp32 tiles, C = 64 at `DV_TILE = 32` asks 112.5 KiB and does not fit, and with f16 tiles it
   asks 74.2 KiB for 1 block per SM against C = 16's 21.6 KiB for 4. Under mma the tensor pipe wants
   the opposite trade, bigger tiles and fewer serial chunks, so the sweep belongs after the swap and
   has to be run again, not inherited.

### Where the 571 us is not: the substitution's barriers, ablated

The instance count closes against the structure before anything is read from the capture: 384 calls
= 192 per pass = 24 SSM layers x 8 ubatches, on the chunked kernel. Inside one capture, the kernel
runs 571.0 us per (layer, ubatch) and takes 28.1 % of prefill, against `mul_mat_q` at 56.7 % as the
untouched control.

The chunked SIMT form executes 1.978 GFLOP per (layer, ubatch) at C = 16 with the 4-way column split
(0.483 MFLOP per CTA-chunk, counted term by term), so 571.0 us is **3.46 TFLOP/s**: 4 % of the fp32
SIMT roof and 2 % of the f16 mma roof. The recurrent kernel after increments 1 and 2 does 1.61 GFLOP
in 320.8 us, which is 5.02 TFLOP/s. Neither form is anywhere near a compute roof, so this whole
kernel is about operand delivery, and that reframes what stage 3 has to clear:

| goal | us per call | TFLOP/s needed | share of the 165 TFLOP/s f16 roof |
|---|---|---|---|
| match the recurrent kernel | 320.8 | 6.2 | 4 % |
| 2x | 286 | 6.9 | 4 % |
| 4x, the acceptance target | 143 | 13.9 | 8 % |

35 % efficiency, the number the cost model used, would be 58 TFLOP/s. The target only needs 8 %,
and fattn is measured in 005 at 57.7 % of the tensor pipe, so the arithmetic in the mma form is not
the risk. Getting the operands to it is, which is the wall 004, 009 and 010 all hit on MMQ.

Before writing mma loops I priced the one hypothesis that would have made them partly pointless: the
forward substitution takes one `__syncthreads()` per row, so 16 barriers per chunk and 512 per
CTA-call. An ablation build that hoists that barrier out (numerically wrong, kept for one measurement
and reverted the same session, `A B D D B A` x 2 rounds, md5-verified, `/tmp/006-ab/D`) came out
within about 1 % of the correct build while the session's own clock drift was about 2 % (the baseline
itself slid 11558 -> 11284 as SM clocks went 2715 -> 2700). So the barrier chain is not the wall,
and a blocked solve is not the fix. The cost is the SIMT loops themselves, which is the case for
moving the six products into the tensor pipe.

### Stage 3's gate is open: the tensor path clears the requirement, and the layout came from mma.cuh

`specs/artifacts/006-mma-rate.cu` holds f16 tiles in shared memory, moves them through `ldmatrix`
into fragments and accumulates with `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on the five
shapes the six chunk products reduce to, and it validates against a CPU matmul before it times
anything.

The first version failed validation, and the failure is worth recording: I wrote the B operand load
as `ldmatrix.x2.trans`, and with `A` set so `D[i][n] = (i+1)(n+1)` every lane's two column registers
came back identical, so both halves of the fragment got the same `n` (`006-mma-rate-diag.cu` shows
it in four structured probes). `mma.cuh`'s `tile<8, 8, T>` `load_ldmatrix` uses `ldmatrix.x2` with
**no transpose**, on the same address pattern I already had. One word, and the check passes at
2e-6 max abs diff, which is f16 rounding of the reference. The lesson is the one 006 keeps teaching:
these layouts come from the shipped code, not from reading PTX tables.

Measured, 8 warps, 256 blocks, fp32 accumulate:

| product | shape | TFLOP/s | share of the 165 f16 roof |
|---|---|---|---|
| KS, QS | M64 N32 K128 | 40.2 | 24 % |
| `T`, `Q K^T` | M64 N64 K128 | 54.0 | 33 % |
| inverse application, intra out | M64 N32 K64 | 39.4 | 24 % |
| state update | M128 N32 K64 | 53.1 | 32 % |

Against the 13.9 TFLOP/s that 4x needs, and the 3.46 the chunked SIMT form achieves, the tensor path
has roughly 3x margin on its weakest shape. So the direction is not dead: stage 3 is worth writing.
Read these as an upper bound on operand delivery, not a prediction of the kernel. The probe runs one
product at a time with no global traffic after the tile load, no barriers in the loop, no staging and
no epilogue, while the real kernel alternates shapes inside one shared-memory budget, which is where
the remaining risk lives.

### The rate depends on the CTA count, and that picks the configuration

`006-mma-rate.cu` re-run at the CTA counts a config actually gets, rather than the 256 blocks a
throughput probe likes:

| shape | 256 blocks | 128 blocks | 64 blocks | 32 blocks |
|---|---|---|---|---|
| KS/QS M64 N32 K128 | 40.2 | 39.2 | 19.5 | 9.8 |
| `T`, `Q K^T` M64 N64 K128 | 54.0 | - | - | 13.3 |
| state update M128 N32 K64 | 53.1 | 51.6 | - | 12.9 |

The rate is flat down to 128 CTAs (one per SM) and then falls roughly linearly, because one block per
SM cannot hide its own `ldmatrix` latency. So `DV_TILE = 128`, the frozen one-block-per-(head,
sequence) shape, is not a smaller version of the same design: at 32 CTAs it lands at 10 to 13 TFLOP/s,
which is below the 13.9 that 4x needs even before the split duplication is counted.

Putting that curve against the FLOPs each config executes, including the duplicated `T` / `Q K^T` /
solve per column slice (2.62 to 4.29 GFLOP per call as the slice count goes 1 to 4), is what actually
settles the design point:

| C | DV_TILE | CTAs | GFLOP/call | TFLOP/s for 4x | achievable | verdict |
|---|---|---|---|---|---|---|
| 16 | 32 | 128 | 2.28 | 28.5 | ~39 | **4x reachable** |
| 32 | 32 | 128 | 2.95 | 36.8 | ~39 | 4x marginal, 2x safe |
| 64 | 32 | 128 | 4.29 | 53.6 | ~39 | **2x at best** |
| 64 | 128 | 32 | 2.68 | 33.5 | ~13 | no |

This contradicts the plan's frozen `C = 64`, and it is worth being exact about why: C = 64 does not
fail because its tiles are wrong, it fails because a wider chunk makes the dv-independent products
bigger, the column split then duplicates them, and the duplication has to be paid for at one CTA per
SM. The shared-memory gate was right that occupancy is the only lever; it was wrong to think of C as
free. Stage 3 should therefore be written and measured at `DV_TILE = 32` with C = 16 as the primary
and C = 32 as the alternative, and the `C = 64` row kept only as a checked prediction that it loses.

### Stage 3 call map, read out of the shipped kernels rather than derived

Verified against `fattn-mma-f16.cuh` (tile typedefs at 1085-1095, the KQ product at 680-716, the V
product at 1036) so that stage 3 is transcription. ggml's f16 mma entry is
`mma(tile<16,8,float> & D, const tile<16,8,half2> & A, const tile<8,8,half2> & B)` (mma.cuh 1216),
A is loaded with `ldmatrix.x4` and B with `ldmatrix.x2`, both untransposed, and strides are counted
in half2 units.

| product | A operand | B operand | notes |
|---|---|---|---|
| KS = k @ S0 | k tile `[t][DK]`, row-major | state tile `[c][DK]`, read as column-major | both `load_ldmatrix`, no transpose |
| QS = q @ S0 | q tile `[t][DK]` | state tile `[c][DK]` | same |
| `T` = k k^T | k tile `[t][DK]` | k tile `[t'][DK]` | one tile, two roles |
| `Q K^T` | q tile `[t][DK]` | k tile `[t'][DK]` | |
| delta = mat @ rhs | scratch `[t][s]` | rhs `[c][s]` | inner dim is C, not DK |
| out intra = A @ delta | scratch `[t][s]` | delta `[c][s]` | |
| state update = k^T @ delta | **k tile via `load_ldmatrix_trans`** | delta `[c][t]` | the only transposed operand; fattn 1036 is the precedent (V^T for the output product) |

Two consequences the table makes concrete, both of which change stage 2's layout rather than just
its inner loops:

- The rhs, delta and the two C x C scratches become **f16 tiles with the `[n][k]` row-major form**,
  because an mma operand cannot be read out of an fp32 tile. Stage 2 keeps them fp32, and the state
  update then also wants delta transposed per column. So stage 3 is not only "swap the loops", it
  re-lays the scratch tiles.
- **delta-as-f16 is the one numerics question still unanswered.** `006-chunk_f16.cpp` rounded the
  operands of the four big products, and `006-mma-rate.cu` measured the f16 state and k / q tiles at
  8.0e-8 to 9.9e-8, but neither rounded `delta`, which in the chunked form carries the result of the
  fp32 solve and is then consumed twice (the output term and the state update). It is cheap to
  measure the same way as before, and it should be measured before the tolerance value is fixed,
  because if it dominates, the fix is to keep an fp32 copy of delta for the state update and let the
  mma read the f16 one, which costs smem.
- ggml pads the B tile row stride: `stride_tile_Q = DKQ/2 + 4` (fattn 696). My rate probes used no
  padding, so their TFLOP/s are a pessimistic lower bound on the shape, not an optimistic one. Use
  the pad.

### Why this closes (2026-09-24)

Measured, paired ABBA, clocks pinned at 2700 MHz throughout: the chunked form is **-10.0 %** on 9B
pp4096 at C = 16 and **-20.2 %** at C = 64, against the shipped recurrent kernel's 11430 t/s. Stage 3
would have to recover that with the tensor pipe, and the case for it did not survive contact with the
instruments:

| config | mma shape | GFLOP per call | TFLOP/s for 4x | rate actually measured |
|---|---|---|---|---|
| C = 16, `DV_TILE` = 32 | M = 16 tiles, skinny | 2.28 | 28.5 | **never measured**, see below |
| C = 64, `DV_TILE` = 32 | M = 64, good | 4.29 | 53.6 | 39 to 54, ceiling not headroom |
| C = 64, `DV_TILE` = 128 | M = 64, good, 32 CTAs | 2.68 | 33.5 | 16.4 combined, 10 to 13 per shape |

C = 64 fails on duplicated dv-independent work. C = 16 fits the FLOP budget but needs `M = 16` tiles,
which is the one shape `006-mma-rate.cu` cannot express: its 8-warp grid asserts `N` is a multiple of
64 when `M` is 16, so every one of its 39 to 54 TFLOP/s figures came from `M` of 32 or 64. The
recommendation earlier in this file to build stage 3 at C = 16 was arithmetic-only and the
measurement that would support it does not exist. The whole-chunk probe that was supposed to settle
it kept deleting its own mmas: four separate causes found and fixed (doubled row offset, 32-bit
`ldmatrix` address against `mma.cuh`'s 64-bit `"l"`, a conditional keep-alive nvcc removed, operands
loop-invariant across chunks) and a fifth still open, where every config except `DV_TILE = 128`
compiles to 272 instructions of staging with zero HMMAs.

**Reopen condition, one number.** Widen `006-mma-rate.cu`'s warp grid to take `M = 16` (`WARPS_M = 1`
with `N` a multiple of 64, or 4 row tiles across `M = 64` stacked products) and measure
`M16 N32 K128`. If it lands near the 39 TFLOP/s that `M64 N32 K128` reaches, C = 16 is alive and
stage 3 is worth a session. Under roughly 20, the direction is dead on measurement, not on suspicion,
and nothing further should be spent on it.

### What is kept, and what to know before using it

- `specs/artifacts/006-chunked-scaffold.patch` holds stages 1 and 2: the fp32 SIMT chunk body,
  `DV_TILE = 32`, C = 16, the wrapper hook, and the 4 static asserts that pin the layout arithmetic.
  It applies to HEAD clean. With it applied, `006-gate.sh` checks arch 89 both flag states, Wave64,
  registers from the compile, and 8 suite runs; that script is only meaningful against the patch, so
  do not read a failure in it as a regression in the shipped tree.
- Stages 1 and 2 were green while in the tree: 40/40 op suite in 8 consecutive runs with the flag on,
  0 failures in 53 runs after the f16 experiment, device code byte-identical to HEAD with the flag
  off, Wave64 building both states, REG 46 STACK 0 LOCAL 0.
- Kept in `tests/test-backend-ops.cpp` on purpose: 5 chunk-edge parameters at `head_size = 128`
  (64, 65, 512 tokens, `n_seqs = 2`, `v_repeat = 2`), which close a real hole in the checked seam
  found here, and the 6 `ggml_scale_mode` / `ggml_scale_flag` casts that were producing warnings. Both
  pass, 62/62 with UPSCALE and INTERPOLATE included.
- The method lessons went to `specs/README.md`: a green suite is not proof the path ran, per-process
  reseeding makes a marginal tolerance look like a flake, read mma layouts out of `mma.cuh`, and a
  tensor probe must prove its own work survived.

### The whole-chunk probe is broken; one row of it survives, and it agrees

`specs/artifacts/006-mma-chunkbody.cu` runs the seven-product chunk in one kernel at the CTA count a
config really gets, which is the measurement that would replace the per-shape weighting above. It is
not trustworthy: three bugs came out of it, and a fourth is still in there.

1. Double offset: `prod()` added the row-slice offset internally while callers also added it to the
   base, so the A address was offset twice. Fixed.
2. Address width: I passed `ldmatrix` addresses as `(unsigned) __cvta_generic_to_shared(...)` with an
   `r` constraint. `mma.cuh` uses a 64-bit value with an `l` constraint. The truncation faulted at
   15 KiB of shared memory while the same cast went unnoticed in `006-mma-rate.cu`. Fixed.
3. Deleted work: with only some accumulator elements reaching the keep-alive test, nvcc removed whole
   chunk loops. Rates came out at 470 and 4054 TFLOP/s on a 165 TFLOP/s card, which is how I caught
   it, and the instruction count confirmed it.
4. Still open: after all three fixes, only 2 of the 8 instantiated kernels contain any HMMA at all.
   The other six still have their loops deleted, and no number from them is used in this spec.

The two kernels with plausible counts are the `DV_TILE = 128` ones (176 HMMA for C = 16 against ~168
expected). That single credible row says C = 16, `DV_TILE = 128`, 32 CTAs reaches **16.4 TFLOP/s** at
114.6 us per (layer, 512 tokens). The validated per-shape probe, with no such bugs, independently put
32-CTA shapes at 9.8 to 13.3 TFLOP/s. Two artifacts, different structure, same conclusion: the frozen
one-block-per-(head, sequence) shape lands at roughly a third of what its own 4x needs, and the
verdict in the table above stands on the per-shape measurement, not on this one.

### Two bugs the measurement tools caught in my own process

`specs/artifacts/006-err-probe.sh` and `specs/artifacts/006-ab.sh` exist because reading a case name
next to an error line is not a measurement. Between them they caught: a `sed` pattern that needed one
character where the value had two, so a "C = 64" run had silently rebuilt C = 16; a library copied
from a tree still sitting on the previous configuration, which the md5 check turned into an obvious
A == C collision; a first f16 comparison that paired errors with the wrong cases and came out
identical to fp32; and `-pg 4096,256`, which is one combined row, not two.

### What f16 actually costs at this seam, measured per configuration

The case-by-case pairing above is only trustworthy one case per process, so each configuration was
rebuilt and run with `-p '.*<case>.*'` on the 512-token d = 128 case, which is the worst of the set
at 32 chunks of state carry. `nbytes` from the launcher probe identifies the layout each build asked
for, so a stale object cannot masquerade as a result:

| build | smem request | NMSE | share of the 1e-7 budget |
|---|---|---|---|
| all fp32 tiles (stage 2 as landed) | 38016 B | 6.19e-12 | 0.006 % |
| f32 state, f16 k and q | 29824 B | 4.71e-8 | 47 % |
| f16 state, f16 k and q | 21632 B | 8.48e-8 | 85 % |

Two conclusions, and the second one is the reason not to take the obvious shortcut:

1. Rounding the operands is what spends the budget, not rounding the state. Carrying the state in
   f32 halves the error (8.48e-8 to 4.71e-8) but does not rescue it: f16 `k` and `q` alone already
   sit at 47 % of the tolerance, before `delta` and the v operand go to f16 as stage 3 intends.
   Keeping the state in f32 to save the seam is therefore the wrong lever; it costs 8 KiB per block
   and buys a factor of two on a budget that is already spent.
2. A per-case documented tolerance is the only route that survives contact with the seam, and the
   `test_ssm_scan` precedent puts it at 2e-7, which covers the worst observed case (1.09e-7) with
   room. The f16 chunked form should get the same treatment: an override on the chunked cases, a
   comment naming the reason, and the perplexity run as the gate that actually guards quality.

So stage 3 keeps the f16 state (its 32 KiB at C = 64 is what makes the layout fit at all), and the
tolerance is a documented, measured exception rather than a number to be discovered by a failing CI
matrix. The 5e-4 relative that `006-chunk_f16.cpp` measured stands as the quality argument; the
perplexity run in stage 3 is what has to confirm it.

### fp32 tiles change the smem budget, which is why stage 2 builds C = 16

| columns per block | C | fp32 tiles (stage 2) | f16 tiles (stage 3) |
|---|---|---|---|
| 32 | 16 | 37.1 KiB, 2 blocks/SM | 21.1 KiB, 4 blocks/SM |
| 32 | 64 | 112.5 KiB: does not fit | 72.5 KiB, 1 block/SM |
| 128 | 16 | 97.1 KiB, 1 block/SM | 57.1 KiB, 1 block/SM |
| 128 | 64 | 208.5 KiB: does not fit | 144.5 KiB: does not fit |

The fp32 form fits only at `DV_TILE = 32, C = 16`, which is what stage 2 builds; the C = 16 versus
C = 64 measurement in stage 3 is a comparison between f16/mma configurations, and the fp32 form here
is a correctness vehicle, not one of the candidates.

### The gate has to be carried in the log domain

`s_gam[t]` accumulates the raw `g` and every ratio is `expf(s_gam[t] - s_gam[s])`. The product form
`gam_t / gam_s` is not usable: the long-sequence cases draw `g` uniform in [-20, -1e-4], so inside a
16-token chunk the cumulative product reaches `exp(-320)` and flushes to zero, making the ratio 0/0.
The host validators missed this because they used a mild decay. Production feeds a small negative `g`
where the two forms agree to rounding, so this is a test-data hazard that would otherwise have
surfaced as NaNs in stage 3.

### One thing the compile caught that reading would not

The first draft scanned the cumulative gate with `warp_prefix_inclusive_sum`, and the wave32
instantiation reported `STACK:16` and `STACK:24` while the wave64 one reported `STACK:0`. A
single-lane scan over at most C values removed it and cost nothing: 46 registers, 0 stack, 0 local.
The stack only appeared in the instantiation that actually runs, and only a compile says so.

## Execution plan for a fresh session

Everything needed to resume is in the tree: the validators are
`specs/artifacts/006-chunk_check.cpp` (algebra) and `006-chunk_f16.cpp` (f16 operand
numerics), the sizing probes are `006-tile_budget.cu` and `006-layout.cu`, and the
counter dumps are `specs/artifacts/tmp-ncu006.txt` (the 91.9% LSU result) and
`tmp-ncu006b.txt` (the opcode split). The paired A/B harness pattern is
`specs/artifacts/004-ab_vec.sh`. Rebuild `build-004` from `master` for the baseline
library; the ones under `/tmp` are not archived.

The kernel is not written. This is the order to write it in, with the verification
step that closes each stage, sized so no stage has to be finished in one sitting.

1. **Scaffold, off by default.** New chunked kernel in the CUDA GDN file, selected
   only by a compile-time flag, nothing wired to the dispatcher. Gate: it compiles
   for arch 89 and for a Wave64 build, and the tree behaves identically with the flag
   off. No numerical claims yet.
2. **Structure with fp32 SIMT inner loops.** *Landed 2026-09-24, see the stage 2 section above.*
   Chunk loop with the state resident in shared memory, the cumulative gate products, `T =
   strictly_lower(beta_t (k_t.k_s) gam_t/gam_s)`, the unit-lower triangular solve, `delta = (I+T)^-1
   rhs`, the output and the state update, all as plain fp32 SIMT matmuls. This is *not* the
   deliverable, 004's cost model says fp32 SIMT tops out near 2.9x; it is the cheapest way to make
   the algebra and the bookkeeping fail loudly in a form that is easy to read. Gate:
   `test-backend-ops -o GATED_DELTA_NET -b CUDA0` fully green with the flag on. It was 36 cases and
   is now 40, because the checked set had no long-sequence case at the production head size; see the
   stage 2 section before assuming a green suite means a exercised kernel.
3. **Swap the six products to `mma.sync.m16n8k16.row.col.f32.f16.f16.f32.`** *Gate opened
   2026-09-24: the operand path sustains 39 to 54 TFLOP/s against the 13.9 that 4x needs, so this is
   worth writing.* Use `mma.cuh`'s `MMA_Traits`, `load_ldmatrix` and `mma` wrappers, not inline asm:
   the B operand is `ldmatrix.x2` with no transpose, and the shipped wrappers are the only reason
   that took one iteration instead of one session. Keep the triangular solve in fp32 SIMT, it is
   0.52 M of the 10.2 M FLOP. Two things land with it, not after it: the f16 operand tiles need the
   documented CUDA-scoped tolerance in the same change (measured 8.0e-8 to 9.9e-8 against a 1e-7
   bar), and the C ordering from the SIMT sweep must not be carried in. Gate: op suite green with the
   flag on, then a perplexity run on the 9B and the 27B against the committed baseline, with the f16
   error expectation of 5e-4 relative already measured in `specs/artifacts/006-chunk_f16.cpp`. If the
   mma form does not clear the recurrent kernel's 11430 t/s on pp4096, this spec closes as no
   change.
4. **Measure, then decide to keep.** Paired ABBA against `build-004` at pp4096 and
   pp65536, plus an ncu pass for LSU instructions per (head, token, column) against
   today's 12.25 and for tensor-pipe utilization. Target from the floor arithmetic is
   4x on this kernel, which is 12% off pp on the 9B and more on the 27B. Sweep the state
   columns per block alongside C = 16 versus C = 64: at the frozen one-block-per-(head,
   sequence) shape the grid is 32 CTAs against 128 SMs, which caps the kernel near 1.5x
   (the table in the stage 1 section), so blocks-per-SM capacity is not the binding
   constraint and C on its own cannot reach the target.

Things already settled, do not relitigate: the algebra is validated to 4e-8 against
the recurrent form and does not depend on chunk size over 32 to 128; f16 operands
hold at 5e-4 relative with 97.5% state retention across 64 chunks; qwen35 is the
non-KDA path with K = 1 in prefill, so the gate is
`!KDA && !keep_rs_t && n_tokens >= 64 && S_v == 128`; the state is stored
transposed, `M[col][i] = S[i][col]`; and the production tile is 32 heads x 128 state
columns with `scale = 1/sqrt(S_v)`.

The one thing that can go wrong in a way that wastes a session: registers and shared
memory. The state as f16 is 32 KB per block, the chunk operands are a few KB more,
Ada gives 100 KB per SM, and the existing recurrent kernel runs at 64 threads per
block with 4 warps and reaches 31 warps per SM. Size the block before writing the
matmul loops, and check the register count with `cuobjdump -res-usage` after each
stage, not at the end.

## Problem Statement

Most of the layers in this model are not attention, they are the gated
delta-net SSM, and its CUDA kernel runs the recurrence one token at a time with
no parallelism over the sequence. On a hybrid model where 48 of 64 layers are
SSM, that kernel is the second largest consumer of prompt processing.

From 001:

| phase | gated_delta_net share | instances | average |
|---|---|---|---|
| pp32768 (P2) | 13.1% = 9.4 s of 71.4 s | 18480 | 508 us |
| pp131071 x6 (D2) | 9.4% = 37.9 s of 402 s | 73776 | 514 us |

The instance counts resolve to one call per (SSM layer, ubatch): 18480 = 6
passes x 64 ubatches x 48 layers. So 508 us buys one 512-token ubatch through
one layer, which is about 1 us of critical path per token per layer.

The kernel itself says why (`ggml/src/ggml-cuda/gated_delta_net.cu`):

- `launch_gated_delta_net` has the comment `//TODO: Add chunked kernel for even
  faster pre-fill`. This is a known gap upstream, not a mystery.
- The grid is `(H, n_seqs, ceil(S_v / num_warps))` with a `(32, 4)` block. For
  the qwen35 shape (32 value heads, S_v = 128) that is 1024 blocks of 128
  threads. There is no dimension over sequence position, so the whole ubatch is
  walked by every block.
- Inside `gated_delta_net_cuda`, the state shard lives in registers
  (`s_shard[rows_per_lane]`) and the token loop is a dependent chain: each step
  needs a reduction across the state rows to form the delta, then updates the
  state, then emits the output. About 0.68 us per step at 2.7 GHz is roughly
  1840 SM cycles per token, which is 4.6x above the cost of its own dependency
  chain and far above its L1 request cost. The wall is therefore not the input
  loads (see the probe results above); it is one of the shuffle pipe, the issue
  slots, or the serial chain itself, and the counters decide which.
- The 32 blocks along grid.z for one head all read the same q, k, v, g, beta row.
  The redundant broadcast is a second known gap, tagged in `ggml/include/ggml.h`
  as `[TAG_GGML_GDN_BCAST]`.

The result: the largest non-GEMM cost in prefill of a hybrid model is spent on a
serial chain that the hardware cannot fill. It also hits the case attention does
not: at short and mid context, where fattn is small, the SSM kernel is the
dominant non-GEMM work in time to first token.

## Solution

Process prefill in chunks. Within a chunk of C tokens the recurrence unrolls
into a small dense triangular product, which is tensor-core work and is
computed in parallel; between chunks only the S_v x S_v state is carried, so the
serial chain becomes `n_tokens / C` steps instead of `n_tokens`.

This is the published chunked delta-rule form (the same shape the upstream
training libraries use for this layer class), not a new algorithm. The output
must remain the same values the recurrent kernel produces, within the tolerance
`test-backend-ops` already applies to `GGML_OP_GATED_DELTA_NET`.

From the operator's point of view: shorter time to first token on this model
class, more so on short and mid prompts, with no flag and no cache change. From
the maintainer's point of view: one new kernel beside the existing one, chosen by
a condition in the CUDA op wrapper, with the recurrent path untouched for
decode.

## User Stories

1. As a llama-server operator, I want prompt processing on hybrid SSM models to
   use the tensor cores, so that time to first token stops scaling with the SSM
   recurrence.
2. As a llama-server operator, I want short prompts (1k-8k) to be faster, since
   that is where the SSM share is highest.
3. As a llama-server operator, I want decode unchanged, so that tokens per
   second does not move when I pick up a build with this in it.
4. As a llama-server operator, I want spec-decode state rollback (K > 1) to keep
   using the kernel that is correct today, so that a prefill optimization cannot
   break draft rejection handling.
5. As a user of a hybrid model, I want quality unchanged, so that perplexity on
   wikitext-2 lands where 003 already measured it.
6. As a llama.cpp contributor, I want the chunk size to be a single fixed
   constant for sm89, so that there is no auto-tuning matrix to maintain.
7. As a llama.cpp contributor, I want the recurrent kernel left in place and
   selected by one condition in the op wrapper, so that other arches and every
   decode path are untouched by this diff.
8. As a llama.cpp contributor, I want the triangular solve or UT-transform
   handled with the primitives that already exist in ggml, so that the kernel
   does not invent a linear-algebra subsystem.
9. As a reviewer, I want the numerics stated: which products run in f16 with
   fp32 accumulate and which stay fp32, so that I know where drift can come
   from.
10. As a reviewer, I want the state written back in the same slot layout as
    today, including `state_slot_stride`, so that the KV-cache memory code needs
    no change.
11. As a person running the 9B dev model, I want the kernel A/B to be one
    llama-bench row set, so that a chunk size can be tried in a few minutes.
12. As a maintainer of other backends, I want the CPU, Metal, Vulkan, and
    Hexagon paths untouched, so that this stays an Ada-scoped experiment until
    the numbers argue otherwise.
13. As a maintainer of the vec/MMQ kernels, I want this spec to not touch the
    GEMM or attention paths at all, so that 004 and 005 can land independently.

## Implementation Decisions

- New kernel file-adjacent to the current one, selected inside the CUDA
  `GGML_OP_GATED_DELTA_NET` wrapper. Condition for the chunked path:
  `n_tokens >= 64`, `K == 1`, single sequence per stream, non-permuted
  broadcast-free layout, sm80 and newer. Everything else keeps the recurrent
  kernel. `K == 1` is the important guard: per the op contract in
  `ggml/include/ggml.h`, K > 1 must publish per-token state snapshots, and the
  chunked form does not produce those cheaply.
- Chunk size C = 64 fixed for this spec. Rationale: it is the smallest power of
  two that gives a 64x parallelism gain over the current chain, it maps onto
  `mma.sync m16n8k16` tiles without padding at S_v = 128, and it matches what
  the reference implementations use. C is a template parameter so a second value
  is a recompile, not a redesign.
- One block per (chunk, head, state-column-tile). Grid gains the chunk dimension,
  which is what fixes the occupancy problem; the intra-chunk products are
  S_v x C and C x C, both tensor-core sized.
- The inter-chunk state pass stays serial over chunks and carries only the
  S_v x S_v state, in shared memory rather than in registers, so that the
  dependency chain is C tokens long instead of 1 token long.
- Intra-chunk math uses f16 operands with fp32 accumulation, which is the
  fastest lane Ada offers with the accumulator width GGML needs (003: FP16 with
  fp32 accum is 165 TFLOPS; the fp16-accum vendor figures are not usable).
  Gate and beta handling stay fp32.
- Do not fold the `[TAG_GGML_GDN_BCAST]` q/k/v broadcast fix into this change.
  With chunks in place the redundant reads move from 32 blocks x 512 tokens to
  32 blocks x 8 chunks, which is already a 64x reduction of the problem, so the
  broadcast work would only muddy the A/B. Record it as 006b.
- The output tensor packs attention scores followed by K state snapshots. The
  chunked kernel writes slot 0 (final state) and the score block only, and the
  wrapper asserts `K == 1` before selecting it.

## Test Seam and Testing Decisions

One seam, already built and already covering the production shape:
`tests/test-backend-ops.cpp`, `test_gated_delta_net`.

- The registered cases include `test_gated_delta_net(GGML_TYPE_F32, 32, 128,
  512, 1)` and `(..., 1024, 1)`, commented as the Qwen3.5-like shape (32 heads,
  d = 128), plus 4-head variants and KDA variants. Those are the acceptance
  tests for the chunked path; no new file, and only parameters added.
- Good tests here assert the op output against the CPU reference, for both the
  `K == 1` and `K > 1` shapes, and for both selection outcomes at the
  `n_tokens` boundary (63 and 64), so that the switch condition cannot silently
  disagree with the wrapper.
- Add parameters that hit chunk edges: `n_seq_tokens` at 64, 65, 128, 192, 512.
  A chunked kernel is wrong at the ragged end first.
- Add a multi-sequence case with different positions per sequence, since the
  state slot stride and the per-sequence reset are where a chunked rewrite
  breaks.
- Correctness of the fast path is also checked against the recurrent CUDA
  kernel, at the same op seam, with CUDA as the reference backend. That is the
  check that catches a formulation error the CPU reference might hide.
- Performance is judged outside the seam, per 001's method: llama-bench pp rows
  and one nsys pass to attribute the gain to this kernel rather than to noise.

## Acceptance Criteria

Baseline is 001: gated_delta_net at 508-514 us per (layer, ubatch) call, 13.1%
of pp32768 and 9.4% of pp131071.

- 9B dev model, `test-backend-ops -o GATED_DELTA_NET -b cuda` green, 5
  consecutive runs, including the new edge cases.
- Per-call kernel average for the chunked path at n_tokens = 512: <= 170 us
  (3x), stretch 85 us (6x). Read from nsys, not from a derived number.
- pp2048 and pp4096 on the 27B: >= 6% faster than baseline (at SSM share 13.1%
  and a 3x kernel gain the expected value is about 8.7%).
- pp131071: >= 4% faster than the 1905 t/s baseline.
- tg128 and tg2048 within 1% of baseline, and the profile shows the recurrent
  kernel still running for decode.
- Verify steps under spec decode: profile shows the recurrent kernel selected
  (n_tokens 2-8), and the 002 verify-step attention sums are unchanged.
- wikitext-2 perplexity on the 9B within +/- 0.02 of the 8.169 +- 0.055 IQ4_XS
  baseline from 003, and the sign of the delta is explained.
- VRAM unchanged (the chunk state is per-block shared memory, not a new global
  buffer). If an implementation needs a global chunk-state buffer, that is a
  design change and must come back for review before it is built.

### Status against acceptance (2026-09-23)

The two landed increments deliver part of the goal without the chunked rewrite:

| criterion | state |
|---|---|
| SSM kernel faster | done, x1.86 per call on the 27B, x1.5 inferred on the 9B |
| pp4096 on the 27B >= 6% faster | **done, +7.26%** paired with clean separation |
| decode within 1% | done, +0.04% on 9B tg256 |
| op tests green | done, 36/36 including KDA, K>1 and PP-1024 |
| committed | done, 42d24e195 on a local branch, not pushed |
| pp2048 on the 27B | not measured yet |
| pp131071 on the 27B | not measured yet, and this change is context-independent so it should carry as a constant share |
| wikitext-2 perplexity within 0.02 | **open**, no corpus cached on this box; the reduction order changed (lane-to-row mapping, and the KDA gate is now computed once per row instead of twice), so this is the check that must pass before the change is proposed upstream |
| 3x kernel gain from chunking | not reached; 268.8 us per call against a ~20 us fp32 floor, so ~13x is still open |

## Rollback and Risk

- Risk 1: numerics. The chunked form changes the summation order, so results
  differ in the last bits, and a hybrid SSM model can amplify that over 48
  layers. Mitigation: the perplexity gate above, plus a greedy-decoding token
  comparison against the baseline library on a fixed prompt. If the drift is
  visible at all, the chunk products move to fp32 and the gain is re-measured.
- Risk 2: shared memory. S_v = 128 gives a 128 x 128 fp32 state (64 KB) plus the
  C x C intra-chunk tiles, on a 100 KB Ada SM. Two blocks per SM may not fit. If
  it does not, either the state stays in registers per warp-group or the
  per-block state is split the way the current kernel splits columns. This is
  the main design risk and it is Ada-specific, so it is the first thing to
  prototype.
- Risk 3: the chunked path is slower than expected because the intra-chunk
  triangular work is not actually tensor-core efficient at C = 64 and S_v = 128.
  Mitigation: prototype at C = 64 before writing the selection logic, and keep
  the condition in the wrapper so a losing kernel never runs in production.
- Risk 4: interaction with the `K > 1` snapshot contract in the memory layer.
  Mitigation: the `K == 1` guard, and a test case at the boundary.
- Rollback: one condition in the op wrapper. Delete it and the previous behavior
  is exact.

## Out of Scope

- Attention kernels and GEMM (004, 005).
- Fusing q/k/v projection GEMMs that feed the SSM layer, and the conv/scanning
  elementwise work around it.
- The `[TAG_GGML_GDN_BCAST]` tiled-versus-interleaved broadcast question (006b).
- Any non-Ada backend, and any change to the CPU reference implementation.
- K > 1 paths, so nothing about spec-decode state rollback.
- A new ggml type, a new op, or a change to the op's tensor contract.

## Further Notes

Next measurement, before any code: split the LSU instructions between global
loads, stores, and shuffles, because that decides which of the two structural
fixes the chunked kernel must carry.

```sh
systemctl --user stop llama-server.service

sudo -E ncu \
  --replay-mode application \
  --kernel-name regex:gated_delta_net --launch-count 1 --launch-skip 8 \
  --metrics sm__sass_inst_executed_op_global_ld.sum,sm__sass_inst_executed_op_global_st.sum,sm__sass_inst_executed_op_shared_ld.sum,sm__sass_inst_executed_op_shared_st.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_tensor.sum \
  /home/seriousjul/src/llama.cpp/build-004/bin/llama-bench \
    -m /home/seriousjul/bench-fp8/qwen35-9b-iq4_xs.gguf \
    -ngl 99 -p 4096 -n 0 -r 1 -b 2048 -ub 512 2>&1 | tee /tmp/ncu006b.txt

systemctl --user start llama-server.service
```

The number the redesign is judged against is `sm__inst_executed_pipe_lsu.sum`
per (head, token, column); today it is about 22 per warp-token. If the shuffle
count is not exposed by these metric names, `sm__inst_executed_pipe_lsu.sum`
versus the global-ld sum gives the same answer by subtraction.

Next command to run (route A, root, no counters permission needed for the user):

```sh
systemctl --user stop llama-server.service

sudo -E ncu \
  --replay-mode application \
  --kernel-name regex:gated_delta_net --launch-count 2 --launch-skip 8 \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats --section InstructionStats \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  /home/seriousjul/src/llama.cpp/build-004/bin/llama-bench \
    -m /home/seriousjul/bench-fp8/qwen35-9b-iq4_xs.gguf \
    -ngl 99 -p 4096 -n 0 -r 1 -b 2048 -ub 512 2>&1 | tee /tmp/ncu006.txt

systemctl --user start llama-server.service
```

The lines that decide the design: the highest-utilized pipe from
`Compute Workload Analysis` (watch for the MIO / LSU pipe and the shuffle
share), `Warp State Statistics` stall reasons, `Active Warps Per Scheduler`,
and `Achieved Occupancy` from a section that reports the per-instruction mix.

Evidence pointers (line numbers as of `bddf8263c`, upstream/master):

- `ggml/src/ggml-cuda/gated_delta_net.cu`: the TODO in
  `launch_gated_delta_net`, the grid and block shape in the same function, and
  the per-token serial body in `gated_delta_net_cuda`
- `ggml/include/ggml.h`: `ggml_gated_delta_net` contract, the q/k/v/g/beta/state
  shapes, the output packing with K state snapshots, and
  `[TAG_GGML_GDN_BCAST]`
- `tests/test-backend-ops.cpp`: `test_gated_delta_net`, the cache-fusion variant,
  and the registered Qwen3.5-like PP-64/256/512/1024 cases
- model shape and shares: `specs/001-baseline.md` (hardware, production stack,
  P2 and D2 tables)
- Ada tensor-core rates with fp32 accumulate: `specs/README.md`, hardware
  section, and `specs/003-fp8-prefill-gemm.md`
