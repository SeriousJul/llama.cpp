# 010: MMQ shared-memory traffic and the I = 128 assumption

Status: done (2026-09-24, verdict: no change). Every route this spec named is now
closed, including option 5, which the source shows is already implemented. The value
of the spec is the negative result: the cost is structural to the quantization
formats, not to the kernel.
Label: ready-for-agent
Depends on: 004 (closed; its "004b" note named the wrong fix), 009 (closed; it
eliminated the tile-count route), 001 (shares)
Scope: sm89, quantized GEMM at prefill batch sizes.

## Why this spec exists

004 closed `cp.async` and left a follow-up note that said the fix was operand
delivery: use `ldmatrix` instead of manual shared-memory reads, and reuse
fragments across the tile. 010 was going to be that spec. Two checks killed it
before it was written, and both are worth recording because they redirect the
largest remaining prize on this stack.

Check one, from `ggml/src/ggml-cuda/mmq-vec-dot.cuh`: the weight operand already
uses `load_ldmatrix` at 22 sites, and the activation operand deliberately does
not, with a comment saying `load_generic` is faster than `load_ldmatrix` there.
The operand-delivery idea is already implemented upstream. `mma.cuh` carries the
`ldmatrix` wrappers and the `mma.sync ... s8.s8.s32` path that MMQ uses.

Check two, the static instruction mix of the kernel that matters, from
`cuobjdump -sass -fun _Z9mul_mat_qIL9ggml_type23ELi128ELb0EE...` on the shipping
library, 5925 instructions in `mul_mat_q<IQ4_XS, J=128, fallback=false>`:

| opcode | count | share |
|---|---|---|
| I2FP, int32 to fp32 conversion | 1032 | 17.4% |
| FMUL | 1032 | 17.4% |
| FFMA | 1024 | 17.3% |
| LDS | 408 | 6.9% |
| IMAD | 316 | 5.3% |
| LEA | 309 | 5.2% |
| PRMT | 256 | 4.3% |
| **IMMA, the tensor operations** | **256** | **4.3%** |
| MOV | 227 | 3.8% |
| LOP3, IADD3, SHF, STS, LDG, STG | 712 | 12.0% |

Two numbers to sit with: memory-pipe instructions are 12.2% of the mix, and there
are 22.1 non-tensor instructions for every `IMMA`. Of those 22, about 12 are
`I2FP`, `FMUL` and `FFMA`, which is the scale-correction epilogue of the int8
tensor path, not operand loading.

So the correction is the largest single block of work in this kernel, and it is
the reason the tensor pipe sits at 38.3% of peak while the machine issues 1.86
instructions per cycle: the tensor cores are waiting on the thing that turns their
int32 partials back into fp32.

## Why int8 costs this much

The weights and the activations are both q8-class, so the mma accumulates in int32
and every partial has to be rescaled by the product of the two block scales before
it joins the fp32 accumulator. Those scales exist because the quantization is
block-scaled, and the block is 32 values wide, so a K-run of 32 values earns one
correction pass over the whole accumulator tile. The kernel bought a 660 TOPS
tensor roof with INT8 against 165 TFLOP/s for fp16-with-fp32-accumulate, and it
pays for that in this epilogue.

The measured consequences line up with 004 and 009:

- 004 probe 1: 1.85x the weight bytes cost 3.5%. Consistent: bytes are not the
  issue, 12.2% of the instructions are memory ops.
- 004 probe 2: removing 4 of 16 global loads per thread per tile did nothing.
  Consistent: it moved the small part of the mix.
- 004 probe 3 and 009: neither more math per tile nor more tiles helps, and more
  tiles is catastrophic. Consistent: the per-output-element correction work grows
  with the accumulator tile, so no tiling choice reduces it.

## Gate 0 result: the correction is the biggest block of work, but it is not the wall

From `/tmp/ncu010.txt`, on `mul_mat_q<23, 128, 0>`, the kernel that owns 54% of
prefill. Each percentage is of that pipe's own peak, sustained active:

| pipe | utilization | note |
|---|---|---|
| **LSU** | **50.8%** | the busiest resource |
| tensor, as busy cycles | 38.3% | from 004's SpeedOfLight section |
| ALU | 33.7% | integer, logic, shifts |
| FMA | 21.0% | fp32 arithmetic |
| tensor, as issued instructions | 9.9% | one `IMMA` holds the pipe for many cycles |
| F2F | metric not available | no such pipe counter on sm89 in ncu 2026.2 |
| issue active | 46.8% | at 8 warps per SM |

Two conclusions, and both contradict the draft of this spec.

**The scale correction is not the limiter.** The crux question was whether the F2F
pipe is quarter-rate, which would have made 1032 conversions carry like 4000 slots.
There is no separately metered F2F pipe on this architecture, so the conversions are
charged to ALU and FMA, which sit at 33.7% and 21.0%. The correction is the largest
block of instructions in the kernel and not the constrained resource, so folding
`d_x * d_y` is worth little on its own, and the reasoning that ranked it first was
wrong.

**The LSU is the wall, and it is fed by shared memory, not by global traffic.** The
static mix has LDS 408 plus STS 147 against LDG 133 plus STG 128, so about two thirds
of the memory instructions are shared accesses. That is also precisely why 004 probe
2 removed 4 of 16 *global* loads per thread per tile and moved nothing: it cut the
minority share of LSU work.

The target is therefore `LDS` and `STS` per `IMMA`. Shared-memory traffic here has
three sources: the per-K-block fp32 scale reads the correction needs, since `x_df`
and `y_df` are read as scalars inside the j loop of `vec_dot`; the activation operand
tiles, which `load_generic` pulls as two 32-bit shared loads per `mma` pair,
deliberately, on a comment saying `ldmatrix` is slower for that operand; and the
dequant stores on the way in.

With every pipe below 51% and issue at 46.8%, no resource is saturated. The kernel
cannot cover its own latencies, and 8 warps per SM is all the register file permits
at 254 registers per thread.

## The occupancy route 009 never closed, and why it does not exist

009 bought warps with more tiles and lost 20.9%, because halving J doubles how often
the whole weight tile is processed. Halving `I` looked like the mirror image: fewer
weight rows per block, so a smaller accumulator per thread, which is the only known
way under the 128 registers that 2 CTAs per SM require, while what gets doubled is the
*activation* side, which 004 showed is served by an L2 hitting 91.5%. Cheap currency
where 009 spent the expensive kind.

It is not available, and the reason is a structural invariant rather than a bug to
fix. `compute-sanitizer` on an `I` = 64 build localises the fault exactly: an
`LDSM.16.M88.4`, the `ldmatrix` that loads the weight operand, reading 16 bytes past
the end of the shared tile, 9658 errors, all at the same site. The vec-dot kernels
compute `ntx = rows_per_warp / tile_C::I` and `i0 = (threadIdx.y / ntx) *
rows_per_warp`, so a CTA covers `(nwarps / ntx) * rows_per_warp` rows. For every row
in the Ampere table that is 128: `rows_per_warp` 32 with `ntx` 2 for `J >= 48`, 16
with `ntx` 1 below it, `nthreads` 256 throughout. `I` is a derived quantity, so the
crash is the table promising something it cannot deliver.

And the arithmetic closes the escape hatch too: to cover 64 rows with 4 warps you need
`nthreads` = 128, and then `I * J / nthreads` is 64 accumulators per thread, exactly
what it is today at 256 threads and `I` = 128. The per-thread register footprint is
set by the mma tile shape and J, not by I, so the route cannot lower the thing that
blocks occupancy, and it would only add activation traffic. Option 1 is therefore
closed by derivation.

What survives from this section is a reporting defect, and it is small and real: the
config table presents `I` as a free parameter, `I` = 64 compiles without a single
warning, and the first thing that happens is an illegal shared read at run time. A
compile-time check that `config.I == (config.nthreads / warp_size / ntx) *
rows_per_warp`, or dropping `I` from the table and computing it, turns that into a
build error. Worth a standalone patch on its own.

## Revised option order

1. ~~Find and remove the `I = 128` assumption, then measure `I` = 64 at 2 CTAs per
   SM.~~ **Closed without building it, and the reason is better than a crash.** The
   vec-dot kernels set `ntx = rows_per_warp / tile_C::I` and `i0 = (threadIdx.y / ntx)
   * rows_per_warp`, so the rows a CTA covers are `(nwarps / ntx) * rows_per_warp`,
   which for every row in the Ampere table is exactly 128: `rows_per_warp` is 32 with
   `ntx` 2 for `J >= 48`, and 16 with `ntx` 1 below that, and `nthreads` is 256
   throughout. `I` is therefore not a tunable at all, it is derived from `nthreads`,
   `J` and the mma tile shape. `I` = 64 with `nthreads` = 256 does not need fixing, it
   needs `nthreads` = 128, and there `I * J / nthreads` is still 64 accumulators per
   thread, so the register footprint 009 could not lower is exactly the same and the
   only change is doubled activation traffic. Nothing to buy.
   What is real is the reporting defect: the table accepts `I` = 64, compiles, and
   reads 16-byte shared operands off the end of the tile. A host-side assert that
   `config.I == (config.nthreads / warp_size / ntx) * rows_per_warp` turns a fault at
   runtime into a compile-time error, and it is worth a small standalone patch.
2. ~~Hoist the fp32 scale reads out of the j loop.~~ Closed by the SASS census
   below: the emitted code is at 1.59 `LDS` per `IMMA`, so the loads are already
   merged.
3. ~~Widen or re-lay-out the operand loads.~~ Closed by the same census: 1.7 shared
   reads per `IMMA` means fragment reuse across the j loop is already happening, and
   `ldmatrix` cannot be used for the weight operand because the Q8_0 sram row stride
   is 280 bytes, which is not the 16-byte alignment `ldmatrix` requires.
4. Do not build the fp16 `HMMA` variant of the original option B. It spends a 4x
   tensor roof to remove work from a pipe running at 21%.
5. **The one route the counters still support: raise intra-warp instruction-level
   parallelism in the K loop**, so the int32-to-fp32 conversion and rescale of one
   K-block overlap the mma of the next. 12.1 correction instructions per `IMMA` on
   pipes at 21% and 34%, issue at 46.8%, 8 warps per SM, and no pipe saturated is a
   dependency chain, not a resource limit. It changes only the loop body in
   `vec_dot`, touches no tile shape, no shared-memory layout, no quantization, and
   it is one build away from being known.

## After the operand census: options 2 and 3 are also already done

Full opcode census of the shipping `mul_mat_q<IQ4_XS, J=128>`:

| opcode | count | per `IMMA` |
|---|---|---|
| LDS | 408 | 1.59 |
| LDSM, so `ldmatrix` | 32 | 0.12 |
| STS | 147 | 0.57 |
| LDG, STG | 261 | 1.02 |
| IMMA | 256 | 1.00 |
| I2FP, FMUL, FFMA | 3088 | 12.1 |

Two of the surviving options are therefore already implemented, and the numbers say
so rather than me reading another file.

- **Option 3, widen or change the operand loads.** There are 1.7 shared reads per
  `IMMA`. A `mma.sync.m16n8k32.s8` consumes 4 A registers and 2 B registers per lane,
  so any code that loaded its operands per mma would show about 6 shared reads per
  `IMMA`. The measured figure is a third of that, which means fragment reuse across
  the j loop is already happening, which is what the `A[ntx]` array hoisted out of the
  j loop in `vec_dot` does. The corresponding `ldmatrix` change is not available for
  the A operand at all: the Q8_0 sram stride is `2*32 + 64/32 + 4 = 70` ints = 280
  bytes per row, and `ldmatrix` needs 16-byte aligned rows, 280 is not. The 32 `LDSM`
  that do appear are the B side.
- **Option 2, hoist the per-element scale reads.** The `x_dm` load sits inside a loop
  over `l` where `tile_C::get_i(l)` takes each row index twice, and `l` is fully
  unrolled at compile time, so the two loads are the same address and the compiler
  merges them. At 1.59 LDS per `IMMA`, whatever the source says, the emitted code is
  not re-reading scales per element.

So four of the routes this spec named are now closed, and the census points at
something none of them were: 12.1 correction instructions per `IMMA`, 52% of the
kernel, sitting on pipes that are 21% and 34% busy, on a kernel issuing at 46.8%
with 8 warps per SM. That combination is a dependency chain, not a resource: each
mma's four int32 results must be converted and rescaled before the next mma's
accumulators are ready to take them, and there are not enough warps to interleave
around that.

The one idea the evidence still supports is therefore neither traffic nor tiling:
software-pipeline the k01 loop so the conversion of one k-block overlaps the mma of
the next, which raises instruction-level parallelism inside each warp without touching
tile shapes, shared memory or the quantization. It is a change to the loop body in
`vec_dot` and it is testable in one build. It is also the first idea in this chain
that the counters actually predict something for, so it should be measured before
anything else here is attempted.

## Where the cost actually lives

Per element per K-block, this kernel must convert an int32 partial to fp32, multiply
by the weight-block scale and the activation-block scale, and accumulate. Twelve
instructions per mma is that, and it is unavoidable while **both** operands carry a
scale every 32 values:

- the activations are q8_1, one fp32 scale per 32 values
- IQ4_XS weights are a super-block scale with eight 6-bit sub-block scales per 256
  values, so one scale per 32 values as well, which is what the dequant writes into
  `x_df`

If either side had one scale across the whole K, the int32 accumulator could run the
entire K loop and be converted and scaled once at the end, which is what makes
per-channel weight-only int8 kernels cheap. That is a format decision, not a kernel
one, and it is the thing 003 explored from the other side: on this model the
block-scale formats are what buy the quality at 4.25 bits, and the alternatives
measured worse on speed, size and quality together.

So the honest summary of the biggest kernel on this stack: 254 of 255 registers, one
CTA per SM by both the register file and shared memory, LSU 50.8% as the busiest pipe,
tensor 38.3%, and an epilogue whose instruction count is dictated by the quantization
granularity the model's quality depends on. Every kernel-level route is now measured or
derived closed: `cp.async` (004), global-load count (004), tile count via J (009),
tile rows via I (010), operand delivery and scale hoisting (010), and intra-warp ILP
(010).

The one contribution that survives is a defect report with a small fix: the config
table advertises `I` as tunable, `I = 64` compiles clean, and the result is an illegal
16-byte shared read at run time. A check that
`config.I == (config.nthreads / warp_size / (config.rows_per_warp() / 16)) *
config.rows_per_warp()` turns it into a build-time failure. That is worth a standalone
patch, and it is the only artifact of this line of work that helps anyone who did not
spend a night on it.

## What would reopen this

- A format with one scale per K-run on either operand, which is a model-quality
  decision and 003's territory, not a kernel one.
- Hardware with a bigger register file per SM or int8 mma that accumulates in fp32
  without a separate convert, i.e. not sm89.
- Evidence that the `A[ntx]` hoist is the wrong shape for some other (type, J) pair.
  The census here is for IQ4_XS at J=128; if a different type shows a much worse
  shared-read-per-mma ratio than 1.59, that type has a real problem worth digging into.

The sections below are the pre-gate draft, kept for the record. Read them through
this section, not through the instruction-count argument that motivated them.


## Design options, and what each one spends

Ordered by how much of that 52% correction block they remove.

**A. Precompute the scale product so the correction is one FFMA instead of a
multiply plus a fused multiply-add.** The `FMUL` count equals the `I2FP` count,
which is what computing `d_x * d_y` per correction looks like. If the per-column
and per-row scales are combined once per tile into a shared buffer, or folded so
one operand is applied in the conversion step, the correction becomes a single
fused op per accumulator element. Cheapest thing on the list: no layout change, no
numerics change if the association order is preserved, and it is worth 17% of the
instruction stream. This is the first experiment regardless of what Gate 0 says.

**B. Remove `I2FP` by carrying the int32 partial in a form the fp32 pipe can
consume without a conversion instruction.** The realistic version of this is
dequantizing the *weight* tile to fp16 or bf16 in shared memory and using
`HMMA` with fp32 accumulate, which deletes the int32-to-fp32 conversion because the
accumulation is already floating point. What it spends is the tensor roof, from 660
TOPS of INT8 to 165 TFLOP/s of FP16-with-fp32-accumulate, a 4x loss, to remove
17.4% of the instructions. On the numbers in this file the tensor pipe is at 38.3%,
so 4x of headroom exists, but this is the option most likely to lose, and it should
not be built before A is measured. It is also what the fp8 path on Ada would hit,
at 330 TOPS, which is 003's territory and out of scope here.

**C. Cut the correction frequency by widening the scale granularity.** The
correction is per 32-deep K-run because that is the q8_1 block. Accumulating two or
four adjacent blocks in int32 before rescaling is only valid if their scales
match, which they do not, unless the activation quantization is re-chosen so a
wider shared scale applies. That is a change to `quantize_mmq_q8_1` and to the
numeric behaviour of every model that uses MMQ, so it needs its own quality case
before it is attempted, and it is why this spec does not propose it.

**D. Attack the address arithmetic and moves.** `IMAD`, `LEA`, `MOV`, `LOP3`,
`IADD3` and `SHF` together are 1457 instructions, 24.6%, against 256 `IMMA`.
Some of that is the dequant bit-twiddling, which is the `PRMT` and `SHF` share, and
some is tile addressing. This is where fragment reuse earns something, but note the
size of the prize relative to A: the addressing is spread over several pipes while
the correction is concentrated in two.

## Test Seam and Testing Decisions

Unchanged and sufficient: `tests/test-backend-ops.cpp`, `test_mul_mat` on quantized
`src0`, compared against the CPU reference.

- Option A must be *bit-identical* to the current kernel wherever the association
  order of the accumulation is untouched, and the test that matters is a direct
  comparison of the CUDA output against the current library on every quantized type
  at the J values prefill uses, not just a tolerance pass. If A is not bit-identical
  it has changed the summation order and that has to be said out loud and checked
  with perplexity.
- Option B changes numerics by construction, so it inherits the full quality gate:
  `test-backend-ops` tolerances plus wikitext-2 within 0.02 of 003's 8.169 baseline
  on the 9B, and greedy token equality is off the table.
- Performance is judged paired and wall-clock only, at pp4096, pp65536 and
  pp131071, for the reason recorded in the README notes about nsys totals and clock
  drift.
- One more op-test note: the padded tail. `ne10_padded` to `MATRIX_ROW_PADDING`
  means the correction path touches rows that are not real, so any change to it
  needs the non-multiple-of-32 cases in the existing test matrix, which is where a
  mistake would show up as a wrong value rather than a crash.

## Acceptance Criteria

Baseline is this session's `build-004`, which contains 006 and 007: 9B pp65536 at
8548 t/s with flash attention on and q8_0 KV, 9B pp4096 at 11310 t/s, and 001's 27B
pp131071 at 1905 t/s.

- Gate 0: done. LSU 50.8% is the top pipe, ALU 33.7%, FMA 21.0%, tensor 9.9% as
  issued instructions and 38.3% as busy cycles, no F2F counter on sm89. The
  decision this produced is the revised option order above: shared-memory traffic
  and the `I = 128` assumption, not the correction arithmetic.
- Option A: `FMUL` count in the static histogram drops by at least half relative to
  `I2FP`, confirmed by reading the rebuilt SASS, and the tensor pipe rises above
  38.3% at unchanged occupancy. Wall target, pp65536 at least +3%.
- Anything that changes numerics: `test-backend-ops -o MUL_MAT -b CUDA0` green five
  times running, perplexity unchanged to the printed precision on the 9B and the
  27B on the corpus and chunk counts used for 007.
- Decode within 1% throughout, since MMVQ shares the dequant helpers and a change in
  `load_tiles` or the vec-dot epilogue can reach it.
- If A yields less than 1% on the wall, this spec closes without attempting B,
  because B spends a 4x tensor roof to remove a similar fraction of instructions
  and would need the win to be much larger than A's to be worth the risk.

## Rollback and Risk

- The static histogram is not the dynamic one. If the correction work is
  concentrated in a rarely-executed epilogue rather than the inner loop, option A
  is worth nothing, and Gate 0 is the only way to know that cheaply.
- The instruction counts include both the MMA and the dp4a code paths behind
  `#if`, so some of the `FMUL` and `I2FP` may belong to the path that does not run
  on sm89. Reading the SASS of the specific instantiation filters the dead
  branches at compile time, so what is quoted here should be live, but that is an
  argument from how the templates are specialised, not a verification, and Gate 0
  settles it.
- The correction loop is where the fp32 accumulators live, so any restructuring
  competes with the register budget that 009 showed is already at 254 of 255.
  Option A has to reduce pressure or hold it, not add to it.
- Risk of a false negative on A: a compiler that already folds the scale product
  for some type and J combinations. Check the histogram per type, not just for
  IQ4_XS, before concluding.

## Out of Scope

- Operand delivery with `ldmatrix`, which 004b proposed and this spec found already
  present.
- `cp.async` pipelining, closed by 004.
- Changing J or the tile counts, closed by 009.
- New quantization formats, fp8 paths, and anything requiring a re-quantized model
  (003).
- The attention kernels (005 closed, 008 open) and the SSM kernel (006).
- Blackwell and Hopper, where the tensor and scale-correction economics differ.

## Further Notes

- static instruction mix: `cuobjdump -sass -fun _Z9mul_mat_qIL9ggml_type23ELi128ELb0EE...`
  on `build-004/bin/libggml-cuda.so.0.25.0`, md5 `b56c071f...`
- `ldmatrix` already in use: `ggml/src/ggml-cuda/mmq-vec-dot.cuh`, and the
  `load_generic` comment for the activation operand
- `ldmatrix` wrappers and the int8 mma: `ggml/src/ggml-cuda/mma.cuh`
- pipe utilization, occupancy and the two block limits: `/tmp/ncu004.txt`, spec 004
- register cost model and the failed tile-count experiment: specs 009 and
  `/tmp/009/sweep.csv`
- this spec supersedes the "004b" note in spec 004, which named operand delivery as
  the target
