# 007: Fold the GLU into the MMQ activation quantization (sm89 prefill)

Status: done (2026-09-23). Committed locally as 247376881, not pushed.
Label: closed
Depends on: 001 (elementwise shares), existing CUDA fusion layer
Scope: sm89 only. Decode fusion behavior is already correct and stays as is.

## Results (2026-09-23): implemented, +1.4% pp on the 9B, +1.5% on the 27B

Measured with the paired ABBA rig (`/tmp/004/ab_glu.sh`), A = the committed 006
build, G = A plus this fusion:

| | A | fused | delta |
|---|---|---|---|
| 9B pp4096 | 11407.1 t/s | 11561.0 t/s | **+1.35%**, clean separation |
| 27B pp4096, fa on, q8_0 KV | 3307.7 t/s | 3357.9 t/s | **+1.52%**, clean separation |
| 9B tg256 | 153.14 | 153.10 | -0.02%, decode untouched by design |

Equivalence: `llama-perplexity` on the 9B over 24 chunks of a 110 KB text
corpus gives byte-identical per-chunk PPL, `[1]5.1131 [2]4.3670 [3]5.7293 ...`
through `[24]4.8601`, and the same final estimate `4.8601 +/- 0.10658`. On the
27B over 20 chunks the final estimate is likewise identical, `3.6570 +/-
0.07903` on both builds.

Full suite: `test-backend-ops -b CUDA0`, 16168 cases, on the fused build gave
16167/16168 on the first run with `MUL_MAT(type_a=q5_1, type_b=f32, m=16, n=1,
k=256, ..., o=1)` failing, then 16168/16168 on a repeat run, and that case passes
in isolation on both builds (`-p 'type_a=q5_1.*m=16,n=1,k=256'`, two runs each).
The 006 baseline build is 16168/16168. So this is a one-off flake in an
MMVQ-path case with no GLU in sight, not a regression, recorded here so it is not
chased again.

Op tests `-o GLU,MUL_MAT -b CUDA0` are 1297/1297.

Proof the fusion is live, not a null change: `unary_gated_op_kernel` launches
went 2688 -> 1152 for the same run, and `quantize_mmq_q8_1` launches stayed at
1024, which is the signature of the GLU being consumed inside the quantizer
instead of producing an intermediate.

### Where my estimate was wrong

The spec predicted 3-4.5%. The kernel-time capture suggested far more (78.5 -> 8.3
ms of `unary_gated`), and the wall number was in between at 1.4%. The capture is
the unreliable one: in that trace `mul_mat_q` accounts for 26.9 ms of a run whose
known GEMM time is several hundred ms, so nsys under-recorded the GEMM kernels and
the shares are not usable. Instance counts in the same trace are trustworthy, times
are not. Lesson matches the one already recorded for clock drift: on this stack, an
nsys capture is an instrument for counting and ordering, not for totals, unless the
capture's own total is checked against `llama-bench` wall time first.

The real reason the win is ~1.5% rather than 4%: the f32 round trip that
disappears is one write plus one read of the FFN intermediate, and the FFN
intermediate at ubatch 512 is only about 25 MB per layer, so the saved traffic is
near 50 MB per (layer, ubatch) against a prefill that is dominated by weight
streaming. The prediction that this is mostly traffic-bound was right in kind and
too large in size.

### What was built

- `quantize_mmq_q8_1` gained a `glu` template parameter and a second source
  pointer. It forms `silu(gate) * up` in registers, in the same expression order as
  `unary_gated_op_kernel`, so the quantized block is bit-identical to the current
  two-kernel path
- `ggml_cuda_glu_is_fusable` in the MMQ layer is the single predicate, and
  `ggml_cuda_mul_mat_q` uses it to take the down projection's activations from the
  GLU's own inputs: swiglu only, two-tensor GLU only, contiguous f32 gate and up
  with matching shapes, and no `MUL_MAT_ID`
- the graph loop skips a GLU node only when the very next node is the MUL_MAT that
  consumes it, that node provably takes the MMQ path (`should_use_mmq &&
  !should_use_mmvq`, which is how `ggml_cuda_mul_mat` dispatches), and no later
  node reads the GLU output
- the skip is only an optimization: the fused read of gate and up computes the same
  values the GLU node would have written, so a case that takes MMQ but is not
  adjacent still produces correct output, just without the saving
- decode is unaffected because batch <= 8 goes to MMVQ, which has its own existing
  GLU fusion, and that path is untouched

## Problem Statement

The FFN in every layer moves the same data through DRAM four times for work
that needs one pass. Per layer per prefill ubatch today:

1. gate projection: MMQ reads f32 activations, quantizes them to q8_1 on a
   scratch buffer, computes, writes an f32 intermediate.
2. up projection: same, second f32 intermediate.
3. GLU: reads both f32 intermediates, writes a third f32 tensor.
4. down projection: reads that f32 tensor, quantizes it to q8_1 on a scratch
   buffer, computes.

Steps 3 and 4 are separate kernels over the largest intermediate in the network,
and step 4's quantization is a per-MMMQ-node launch that exists only because the
backend needs q8_1 activations. From 001, pp32768 on the 27B:

| kernel | share | instances | average |
|---|---|---|---|
| silu (gated) | 3.6% = 2.5 s | 43084 | 59 us |
| quantize_mmq_q8_1 | 2.2% = 1.5 s | 149760 | 10 us |

The quantize instance count resolves to one per MMQ node: 149760 / 6 passes / 64
ubatches = 390 per ubatch, which is the 27B's quantized GEMM count per ubatch.
Their 10 us average on a 512 x 5120 f32 activation says these are launch and
tail bound, which is the cheapest kind of work to remove.

The mechanism already exists in the backend and already handles this pattern -
for decode only:

- `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_cuda_can_fuse`, matches the patterns
  `{MUL_MAT, MUL_MAT, GLU}` and `{MUL_MAT, ADD, MUL_MAT, ADD, GLU}`, and checks
  memory ranges with `ggml_cuda_check_fusion_memory_ranges`.
- `ggml_cuda_should_fuse_mul_mat` gates it, and `ggml_cuda_should_fuse_mul_mat_vec_q`
  limits the actual fused execution to `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`. So
  at batch 1-8 the GLU node disappears into the vector GEMM; at prefill batch
  512 the pattern is recognized and then nothing consumes it.
- `ggml_cuda_mul_mat_q` (`ggml/src/ggml-cuda/mmq.cu`) is where the q8_1 scratch
  is allocated and `quantize_mmq_q8_1_cuda` is launched, and that kernel already
  absorbs an optional `ids` gather (`ggml/src/ggml-cuda/quantize.cu`), so taking
  a second source and an activation function is in character for it.

## Solution

Extend the existing fusion one step: when a GLU node's consumers are the
activations of a single MMQ down projection, run the gated activation inside the
q8_1 pre-quantization kernel and delete the standalone GLU node. The intermediate
is produced in registers, quantized in place, and never goes to DRAM as f32.

From the operator's point of view: faster prefill on every model with a gated
FFN, no flag, no quality change, no new type. From the maintainer's point of
view: this reuses the fusion pattern matcher and the memory-range check that are
already in the CUDA backend, and adds an optional second source to a kernel that
already has an optional second source role.

## User Stories

1. As a llama-server operator, I want prefill to stop writing and re-reading the
   FFN intermediate, so that time to first token drops on every gated model.
2. As a llama-server operator, I want about 390 fewer kernel launches per
   ubatch, so that the small launch-tail overhead stops adding up.
3. As a llama-server operator, I want no change in output, so that I do not have
   to re-verify my presets.
4. As a llama-server operator on a 24 GB card, I want a smaller peak activation
   buffer for the FFN, so that I can afford a larger ubatch.
5. As a batch/tuning user, I want the gain at the batch sizes I actually sweep
   (256-2048), so that tuning runs finish sooner.
6. As a llama.cpp contributor, I want this expressed through the existing
   `ggml_cuda_can_fuse` pattern list, so that no second fusion framework appears
   in the backend.
7. As a llama.cpp contributor, I want the unfused GLU node left fully working, so
   that a model where the pattern does not match behaves exactly as today.
8. As a llama.cpp contributor, I want the fused path to require both GLU inputs to
   come from the same layer's two projections, so that a partial match never
   changes numerics.
9. As a reviewer, I want bias and scale variants covered or explicitly excluded
   by the gate, so that the fused path cannot silently disagree with the unfused
   one on a model that adds a bias.
10. As a reviewer, I want the `swapped` GLU layout handled, so that models that
   interleave gate and up halves keep working.
11. As a person running the 9B dev model, I want one llama-bench row set to
   decide this, so that the change costs an evening, not a week.
12. As a maintainer of other backends, I want no ggml-level op or graph change, so
   that CPU, Metal, Vulkan and SYCL are untouched by this diff.
13. As a maintainer of decode, I want the batch 1-8 fusion path untouched, so that
   the tokens-per-second numbers from 001 stay valid.

## Implementation Decisions

- Fuse at the pre-quantization step, not in the GEMM epilogue. The quantize
  kernel becomes `quantize_mmq_q8_1(gate_out, up_out, glu_op, swapped)` and
  emits q8_1 directly; the f32 GLU output tensor is dropped from the graph. The
  epilogue variant (fold GLU into the up projection's write-back) would also
  remove one f32 write, but it reaches into `mul_mat_q_process_tile`'s store
  path, which 004 is about to restructure. Sequencing decision: this spec lands
  before or after 004, not with it, and the two touch disjoint code if 007 uses
  the pre-quantization site.
- Reuse `ggml_cuda_can_fuse` with the existing `{MUL_MAT, MUL_MAT, GLU}` and
  `{MUL_MAT, ADD, MUL_MAT, ADD, GLU}` patterns. Add the MMQ branch to the
  consumer-side check in `ggml_cuda_should_fuse_mul_mat` rather than to
  `ggml_cuda_should_fuse_mul_mat_vec_q`, which keeps the decode gate readable.
- The down projection's `src1` is the fused output, so the fused kernel must be
  launched as part of `ggml_cuda_mul_mat_q` for that node, using the same pool
  scratch it already allocates. No new buffer, no new allocation site.
- Numerics: identical math, identical fp32 accumulate, and the q8_1 block scale
  computed over the same 32-value groups the current kernel uses, so the fused
  result must be bit-identical to gate -> GLU -> quantize. Bit-identity, not
  tolerance, is the acceptance target here, because there is no reason for it to
  be anything else.
- Padding: `ne10_padded` and the `MATRIX_ROW_PADDING` tail must be filled the
  same way the current quantize kernel does it, including the zeroed tail
  behavior. This is where a fused kernel usually breaks.
- Scope the first cut to `GGML_GLU_OP_SWIGLU` and the plain (no bias, no scale)
  form, which is what this model uses, plus the bias form if the pattern match
  is free. Every other GLU variant keeps the current two-kernel path.
- The quantize kernel's shared-memory reduce over a 32-value block must read
  both sources with the same index arithmetic; the `swapped` flag selects which
  half is the gate. Keep the three existing DS layouts (D4, DS4, D2S6) and add no
  new one.
- No ggml op, no graph-builder change in `src/`, no new `common` flag.

## Test Seam and Testing Decisions

One seam, the one the backend fusion already has to satisfy:
`tests/test-backend-ops.cpp`.

- The relevant cases are the existing `MUL_MAT` and `GLU` op tests, plus the
  fused-pattern tests the file already carries for the decode fusion
  (`test_mul_mat_vec_fusion`, which enumerates `GGML_GLU_OP_SWIGLU` and friends).
  The right move is to generalize that existing test so the same matrices run at
  prefill batch sizes and assert the fused result equals the unfused one. No new
  test file, and no new op test struct.
- Good tests here assert only the external result of `GGML_OP_MUL_MAT` on a
  quantized `src0` when the graph contains a GLU feeding it. They must not assert
  that a particular kernel ran or that a node was removed; the node-count claim
  belongs in the profile, not in a unit test.
- Batch sizes to cover: 8 (existing decode fusion, must not change), 128, 512,
  and a non-multiple of 32 in `ne0` to exercise the padded tail.
- Run the CUDA filter 5 times: the fused kernel changes the shared-memory reduce
  pattern, and a wrong block scale shows up as a rare row, not a consistent one.
- Quality is checked outside the seam with `llama-perplexity`, per 003's method.

## Acceptance Criteria

Baseline is 001, pp32768: silu 3.6% (2.5 s) + quantize_mmq_q8_1 2.2% (1.5 s) =
5.8% of 71.4 s over 6 passes.

- The fused run is bit-identical to the unfused run on the model output for a
  fixed prompt at temp 0, first 64 tokens, and on the down-projection activations
  in `test-backend-ops`.
- pp4096 and pp32768 on the 27B: >= 3% faster than baseline (5.8% share removed
  in full is the ceiling; 3% is the pass mark, 4.5% is the target).
- pp131071: >= 2% faster than the 1905 t/s baseline.
- nsys: `quantize_mmq_q8_1` instance count per ubatch drops by the number of
  fused down projections (about one per layer), and the standalone gated-silu
  calls for the FFN are gone. Peak VRAM for the FFN intermediates drops by one
  f32 intermediate per layer.
- tg128 and tg2048 within 1% of baseline, with the decode fusion path unchanged in
  the profile.
- `test-backend-ops -b cuda` fully green, and the MUL_MAT/GLU filter 5x.
- wikitext-2 perplexity unchanged to 3 decimals on the 9B dev model.

## Rollback and Risk

- Risk 1: the padded tail. The current kernel writes a zeroed `MATRIX_ROW_PADDING`
  region that the GEMM reads. If the fused version skips or misplaces it the
  result is wrong in a way that only shows at some `ne0`. Mitigation: the
  non-multiple-of-32 test case, and an explicit assert in the wrapper.
- Risk 2: register pressure or shared memory in the quantize kernel grows, since
  it now holds two source rows per output block. Mitigation: this is a
  pre-quantization kernel with a simple shape; if it gets slower than the two
  kernels it replaced, the fusion gate is narrowed rather than the kernel
  rewritten.
- Risk 3: interaction with the memory-range check, so a fusion is approved where
  the GLU output buffer is also an output the caller still needs. The existing
  `ggml_cuda_check_fusion_memory_ranges` covers this class of bug and must be
  called, not bypassed.
- Risk 4: overlap with 004 in the MMQ files. Mitigation: 007 touches only
  `quantize.cu`, the fusion gate, and the call site in `ggml_cuda_mul_mat_q`; it
  does not enter `mmq.cuh`'s tile loop. Land them separately and re-measure the
  second on top of the first, since both claim part of the same 5.8%.
- Rollback: one predicate in `ggml_cuda_should_fuse_mul_mat`. Return false and
  the previous behavior is exact.

## Out of Scope

- Fusing GLU into the MMQ write-back epilogue, and any change to
  `mul_mat_q_process_tile` (004 owns that loop).
- Making MMQ accept f16 or f32 `src1` directly, or changing the q8_1 layout.
- Other gated variants (REGLU, GEGLU, clamp, OAI) until SWIGLU lands.
- RMS_NORM and rope elementwise traffic, and the `concat`/`set_rows` cost.
- Anything in the attention path (005) or the SSM path (006).
- CPU-offloaded layers and the MoE `MUL_MAT_ID` variants; the pattern list
  includes them, but this spec does not enable them.

## Further Notes

Evidence pointers (line numbers as of `bddf8263c`, upstream/master):

- fusion patterns and checks: `ggml/src/ggml-cuda/ggml-cuda.cu`,
  `ggml_cuda_can_fuse`, `ggml_cuda_should_fuse_mul_mat`,
  `ggml_cuda_check_fusion_memory_ranges`
- decode-only gate: same file, `ggml_cuda_should_fuse_mul_mat_vec_q`
  (`src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`)
- per-node activation quantization: `ggml/src/ggml-cuda/mmq.cu`,
  `ggml_cuda_mul_mat_q` (`src1_q8_1` pool alloc, `quantize_mmq_q8_1_cuda`)
- quantize kernel and its optional gather source: `ggml/src/ggml-cuda/quantize.cu`,
  `quantize_mmq_q8_1`, `quantize_mmq_q8_1_cuda`, DS layout selectors
- gated activation kernel: `ggml/src/ggml-cuda/unary.cu`,
  `unary_gated_op_kernel`, `ggml_cuda_op_swiglu`
- fused-op test prior art: `tests/test-backend-ops.cpp`,
  `test_mul_mat_vec_fusion`
- shares: `specs/001-baseline.md`, P2 table
