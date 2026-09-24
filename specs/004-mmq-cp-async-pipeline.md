# 004: MMQ weight-tile prefetch with cp.async (sm89 prefill GEMM)

Status: done (2026-09-23, verdict: no change; the pipelining premise is falsified
by measurement, the headroom is real but is not a load-latency problem)
Label: ready-for-agent -> closed
Depends on: 001 (baseline numbers, all shares below come from it)
Scope: sm89 only. The async path lives behind the Ampere/Ada template branch.

## Problem Statement

Prompt processing is dominated by quantized GEMM, and that GEMM runs far below
both hardware roofs. From 001, pp32768 on the 27B IQ4_XS (6 passes, 71.4 s of
GPU kernel time):

| kernel | share of pp32768 |
|---|---|
| mul_mat_q IQ4_XS | 28.9% |
| mul_mat_q IQ3_S | 9.2% |
| mul_mat_q Q4_K | 5.6% |
| mul_mat_q Q5_K | 4.7% |
| mul_mat_q IQ3_XXS | 3.3% |
| mul_mat_q IQ2_S | 1.8% |
| quantize_mmq_q8_1 | 2.2% |
| all mul_mat_q + quantize (GEMM) | ~57% |

Derived roof position for that 57% (001 weights = 13.3 GB of quantized 2D
weights, 32768 tokens x ~50 GFLOP/token, ubatch 512 = 64 weight passes):

- tensor-core work: ~245 TFLOP/s effective = 37% of the 660 TOPS INT8 peak
  (fp32 accum, the only rate GGML can use on Ada)
- weight traffic: ~124 GB/s = 12% of the 1008 GB/s DRAM peak

A kernel that sits at 37% compute and 12% memory is stalled, not limited. The
stall is visible in the source: `mul_mat_q_process_tile` (`ggml/src/ggml-cuda/
mmq.cuh`, the `for (int kb0 = kb0_start; ...)` loop) issues the weight-tile
loads, waits on `__syncthreads()`, runs `vec_dot`, waits again, loads the second
half of the activation tile, waits, runs `vec_dot` again. Every DRAM round trip
is paid in full, with the tensor cores idle. The loaders
(`ggml/src/ggml-cuda/mmq-load-tiles.cuh`, e.g. `ggml_cuda_mmq_load_tiles_q8_0`)
read block bytes with plain global loads into registers, then store to shared
memory. There is no `cp_async` anywhere in `mmq.cuh`, `mmq-load-tiles.cuh`, or
`mmq-vec-dot.cuh`, even though `cp-async.cuh` exists in the same directory and
the flash-attention MMA kernel already pipelines with it.

The launch config removes the last escape route: the Ada/Ampere MMQ table gives
every hot type `nthreads=256, occupancy=1` (`mmq-config-ampere.cuh`, the
IQ4_XS / IQ3_S / Q4_K / Q5_K CASE rows). One 256-thread block per SM is 8 warps
out of 48 warp slots, so there is no other warp around to cover the memory
latency. Low occupancy is only affordable when loads and math overlap.

## Solution

Software-pipeline the MMQ K-loop the way every modern quantized GEMM does it:
copy the raw (still quantized) weight bytes for tile n+1 into a second shared
buffer with `cp.async` while the tensor cores consume tile n. The dequantize
step stays where it is, but it stops sitting in front of a blocking DRAM wait.

From the operator's point of view nothing changes: same model files, same
quantized types, same numerical result within the existing tolerance, and a
shorter time to first token on every prompt. From the maintainer's point of view
the change is confined to the MMQ tile loop and the type-specific loaders, and
gated by the existing arch macros so no other backend or arch moves.

## User Stories

1. As a llama-server operator, I want prefill to use the tensor cores I already
   paid for, so that long prompts stop costing 3x their theoretical time.
2. As a llama-server operator, I want lower time to first token at ubatch 512,
   so that interactive sessions feel the gain without me changing any flag.
3. As a llama-server operator, I want the gain to be free of quality change,
   so that I do not have to re-run evals after a backend update.
4. As a llama-server operator, I want the change to be a runtime no-op, so that
   no new CLI flag or env var enters my service unit.
5. As a batch user, I want the same speedup on llama-cli and llama-bench, so
   that the improvement is visible outside the server.
6. As a llama.cpp contributor, I want the async path behind an arch macro, so
   that Volta, Turing, Pascal, and AMD builds compile the code they run today.
7. As a llama.cpp contributor, I want the number of shared-memory stages to be
   a config-table value, so that a later arch retune is a data change and not a
   kernel change.
8. As a llama.cpp contributor, I want per-type loaders left in one place, so
   that a type that cannot be pipelined (huge block, or scale-heavy layout)
   falls back to the synchronous loader without a special case in the loop.
9. As a reviewer, I want the fp32 accumulation order unchanged, so that outputs
   stay bit-comparable to the current kernel where the tile shape does not move.
10. As a reviewer, I want the shared-memory budget stated per instantiation, so
    that I can confirm no kernel silently drops below its occupancy target on a
    100 KB Ada SM.
11. As a person running the 9B dev model, I want a kernel A/B that resolves in
    one llama-bench matrix, so that the change can be judged in an evening.
12. As a maintainer of decode performance, I want MMVQ left alone, so that batch
    1-8 generation, which is bandwidth bound, is not put at risk by a GEMM
    pipeline change.

## Implementation Decisions

- Target the tile loop in `mul_mat_q_process_tile` only. The stream-K fixup
  kernel and the `fallback` variants keep the current body; the pipeline goes in
  the non-fallback path, which is what the Ada table selects at ubatch 512.
- Two stages. `nstages=2` mirrors what the fattn MMA kernel already does on this
  hardware and is the minimum that removes the round trip from the critical
  path. More stages are recorded as follow-up, not part of this spec, because
  the IQ4_XS tile at `I=128` leaves little room on a 100 KB SM.
- Copy raw quantized bytes with `cp.async.cg` 16 B, do not copy dequantized
  values. The existing layout in shared memory stays byte-identical, so
  `vec_dot` is untouched. This also keeps the fp32 accumulate order identical.
- Activation tiles (`tile_y`, q8_1) are already L2-resident and small; they are
  not part of the first stage of this work. Only the weight side is pipelined.
  If the weight side lands and profiles well, the activation side is a separate
  small follow-up.
- Per-type rollout order, cheapest first, each one gated by its own
  `test-backend-ops` run: Q8_0, Q4_K, Q5_K, IQ4_XS, IQ3_S. IQ2/IQ1 and the
  block-quant types with two-level scales stay synchronous until a measurement
  says otherwise.
- The wait primitive must be the stage-count-aware one. `cp_async_wait_all()`
  drains everything and would leave the loop serialized in a different way; the
  pipeline needs `cp.async.wait_group 1` semantics (wait for all but the most
  recent group).
- Shared memory: the second weight tile is added to the per-instantiation
  requirement, and the occupancy target in the config table is lowered to 1 if
  the extra buffer would exceed the Ada limit. State the resulting byte count in
  the PR body.
- No new ggml op, no new type, no dispatch change in
  `ggml_cuda_should_use_mmq`. This spec does not reopen 003.

## Test Seam and Testing Decisions

One seam, and it already exists: `tests/test-backend-ops.cpp`, at the ggml op
level (`GGML_OP_MUL_MAT` with a quantized `src0`), compared against the CPU
reference. This is the highest seam that can see the change, and it needs no new
harness.

- Good tests here assert external behavior only: the op result, within the
  per-type tolerance the file already uses. They must not reach into shared
  memory layout, stage counts, or kernel names.
- The relevant cases are the existing quantized `test_mul_mat` rows, so a real
  model shape must be run, not just small squares: at least the J values the Ada
  table uses for 512-token batches, and an `n_kb` (K dim) that produces more
  than 2 tile iterations, since a 1-iteration loop cannot show a pipeline bug.
  Add the long-K case as a parameter to the existing test, do not create a new
  test file.
- The classic double-buffer bug (read before the copy lands) shows up as a
  wrong value that depends on thread scheduling. Run the CUDA filter 5 times
  before calling it green.
- Performance is judged outside that seam, never inside it: llama-bench pp
  matrix plus one nsys pass, per 001's method section.

## Gate 0 Results (2026-09-23)

`ncu` is blocked on this machine: `ERR_NVGPUCTRPERM`, the user has no access to
the NVIDIA GPU performance counters. So Gate 0 was run with counters-free
measurements instead, all on the 9B dev model, CUDA-only build (`build-004`,
`GGML_VULKAN=OFF`), `-b 2048`, pp4096, r=4-5.

Methodology note that changed a conclusion: the first three-type run below used
`build/`, which has Vulkan ON, so llama-bench split layers across the 4090 and
the AMD iGPU in that same host. Those numbers are not pure-CUDA numbers and were
discarded. See the README workflow notes.

### Probe 1: is the GEMM limited by weight bytes?

| model (same 8.95 B params, same shapes) | weight bytes | pp4096 t/s | vs IQ4_XS |
|---|---|---|---|
| Qwen3.5-9B Q8_0 | 8.86 GiB | 9776.6 | -3.4% |
| Qwen3.5-9B IQ4_XS | 4.80 GiB | 10121.1 | baseline |
| Qwen3.5-9B MXFP4 | 5.44 GiB | 9815.0 | -3.0% |

1.85x the weight bytes costs 3.4% of the time. Weight traffic is not the
limiter. At ubatch 512 the whole prefill moves 8 x 4.80 GiB = 38.4 GiB in
0.412 s = 95 GB/s = 9% of the 1008 GB/s roof.

It also rules out the dequant ALU as the shared limiter: Q8_0's loader does no
unpack at all, IQ4_XS's does a 16-entry table lookup per 8 values, and they run
3.4% apart.

### Probe 2: is the loader's load-instruction count on the critical path?

E1 prototype, reverted: `ggml_cuda_mmq_load_tiles_iq4_xs` scale loop merged the
three 1/2-byte field loads per row (`d`, `scales_l[i]`, `scales_h`) into two
aligned 4-byte loads, since those fields are the first 8 bytes of
`block_iq4_xs`. Same index math, same values, 12 -> 8 global loads per thread
per tile iteration.

| | pp1024 | pp4096 | pp8192 |
|---|---|---|---|
| A baseline | 10113.1 | 10139.8 | 9972.3 |
| B patched | 9993.0 | 10081.4 | 9937.1 |
| delta | -1.19% | -0.58% | -0.35% |

No gain, mild regression (pp1024 is within its own spread). Load instruction
count is not what the tile loop waits on.

### Probe 3: does more math per load chain pay? (batch sweep, IQ4_XS)

| ubatch | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| pp4096 t/s | 5420 | 6759 | 8793 | 9959 | 10158 | 9879 |

Throughput rises 1.84x from batch 64 to 512, then goes flat, then falls. Weight
traffic per token keeps halving from 512 to 1024 (95 -> 50 GB/s) and the time
does not move at all. So above batch ~512 the kernel is not waiting on weights
at all; it sits on a roof that scales with work done, not with data fetched.

### What the three probes say together

The premise of this spec was: "37% of the INT8 tensor roof and 12% of the DRAM
roof means the kernel is stalled, and the stall is the un-overlapped load".

- "the kernel is stalled" - confirmed. It is far from both roofs.
- "the stall is the un-overlapped load" - falsified. Bytes do not matter
  (probe 1), load instruction count does not matter (probe 2), and hiding the
  weight fetch behind more math per fetch does not help past batch 512
  (probe 3).

The Nsight Compute results below identify the actual limiter, and they explain
why all three probes came out the way they did.

## Counter results (2026-09-23, route A: ncu run as root)

Target: the kernel that owns prefill. An nsys capture of the same run
(`pp4096.nsys-rep`, 3 reps of pp4096, ubatch 512) puts the GEMM family at 61.6%
of all kernel time, and within it:

| kernel | type | J | fallback | ms | share of GEMM family | avg |
|---|---|---|---|---|---|---|
| `mul_mat_q` | IQ4_XS | 128 | no | 872.0 | 88.1% | 162 us |
| `mul_mat_q` | Q5_K | 128 | no | 69.2 | 7.0% | 68 us |
| `mul_mat_q` | Q8_0 | 128 | **yes** | 18.2 | 1.8% | 12 us |
| `mul_mat_q_stream_k_fixup` | Q8_0 | 128 | yes | 19.1 | 1.9% | 12 us |

So `mul_mat_q<(ggml_type)23, 128, 0>` alone is **54.3% of prefill kernel time**
on this model, which matches 001's 57% GEMM share on the 27B. That is the kernel
ncu profiled. Numbers for it:

| metric | value | reading |
|---|---|---|
| Duration | 140.1 us | one launch |
| Compute (SM) throughput | 49.0% | neither roof is close |
| DRAM throughput | 23.4% | 229 GB/s effective |
| L2 hit rate | **91.5%** | weights mostly arrive from L2, not DRAM |
| L1/TEX hit rate | 19.7% | |
| highest-utilized pipe | **LSU 49.0%** | ncu: "appears to be caused by frequent, low-latency instructions" |
| tensor pipe (INT) | 38.3% | the tensor cores are *not* the wall |
| Registers per thread | **254** | at the hardware maximum |
| Dynamic shared memory per block | 57.86 KB | |
| Block Limit Registers / Shared Mem | 1 / 1 | both cap the CTA count |
| Theoretical = achieved occupancy | 16.67% = 16.65% | 8 warps of 48 warp slots |
| Active warps per scheduler | 2.00 (of 12) | |
| Issued warps per scheduler | 0.46 | an instruction every 2.2 cycles |
| Warp cycles per issued instruction | 4.31 | warps are *not* stalled long; there are just too few |
| ncu `Est. Local Speedup` from the scheduler rule | **50.96%** | what fixing issue/occupancy is worth on this kernel |
| FP32 fusion note | 16.8 M non-fused vs 16.8 M fused FP32 ops | ~5% extra available from FMA hygiene |

### The corrected diagnosis

MMQ on Ada is **LSU-and-issue bound at 1 CTA per SM**, not load-latency bound:

- L2 serves 91.5% of the weight traffic, so DRAM bandwidth and global-load
  latency are largely out of the picture. This is why probe 1 (1.85x bytes) and
  probe 2 (fewer global loads) moved nothing: probe 2 cut global loads, but the
  LSU pressure is in the shared-memory operand reads inside `vec_dot`, where each
  `mma.sync m16n8k32 s8` consumes 6 LDS.32 per lane.
- With 254 registers and 57.86 KB of shared memory per block, both the register
  limit and the shared-memory limit say 1 CTA per SM. Two warps per scheduler
  cannot hide 49%-busy LSU, so the schedulers idle 54% of cycles and the tensor
  pipe runs at 38%.
- The register count is the deliberate design: `sum[I*J/(nwarps*warp_size)]` =
  64 fp32 accumulators per thread at I=J=128, plus operand fragments held live
  for reuse across the tile. Fragment reuse was bought with occupancy.

### The occupancy lever is not a config-table edit

The probe I expected to be one line is not. Every one of the 351 rows in the
Ampere table and all 32 rows in the Blackwell table use
`nthreads=256, occupancy=1, I=128`. Setting the IQ4_XS J=128 row to
`occupancy=2, I=64` compiles cleanly (the `CASE` static_asserts allow it) and
crashes at run time with `CUDA error: an illegal memory access` in
`ggml_cuda_compute_forward: MUL_MAT`. So `I=128` is assumed by the index math
and the buffer sizing, even though the table presents `I` as a tunable. Two
consequences:

1. Raising occupancy requires re-deriving the tile indexing, not editing a row.
2. That hidden assumption is worth reporting upstream on its own, independent of
   any performance work: either the table should assert it, or the code should
   support the value it advertises.

### What 004b is now, precisely

The prize, from ncu rather than from inference: about 1.5x on 54% of prefill
kernel time, so roughly 18% off pp wall. The work, in the order the counters
imply:

1. Cut LSU work per tensor-core op. Deliver the int8 operands with `ldmatrix`
   instead of 6 separate `LDS.32` per lane per mma, and raise the number of mmas
   each loaded fragment feeds. This is a change to `mmq-vec-dot.cuh` plus the
   shared-memory swizzle in `mmq-load-tiles.cuh`, and it is the only change that
   addresses the top pipe.
2. Reduce or share the accumulator footprint so 2 CTAs per SM becomes reachable
   without spilling, which means the `I=128` assumption from step 1's layout has
   to be revisited as part of the same restructure.
3. Only after 1 and 2: consider async copies. With L2 at 91.5% hits they are the
   least of the three.

That is a mainloop rewrite for one arch family, which is why 004 stays closed.

### Measured and declined: the fallback path

`fallback = ne01 % 128 != 0` in `ggml_cuda_mul_mat_q` sends whole matrices down
a slower kernel. On this model those are the 64 Q8_0 tensors per ubatch, and they
cost 3.8% of the GEMM family plus 1.9% of matching `stream_k_fixup` work, about
2.4% of prefill, at 8.7% SM busy and dominated by instruction-fetch stalls
because the grid is one partial wave. Real but small, and it does not compete
with anything in 005-007. Noted so it does not get reopened.

## Verdict

What is left is the inner loop's instruction mix at the launch shape the Ada MMQ
table picks: 1 CTA of 256 threads per SM (8 of 48 warp slots, 2 warps per
scheduler), operands staged through shared memory with manual `LDS.32` indexing
rather than `ldmatrix`, one
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` surrounded by address math
and an int32 -> fp32 per-K-block scale correction. That is an issue-rate and
operand-delivery problem, which is what the counters name directly: LSU is the
top pipe at 49.0%, the tensor pipe is at 38.3%, and L2 already absorbs 91.5% of
the weight reads. `cp.async` does not touch any of those. Neither does a second
shared-memory buffer, and neither does a table edit to force 2 CTAs per SM.

Closing this spec as "no change". The headroom is real but is now sized properly:
ncu's scheduler rule puts ~1.5x on this one kernel, which is 54.3% of prefill
kernel time on this model, so roughly 18% off pp wall. Capturing it is a mainloop
rewrite for sm89 (see "What 004b is now, precisely"), not a quick win, and it is
out of scope for the 004-007 set. It stays logged as 004b if it is ever worth
opening.

## Acceptance Criteria as originally planned (superseded by the results above)

Gate 0, before any code, one command, decides whether this spec is real:

```sh
ncu --set full -k regex:mul_mat_q --launch-count 1 --launch-skip 40 \
    ./build/bin/llama-bench -m <27B IQ4_XS> -p 4096 -ngl 99 -ub 512 -ngd 0
```

- Proceed only if the warp stall breakdown is dominated by long-scoreboard
  (memory latency) with `sm__pipe_tensor` well under 50% active.
- If it is dominated by the math/ALU pipe, the dequant is the limiter, this spec
  is closed as "no change", and 003's conclusion stands with a mechanism
  attached.

Then, on the 27B with the 001 matrix (`tools/bench/server-bench`, service
stopped, per the 002 workflow notes):

- pp1024 / pp4096 / pp32768 / pp131071 improved by >= 5% each, or the spec
  records which batch size the gain stops at.
- pp32768 GEMM share (001 baseline: 57% of 71.4 s over 6 passes) drops by >= 8%.
- Effective GEMM throughput rises from 245 TFLOP/s toward 300 TFLOP/s.
- tg128 and tg2048 within 1% of baseline (no decode regression; MMVQ untouched).
- `test-backend-ops -o MUL_MAT -b cuda` green, 5 consecutive runs.
- wikitext-2 perplexity unchanged to 3 decimals on the 9B dev model.
- VRAM: peak allocation within 200 MB of baseline (occupancy may drop to 1,
  which is allowed and must be stated).

## Rollback and Risk

- Risk 1: extra shared memory drops occupancy below the config-table target for
  some type x J combination, which can make it slower rather than faster.
  Mitigation: stages are a per-table-entry value; ship only entries that win.
- Risk 2: a race that only appears under real load. Mitigation: 5 repeat runs of
  the op filter, and the nsys pass is done on a production-shaped prefill.
- Risk 3: 16 B cp.async needs 16 B aligned sources. Some quantized block rows
  are 4 or 8 byte aligned only (the `static_assert`s in `mmq.cuh` about the
  `+ 4` shared-memory stride exist for exactly this reason). Where the raw row
  cannot be copied in 16 B units, that type stays synchronous; do not force it.
- Rollback: per-table-entry opt-out. The kernel body keeps the synchronous
  branch, so rollback is a data change in the config table.

## Out of Scope

- Any new quantized type, FP8/FP4 path, or tensor-core data type (003 closes
  that for Ada).
- MMVQ (decode batch 1-8), the cuBLAS path, MoE `mul_mat_id`, and all
  arches other than sm80/sm89.
- Changing the dequantize math or the shared-memory tile layout.
- Fusing the GLU epilogue into the activation quantization: see 007.
- Direct quantized-KV reads in flash attention: see 005.

## Further Notes

Evidence pointers (line numbers as of `bddf8263c`, upstream/master):

- serialized tile loop: `ggml/src/ggml-cuda/mmq.cuh`, `mul_mat_q_process_tile`
- synchronous weight loads: `ggml/src/ggml-cuda/mmq-load-tiles.cuh`
- Ada/Ampere config rows: `ggml/src/ggml-cuda/mmq-config-ampere.cuh`
- arch macro that gates the branch: `turing_mma_available` /
  `ampere_mma_available` in `ggml/src/ggml-cuda/common.cuh`
- async copy helpers already in tree: `ggml/src/ggml-cuda/cp-async.cuh`
- MMQ is chosen at every batch size on Ada: `ggml_cuda_should_use_mmq` in
  `ggml/src/ggml-cuda/mmq.cu` (003 finding, still true here)
- Shares and totals: `specs/001-baseline.md`, section "Kernel breakdown", P2
- Blocked tooling is now unblocked: `ncu` works when run as root
  (`sudo -E ncu ...`, `--replay-mode application` keeps the 4.8 GB model from
  being snapshotted). Unprivileged runs fail with `ERR_NVGPUCTRPERM`; the driver
  option is `NVreg_RestrictProfilingToAdminUsers=0` in
  `/etc/modprobe.d/nvidia.conf`, then `mkinitcpio -P` and a reboot. Your
  `/sys/module/nvidia/` exposes no `parameters` directory, so verify by running
  ncu without sudo.
- Probe artifacts: `build-004` (CUDA-only, arch 89), A/B json in
  `/tmp/004/{A,B,C,types,ub}.json`, baseline library copy in `/tmp/004/A/`,
  nsys capture `/tmp/004/pp4096.nsys-rep` plus `pp4096.sqlite`, ncu text
  `/tmp/ncu004.txt`.
- The MMQ `I=128` assumption above was found with
  `CASE(GGML_TYPE_IQ4_XS, 256, 2, 64, 128, ...)` in the Ampere table. That edit
  is reverted; `build-004/bin/libggml-cuda.so` matches `/tmp/004/A/`.
- 004b is superseded by spec 010. The step it recommended first, `ldmatrix` operand
  delivery, turned out to be already implemented: MMQ calls `load_ldmatrix` for the
  weight operand at 22 sites in the vec-dot layer, and uses plain shared-memory loads
  for the activation operand deliberately, with a comment that this is faster. The
  static instruction mix of the shipping kernel is 12.2% memory-pipe instructions and
  52% int32-to-fp32 scale-correction work (`I2FP`, `FMUL`, `FFMA`) against 4.3%
  `IMMA`. The target is the correction, and 010 works it out from those measurements.
