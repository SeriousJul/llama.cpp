# Specs: 4090 optimization of the inference engine

One spec per file. Statuses: `active`, `blocked`, `deferred`, `done`.
Opened 2026-09-23 through 2026-09-24: 004-010, all prompt-processing on sm89. Two
landed and are committed locally (006, 007). Everything else is closed as `no
change`, by measurement or by reading the code: 002, 003, 004, 005, 006, 009, 010. One
spec is open with a frozen design (008). 006's two SSM increments are the shipped gain; its
chunked restructure was built through stage 2, measured at -10.0 % and -20.2 % on pp4096, and
closed, with the code kept as a patch under specs/artifacts/.

| file | title | status |
|---|---|---|
| 001-baseline.md | Baseline measurements and kernel breakdown | done (2026-09-23) |
| 002-long-context-decode-attention.md | GQA redundancy and q8_0 verify-batch path | done (2026-09-23, verdict: no change) |
| 003-fp8-prefill-gemm.md | FP8 block-scale GEMM for sm89 | done (2026-09-23, verdict: no change) |
| 004-mmq-cp-async-pipeline.md | cp.async weight-tile pipeline in MMQ | done (2026-09-23, verdict: no change; ncu says LSU/issue bound at 1 CTA per SM, not load bound) |
| 005-fattn-mma-quantized-kv.md | q8_0 KV read by the MMA attention kernel | done (2026-09-24, verdict: no change): removable part is 3.1%, kernel is tensor/L2 bound at 212 registers per thread, both table retunes lose |
| 006-gdn-chunked-prefill.md | Chunked gated-delta-net prefill kernel | closed (2026-09-24), verdict no change. Shipped gain is increments 1 and 2, committed as 42d24e195 local only (SSM kernel x1.86, 27B pp4096 +7.3 %). Chunked stages 1 and 2 were built and are preserved as `specs/artifacts/006-chunked-scaffold.patch`; measured paired ABBA they are -10.0 % (C=16) and -20.2 % (C=64) on 9B pp4096, and stage 3 was not written because the one shape that could rescue it, `M = 16` mma tiles, has never been measured. Findings that stand: one block per (head, sequence) is 32 CTAs against 128 SMs; f16 tiles land on the seam's 1e-7 NMSE bar by construction; the gate has to be carried in log space |
| 007-fuse-glu-into-mmq-quantize.md | GLU folded into the q8_1 pre-quantization | done (2026-09-23, committed 247376881 local only): +1.5% 27B pp4096, perplexity identical |
| 008-fattn-dv-split-occupancy.md | Split DV across CTAs to buy attention occupancy | active, both gates measured (2026-09-26), criterion 1 fails. A half-DV CTA at the production tile wants 192 registers against a 128 budget, and forcing it pays in local memory (STACK 16 -> 112). Shared memory does not refuse it - 34816 B/CTA leaves room for 2 CTAs, so 16 warps was on the table; my first pass claimed otherwise from arithmetic on one term of a max(). Quartering DV is the only shape that meets 128 (REG 128, STACK 48) and half DV at ncols=32 is the only clean step up (REG 168/170, 12 warps). The premise itself broke: DV 128->64 buys 32 registers at ncols=64 and zero at ncols=32, so there is a ~168-register per-thread floor that is not the accumulator, and full DV never goes below 186 at any tile or pipeline setting. The reachable shapes need 2.5x-3x the KV reads against an L2 already at 63.0%. Decision pending |
| 009-mmq-j-occupancy.md | J as the occupancy knob for the prefill GEMM | done (2026-09-24, verdict: no change): occupancy was reached and cost -20.9% on pp65536; every non-instruction-count route into MMQ is now measured and lost |
| 010-mmq-scale-correction-epilogue.md | MMQ shared-memory traffic and the I = 128 assumption | done (2026-09-24, verdict: no change): six kernel routes into MMQ all closed; the epilogue cost is set by scale granularity, a format decision, not a kernel one |
| 011-uvm-prefetch-hints.md | `cudaMemAdvise` / `cudaMemPrefetchAsync` on the UVM path | active (baseline 2026-09-26): managed == cudaMalloc while the working set fits (0.3% at q8_0 262K); at ~3.4 GiB spill decode falls 31.1 -> 0.6 t/s. The hints target that cliff |
| 012-phase-arena-workspace-reuse.md | Return prefill graphs and pp workspace to the decode KV budget | done (2026-09-26, verdict: no change): graphs cost 16 MiB and pp and decode never hold two at once - one `llama_context` reuses one cgraph, so both phases key on the same `nodes[0]` and alternate through one graph slot, re-capturing in place for 0 MiB. The band is a fixed ~976 MiB init floor, 391 MiB of it the CUDA context. Ceiling ~36 MiB, not the fork's 2304 MiB |
| 013-pipelined-kv-streaming.md | Block-granular, lookahead KV staging beyond VRAM | deferred, gate zero answered: q8_0 ctx_max ~285K > production need 160K; re-open only for f16 >150K, q8 >280K, or a second co-located model |

Each spec states: measured problem, design, acceptance criteria, test method.

011-013 are a different family from 001-010: they come from the 2026-10 analysis of
`RaymondHuang210129/llama.cpp-adaptive-kv-streaming` (adaptive KV streaming for
long-context on small-VRAM CUDA boxes). None of them copies code from that fork. 011
addresses the missing page hints in our UVM path, 012 the phase-idle bytes its arena
reclaims, 013 the lookahead streaming its ring adds over our `-nkvo` scheduler path. All three are
gated: no patch until each file carries its own baseline, per the standing rule. 011 and 012
are now closed by measurement; 012's `cudaMemGetInfo` probe is what tells 013 that a pool
flush really does return bytes to the driver (0 MiB sticky after a 12 GiB round trip), which is
the mechanism 013 would need if its gate ever reopens.

`specs/artifacts/` holds the throwaway validators, the ncu text dumps and the paired
A/B raw data the numbers in these files come from. They are not part of the build and
are not meant to be upstreamed; they exist so a measurement can be re-read without
re-running it. The A/B libraries under `/tmp` were not archived, they are one rebuild
away.
No spec starts work until its baseline numbers exist in the file.

005 and 007 targeted prompt processing; 007 landed (247376881), 005 remains open. They share the same seam
(`tests/test-backend-ops.cpp`) and the same measurement harness
(`tools/bench/server-bench`, `tools/bench/fattn-ab-capture`). Expected gain if
all three land, from the 001 shares: 15-25% off pp32768, 12-25% off a 131k
prefill. 005 and 007 claim different kernels, so their gains add; re-measure the
second on top of the first rather than adding the numbers. 004 is closed by ncu:
the largest block in the profile (IQ4_XS MMQ, 54.3% of 9B prefill kernel time)
sits at 38.3% tensor-pipe and 49.0% LSU with occupancy pinned at 16.7% by 254
registers and 57.9 KB of shared memory per block. That is an operand-delivery
problem, not a memory-fetch problem. See 004.

006 has two landed increments and no chunked algebra yet. Increment 1 gives each
lane 4 contiguous state rows so k, q and the state move in 16 B accesses; increment
2 gives each warp 4 columns so the k and q reads amortize, with one vectorized
output store. Together: LSU instructions per column-token 22 to 12.25, the SSM
kernel x1.86 per call on the 27B (500.4 to 268.8 us), its share of prefill 14.85%
to 8.56%, pp4096 +7.26% on the 27B and +6.22% on the 9B, decode +0.04%, op tests
36/36. Shuffles are 10 of the remaining 12.25, which is why the next real step is
the tensor-core chunk form and not more SIMT tuning. Perplexity is not yet
re-measured and is the gate for proposing this upstream.

Stages 1 and 2 of the chunked kernel landed 2026-09-24 behind `GGML_CUDA_GDN_CHUNKED`
(default off): the scaffold, then the fp32 SIMT chunk body, green through the op suite
with the flag on. With the flag off the device code is byte-identical to the committed
tree, and Wave64 builds are compiled in the same switch as the shipped shapes. Three
things the plan did not know are recorded in 006: the grid, not shared memory, is 006's
binding constraint; the gate has to be carried in the log domain or the long-sequence
test cases produce 0/0; and f16 tiles sit on the seam's tolerance exactly, which is a
decision stage 3 has to make before it writes mma loops, not after.

New measurement from the 9B nsys capture (pp4096, ubatch 512): the shares differ
from 001's 27B table in one important way. `gated_delta_net` is **17.2%** of
kernel time on the 9B against 13.1% on the 27B, and `flash_attn_ext` is only
2.0% at pp4096. So 006 is a bigger share on short-to-mid prompts than 001
suggested, and 005's attention share grows only with context length. These are
within-capture ratios, which are the part that survives clock drift; the
absolute microseconds in that capture are not comparable to a later one.

## Hardware

- NVIDIA GeForce RTX 4090, 24 GB GDDR6X
- 128 SM, 16384 CUDA cores, 65536 x 32-bit registers per SM
- 100 KB L1/shared memory per SM, 99 KB max per CTA, 72 MB L2
- 1008 GB/s memory bandwidth
- Tensor peak (dense, fp32 accum): FP16 165 TFLOPS, FP8 e4m3 330
  TFLOPS, INT8 660 TOPS. The 2x-larger vendor numbers (330/661/1321)
  are fp16-accum and/or 2:4 sparsity; GGML needs fp32 accum for f32
  output, so INT8 is the fastest 8-bit lane on Ada (see 003).
- No FP4 tensor cores (Blackwell only), no wgmma (Hopper only)
- FP16 MMA atom m16n8k16, FP8 MMA atom m16n8k32, both warp-level mma.sync

## Production stack

- llama-server fork, service `llama-server.service`, port 8080
- `--models-preset presets.ini --models-max 1 --sleep-idle-seconds 1800`
- Target model: `unsloth/Qwen3.8-27B-GGUF:IQ4_XS` (alias `qwen3.8-27b-dflash`)
- Spec decode: dflash draft `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`,
  `spec-draft-n-max 7`, plus ngram-map-k4v; verify step = 1 to 8 tokens
- KV cache: q8_0 (K and V), ctx-size 160000, cache-ram 32768,
  cache-prompt on, flash-attn on, kv-unified on
- Observed (journalctl, 2026-09-22): prefill 530-1185 t/s on fresh tails,
  decode 51-64 t/s with spec at 60k-134k context

## Model architecture (Qwen3.8-27B, arch `qwen35`)

- 64 decoder layers + 1 MTP layer
- Hybrid: `full_attention_interval = 4`, so 16 full attention layers and
  48 gated-SSM (linear attention) layers
- Attention: 24 query heads, 4 KV heads (GQA 6:1), head dim 256, RoPE 64
- Embedding 5120, FFN 17408
- KV per token (q8_0): 2 x 16 x 4 x 256 x 1.03 B = 33.6 KB
- KV at 131072 tokens: ~4.4 GB; bandwidth floor at full 1008 GB/s: ~4.5 ms/token
- SSM state is O(1) per token, not proportional to context

## Dev model

`unsloth/Qwen3.5-9B-GGUF:IQ4_XS` (arch `qwen35`, 32 layers, 8 full attention
layers, 16:4 GQA, head dim 256, same SSM structure). Same kernel paths as the
27B, ~3x faster iteration. Used for kernel A/B runs.

## Terminology

- **prefill**: prompt processing, batch up to 2048, compute bound
- **decode step**: one forward pass of the target model during generation
- **verify step**: a decode step under spec decode, batch 1-8 tokens
- **KV floor**: minimum decode attention time = KV bytes per token / 1008 GB/s
- **fattn**: the CUDA flash attention kernel family in `ggml/src/ggml-cuda/`
  (`fattn.cu`, `fattn-vec.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cu`)
- **MMQ**: the quantized-weight GEMM kernel (`mmq.cu`), used for
  `ne11 < 64`; larger batches dequantize to F16 and run cuBLAS

## Method

- Kernel A/B: `llama-bench` on the target model alone, q8_0 KV, ngl all
- Production acceptance: `tools/bench/server-bench`, which unloads the live
  model, runs the matrix, and reloads it
- Kernel attribution: nsys profiles of one prefill and one decode step

## Workflow notes (lessons learned)

- **Verify what a benchmark row measures before trusting it.** llama-bench
  splits `-p X -n 1` into `ppX` (prompt only) and `tg1` (generation in a
  fresh context, `n_prompt = 0`). The test column prints exactly this:
  `ppN`, `tgN`, or `ppN+tgN`. Long-context decode requires `-pg X,1`.
  The 001 matrix originally used `-p/-n` for D1/D2, so the "long ctx
decode" rows were actually small-context decodes. Caught only when the
  kernel times did not match the physics (a 10 us attention kernel cannot
  read 273 MB). Cross-check: a number that should scale with context but
  does not is a measurement-semantics bug, not a kernel fact.
  Same class of bug: the `ppN+tg1` combined t/s is (N+1)/(t_pp + t_tg),
  so with N = 131071 and t_pp ~ 69 s the ~20 ms decode is invisible in
  the table. The decode at long context must be read from the nsys
  capture (nd2l), not from the combined t/s.
- **Absolute t/s and kernel durations drift with the SM clock, so only a paired
  A/B is a result.** On 2026-09-23 the *same* baseline library measured 10139.8
  t/s and, about an hour later, 10661.6 t/s, a +5.1% swing with no code change:
  the SM clock moved from the ~2.45 GHz that the ncu report recorded to
  2700-2715 MHz. An unpaired comparison taken across that gap produced a
  convincing-looking "+6.3% on pp1024" for a change that was actually -0.05%.
  Rules from now on: interleave A/B as ABBA within each round over 2-3 rounds;
  md5-verify the on-disk library before every run; sample `clocks.sm` on each
  row; never compare absolute kernel microseconds across two nsys captures,
  compare shares within one capture, or the target kernel's change against a
  control kernel in the same capture. Harness: `/tmp/004/ab.sh`, which does all
  four and refuses to run on an md5 mismatch.
- **Cross-check kernel instance counts against the expected structure**
  (passes x ubatches x layers) before trusting a breakdown. 24576 =
  6 x 256 x 16 confirmed the D2 profile is 6 prefill passes and isolated
  the single decode step.
- **Time estimates must be checked against bandwidth before believing a
  kernel analysis.** Derive bytes moved from the tensor shapes and the
  grid mapping, divide by 1008 GB/s (or the per-CTA share, ~7.9 GB/s),
  and compare with the measured duration. Mismatch means a wrong shape
  assumption, not a fast kernel.
- **nsys sqlite:** kernel rows live in `CUPTI_ACTIVITY_KIND_KERNEL`;
  names resolve via `StringIds` (join on `k.shortName = s.id`); start/end
  are nanoseconds since session start. There is no memory table, so bytes
  moved must be derived from shapes, not read from the trace.
- **GGUF v3 header layout:** magic, u32 version, u64 tensor count, u64 kv
  count, then keys and values; strings are u64 length-prefixed (no null
  terminator); array values carry an element type (u32) and a u64 count.
  The qwen35 GGUF here carries no SWA/window keys: the 16 attention
  layers are full attention over the whole context.
- **Check the binary matches the source you are reading.** llama-bench
  prints its build commit. If the tree moved (upstream pull), diff the
  relevant subtree between the build commit and HEAD before interpreting
  dispatch logic.
- **Never run the bench in the background (bash_bg). Run it as a
  blocking foreground call.** The agent session is served by the same
  llama-server the bench stops, so the session cannot run while the
  bench is active. A backgrounded bench was killed after ~2 minutes
  (2026-09-23) and left a partial nsys capture. The only way that
  works is sequential: one blocking foreground command, then continue.
- **The production server serves continuously.** Before any unload
  (server-bench, or a manual probe), check recent `journalctl` for
  running tasks and time the run for a gap; do not stop it mid-task.
  The script's EXIT trap restarts the service even when the bench is
  killed, which is what recovered it on 2026-09-23.
- **`llama-bench` hides the ggml error text unless you ask for it.** A crashed
  run shows only `ggml-cuda.cu:108: CUDA error` plus a backtrace, because
  `GGML_LOG_ERROR` goes to buffered stdout and `abort()` drops it, and
  `stdbuf`/`CUDA_LAUNCH_BLOCKING` do not bring it back. Run with `-v`:
  that is what turned a mystery crash into `CUDA error: an illegal memory access
  was encountered` in `ggml_cuda_compute_forward: MUL_MAT`.
- **nsys gives the MMQ template arguments for free, which is how you tell the
  paths apart.** Join `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` (not
  `shortName`) to `StringIds` and the names arrive as
  `mul_mat_q<(ggml_type)23, (int)128, (bool)0>`, i.e. type, J, fallback, which
  separates the main path from the `fallback = ne01 % 128 != 0` path and from
  `mul_mat_q_stream_k_fixup`. `nsys stats --sqlite` gives a queryable file.
- **Route A for counters, no system change:** ncu works as root because the
  driver restriction only covers non-admin users. Use
  `sudo -E ncu --replay-mode application ...` so the 4.8 GB model is not
  snapshotted per pass, `--section` with an explicit list rather than
  `--set full`, and pipe through `tee` as the unprivileged user so the output
  file stays yours. Expect ~18 application replays for 8 sections, roughly 3
  minutes. The permanent form is
  `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in
  `/etc/modprobe.d/nvidia.conf`, `sudo mkinitcpio -P`, reboot: a reload without
  a reboot is not possible while the desktop and llama-server hold the GPU.
- **A llama-bench row with several backends is not a CUDA number.** `build/` is
  configured with `GGML_VULKAN=ON`, and this host has a second GPU (the
  9950X3D iGPU, `RADV RAPHAEL_MENDOCINO`). llama-bench with `-ngl 99` then
  spreads layers over both, so its pp t/s is a hybrid figure and cross-type
  comparisons come out wrong (it made Q8_0 and IQ4_XS look 2.8% apart when the
  true CUDA-only gap is 3.4% with different absolute rates). For kernel work,
  build a `GGML_VULKAN=OFF` tree (`build-004` as of 2026-09-23) and compare
  only within it.
- **`ncu` cannot be used by the unprivileged user here.** It fails with
  `ERR_NVGPUCTRPERM`. Either run it as root or set
  `NVreg_RestrictProfilingToAdminUsers=0` in an `nvidia` modprobe drop-in and
  reload. Until then, answer stall questions with counters-free probes: change
  the bytes, change the load count, change the batch, and see which one moves.
- **Decode attention dispatch (quantized KV, Ada):** Q->ne[1] <= 2 takes
  the vec kernel (q8_0 KV consumed directly, one CTA per
  (query-tile, q-group, kv-head)); Q->ne[1] > 2 takes the mma-f16 kernel
  with a full layer KV dequant to f16. The vec kernel already supports a
  KV split along gridDim.y with a dst_meta combine; the split size comes
  from the launch_fattn efficiency loop, which degenerates to 1 when the
  KV is short. Confirmed by the D2L capture (2026-09-23): small-ctx
decode gridY = 1, 131k decode grid 1x21x24 (504 CTAs), combine pass
0.05 ms per step. The split exists and is near DRAM peak; the remaining
  attention waste is the GQA re-read (6 q-groups read the same kv-head
chunk) and the mma-f16 full-KV dequant for verify batch > 2. UPDATE
(2026-09-23, 002 results): the dispatch relaxation and the cols=8 vec
attempt were both falsified by the nsys A/B. The mma-f16 + dequant path
(~25 ms per verify step at 131k) is near its traffic floor and stays;
the cols=8 vec kernel is 2x slower. See the spec.
- **The router auto-reloads the model on status endpoints.**
  `GET /models`, `/slots`, and friends call `ensure_model`, which loads
  the model if it is not loaded. The agent's own polling during a
  capture window kept re-triggering unloads and reloads (2026-09-23).
  For any exclusive-GPU window, stop the whole service
  (`systemctl --user stop`), never unload through the router, and send
  no HTTP to the port at all until the window closes.
- **Killing an nsys-wrapped server needs the process group.** TERM to
  the nsys wrapper PID does not reach the child llama-server; the child
  survives, holds the GPU, and the .rep is never finalized. Start the
  server with `setsid` and kill with `kill -TERM -- -PID` (negative pid
  = the group). Verify with `nvidia-smi` that the GPU is free before
  the next run.
- **bash `local` shadows globals under `set -u`.** A `local srv_pid`
  in the calling function made a helper that reads the global see an
  empty string and die with "unbound variable". Declare locals only
  where they are set, or pass values as arguments.
- **E2 wall time at long context on repetitive prompts is CPU ngram-map
  work, not GPU.** The ngram-map-k4v speculator runs on the CPU and, on
  the repeated-filler E2 prompt at 131k, covers most of the 0.4-1.3 s
  gaps between GPU bursts (81% of the eval window, 2026-09-23). E2 t/s
  at long context is not a clean metric for GPU-path changes; use the
  per-step kernel sum from the nsys eval window. The `eval time` line
  from `print_timing` anchors the window; check it against the trace
  span and the gap structure first.
- **The model loads lazily on the first real request.** After a service
  restart, the router answers but the model shows `unloaded` until
  someone sends a request. A post-capture check that polls for
  `loaded` can never pass on its own; check only that the router
  answers, and let the next user request do the load.
- **A library swap takes effect only on service restart.** Unloading
  and reloading the model does not re-read `libggml-cuda.so`. For .so
  A/B runs: copy the file into `build/bin`, restart the service, and
  verify the md5 of the on-disk file matches the intended library.
  `fattn-ab-capture` takes tag arguments (`old`, `new`, or both) so a
  missing capture is re-run without touching the good one, and it
  honors `REAL_PROMPT=<file>` to profile a non-filler context (count
  the tokens with llama-tokenize first; this code corpus ran ~4.0
  chars per token).
- **Attribute a per-case error only from a one-case-per-process run.** test-backend-ops prints
  `[OP] ERR = ...` with no newline and repeats each case name in its own result line, so a grep that
  pairs an ERR with a nearby case name silently mixes cases. Confirmed the hard way in 006 stage 2:
  mispaired reads made an f16 build look identical to an fp32 one, and a build that had failed to
  compile reported the previous library's numbers. Run `-p '.*<case>.*'` per case, and have the
  build print something that identifies the configuration (006 prints its `nbytes_shared` request)
  so a stale object cannot masquerade as a result.
- **build-004 is now configured `LLAMA_BUILD_SERVER=ON`.** It was OFF, which made `cmake --build`
  fail on `test-chat` and `llama-app` (both link `server-context`, which only exists with the server
  on), and the reported error was a missing `mtmd.h` that had nothing to do with the real cause.
  The full tree builds clean as of 2026-09-24 and `libggml-cuda.so` md5 is unchanged by it.
  `tests/test-backend-ops.cpp` also had six `-Wdeprecated-enum-enum-conversion` warnings from mixing
  `ggml_scale_mode` with `ggml_scale_flag`; the casts are named now and UPSCALE/INTERPOLATE still
  pass 22/22.
- **test-backend-ops reseeds its inputs per process, so one green run is one sample.**
  `init_tensor_uniform` uses `std::random_device`, so the same case in the same binary gives a
  different error every time the process starts. A marginal tolerance therefore looks like an
  intermittent bug (006 stage 2: one failure in fifteen) and cannot be chased by re-running the
  suite; it has to be characterized as a distribution, by running the single case in N processes and
  reading the spread. Corollary: a gate that runs the suite once proves less than it looks, so
  `specs/artifacts/006-gate.sh` runs it eight times.
- **A tensor-core rate probe must prove its own work survived.** If only some accumulator elements
  reach an observable, nvcc deletes the rest of the mmas and the probe reports a rate above the
  hardware roof: 006's whole-chunk probe printed 470 and 4054 TFLOP/s on a 165 TFLOP/s card, and only
  `cuobjdump -sass | grep -c HMMA` per instantiation exposed it. Two rules follow: keep every
  fragment live through a sum that reads all of them, and print the HMMA count beside every number,
  rejecting any config whose count does not match the tiles it should issue. Pass `ldmatrix`
  addresses as 64-bit with an `l` constraint like `mma.cuh` does; a `(unsigned)` cast faults at
  15 KiB of shared memory and can pass unnoticed in a smaller probe.
- **`llama-cli` under a non-TTY stdin can spin the conversation loop.** With
  `-f prompt.txt` and no `-st`/`--single-turn`, it enters interactive mode against
  an EOF'd stdin and emits `> ` forever: one such run grew a 9.9 GB artifact file
  and looked exactly like a UVM thrash hang until `wc -c` said otherwise (2026-09-26,
  window 3). Always pass `-st`, always per-run `timeout`, and check the artifact
  size before believing a hang is a measurement.
- **`pkill -f` matches the invoking shell's own command line.** Two windows were
  self-killed (rc 143) because the kill pattern text appeared in the bash -c line
  running it. For port-owned single processes kill by port: `fuser -k PORT/tcp`.
- **Size synthetic prompts with `llama-tokenize -m MODEL --stdin --show-count`, not
  bytes.** This filler tokenizes at 5.58 B/token, so a 657 KB "260K" file was 117.8K
  tokens and three matrix rows were mislabeled until recounted (2026-09-26).
- **Read mma layouts out of `mma.cuh`, not out of the PTX tables.** 006 stage 3 lost a session step
  to a hand-written `ldmatrix.x2.trans` for the B operand of `m16n8k16.row.col`: validation against a
  CPU matmul showed both column registers getting the same `n`. `mma.cuh`'s `tile<8,8,T>`
  `load_ldmatrix` uses `x2` with no transpose on the same address pattern. One word, and a probe that
  had been reporting nothing became a result.
- **A green op suite is not proof the path under test ran.** 006 stage 1 passed 36/36 with the new
  chunked kernel compiled and never selected: the only long-sequence cases at the production head
  size live in `make_test_cases_perf()`, which is a timing list, not a correctness check, while the
  checked eval set paired head_size 128 with 1 token and long sequences with head_size 64. Count
  launches (a `fprintf` in the launcher, or a temporary `GGML_ABORT`) before believing a gate, and
  when a case list is split between an eval function and a perf one, say which one covers the shape.
- **An intermittent correctness failure with a clean racecheck is a tolerance overshoot, not a race.**
  006 stage 2's f16 tile draft failed roughly one full suite run in fifteen at `ERR = 1.09e-7`
  against `max_nmse_err = 1e-7`, while the same case passed 10/10 in isolation and
  `compute-sanitizer --tool racecheck` found 0 hazards. Two things explain it: the harness draws
  inputs from an rng stream that advances with the number of cases run ahead of the failing one, so
  a marginal case lands differently alone than in the full list; and the seam's metric is NMSE, so
  `sqrt(1e-7) = 3.2e-4` is the relative error it can carry, which is exactly f16 epsilon. Any f16
  operand path sits on that boundary by construction. `test_ssm_scan` is the in-tree precedent for
  handling it: it overrides `max_nmse_err()` to 2e-7 because its path uses fp16 intermediates.
- **Exhaust the existing infrastructure before designing new code.**
  003 assumed a missing FP8 path and a multi-week kernel project. The
  tree already had MXFP4/NVFP4 types, quantizers, MMQ configs, and
  per-tensor `--tensor-type` overrides in llama-quantize: a dense
  FP4 build took one command, and the full verdict (speed, quality,
  size) cost one evening instead of a kernel rewrite. Always check
  dispatch gates (`should_use_mmq`, arch availability) and existing
  types first; prototype with what exists, measure, then decide.
- **Vendor peak numbers are marketing unless you match the
  accumulator and density.** 4090 "FP8 661 / INT8 1321" are fp16-accum
  and/or sparse rates. With the fp32 accum GGML needs: FP16 165,
  FP8 330, INT8 660. A design premised on "FP8 is 2x INT8" is wrong
  by 2x in the bad direction. Derive headroom from measured effective
  throughput (001's mul_mat_q timings) instead.
- **A throwaway instantiation is the cheapest gate there is.** 008 froze a design on an
  estimate of ~130 registers for a half-DV CTA. Compiling DKQ=256/DV=128 next to the shipping
  rows and reading `cuobjdump -res-usage` on the object costs one file compile (~1 min) and
  said 192, with STACK 16 -> 112 when launch bounds force the 128 budget. Do that before
  writing any grid or fixup code, and before any benchmark.
- **Ask the device which limit binds; do not hand-combine two of them.** 008's shared-memory
  table was wrong in the permissive direction on the first pass: `nbytes_shared_total` is
  `max(combine, max(Q, KV + mask))`, the Q term has no DV in it but the combine term scales with
  `nwarps`, so reasoning from one term and ignoring the other produced a ceiling that did not
  exist. Replay the launcher's own expression and call the same
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` `launch_fattn` calls, with a dummy kernel whose
  only relevant property is its thread count, so the register file cannot interfere and the smem
  answer is the smem answer (`specs/artifacts/008-smem-occupancy.cu`). Read registers separately
  from `cuobjdump -res-usage` on a probe instantiation, and take STACK, not just REG, as the
  verdict: REG at the budget with STACK 7x baseline is the compiler paying the bill out of local
  memory, which is what -23.9% looked like before it was a number.
- **Read the launch path for unconditional dispatch before scoping a "first cut".** 008
  planned to decline the split for stream-K, fixup and sparse. On sm89 `should_use_stream_k`
  returns true for every NVIDIA cc >= Ada (`fattn-common.cuh:1144`) and the MMA case always
  passes `stream_k = true`, so that scoping would have declined the feature everywhere.
- **An unattributed number is not a ceiling.** 012 read `unaccounted`
  (924-994 MiB) for two days as a possible reclaim. It is a fixed
  per-process init cost: 976 MiB with 19.0 GiB of buffers, 976 MiB with
  13.3 GiB, 931 MiB with `-ngl 0`, 391 MiB of it the CUDA context and
  ~8 MiB per cuBLAS handle (probe `specs/artifacts/012-cuda-floor.cu`).
  The cheapest attribution test exists already: run the same binary and
  send it no request at all.
- **Read the whole tree before declaring a switch missing.** 012's
  finding 3 claimed "there is no user-facing switch to disable CUDA graph
  capture". `GGML_CUDA_DISABLE_GRAPHS` is read in
  `ggml_cuda_graph::is_enabled()` (`common.cuh:1288`, consulted at
  `ggml-cuda.cu:4489` and `:4508`). The grep that missed it looked only in
  `ggml-cuda.cu`; the switch lives in the header next to the struct.
- **`graphs reused` is not a CUDA graph counter.** It is
  `llama_context::n_reused` (`llama-context.cpp:1415`), bumped whenever the
  previous ggml cgraph object is reused. It printed 1019 in a run with CUDA
  graphs disabled by env. Proof that the capture path ran must come from the
  capture site, or from the memory delta it causes.
- **A capture log without the graph key cannot say which phase captured.**
  The pp graph and the tg graph of the same model have the same node count
  (3942 on the 27B), so 012 window 4 read the same either way; printing
  `graph_key` alongside it is what turned "the decode graph owns the 16 MiB"
  from an inference into a measurement, and it showed the inference was wrong
  - pp and tg share one slot.
