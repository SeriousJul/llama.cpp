# 002: Long-context decode attention (GQA redundancy + batch path)

Status: done (2026-09-23, verdict: no change; GPU-side scope exhausted)

## Problem

The 001 baseline measured the true 131k decode step (nd2l nsys capture,
batch 1, q8_0 KV):

- Step: 30.0 ms (kernel sum 27.2 ms under nsys)
- Weight GEMV (mul_mat_vec_q): 15.7 ms, 53%
- Attention (fattn vec): 8.8 ms, 29%
- Norm/quant/SSM/rest: ~6.5 ms, 22%

The KV floor for one 131k step is 4.4 GB / 1008 GB/s = 4.35 ms. The vec
kernel runs at 2.0x the floor. The excess is GQA redundancy: grid z = 6
q-groups x 4 kv-heads, so every kv-head chunk is read by 6 CTAs. Per
layer: 903 MB of read requests vs 150 MB unique. L2 absorbs about 40%.

Production adds a second path. The dispatch in fattn.cu sent
`Q->ne[1] <= 2` to the q8_0 vec kernel and everything else to mma-f16,
which dequantizes the whole layer KV to f16 first. The E2 production
verify steps run at batch 2-8 (spec-draft-n-max = 7), so most of them
took the mma-f16 path.

## Design (as originally proposed)

- Item A: relax the Ada quantized-KV dispatch from `<= 2` to `<= 8` so
  verify batch 3-8 takes the q8_0 vec kernel (no f16 materialization).
- Item B (cols8): set `cols_per_block = 8` in the vec kernel for batch
  3-8 so one CTA serves all verify tokens and the KV chunk is read once
  per (q-group, kv-head) instead of once per 2-token tile.
- Item B (packing, never built): pack the 6 q-groups of one kv-head into
  one CTA so the kv chunk is read once per layer. The real kernel
  redesign.
- Item C: Ada tile table for mma-f16 prefill attention. Deferred with
  003; prefill is not the felt bottleneck.

## Results (2026-09-23, measured)

Method: `tools/bench/fattn-ab-capture`. The service is stopped, the
llama-server child runs standalone under nsys with the target preset,
one 131k-context E2 request (131127 prompt, 256 max tokens) per
library. The .so is swapped between runs; the kernel sums in the eval
window are the A/B metric, not the wall t/s (see the ngram finding
below). Both captures: service stopped, no router traffic, no other GPU
work.

### Verify-step attention at 131k, per layer (steady state)

| path | kernels per layer | per layer | per step (16 layers) |
|---|---|---|---|
| OLD: mma-f16 + dequant | dequant K 0.41 ms + dequant V 0.46 ms + mma 0.62 ms | ~1.5 ms | ~25 ms |
| NEW: vec cols=8 (one CTA, batch 3-8) | vec<256,8,q8_0> grid 1x21x24 | ~3.0 ms | ~48 ms |
| vec batch 1 (001, reference) | vec<256,1,q8_0> grid 1x21x24 | ~0.55 ms | ~8.8 ms |

The mma-f16 + dequant path costs ~25 ms per verify step, close to the
22 ms traffic estimate in the problem section. It reads the KV once per
layer (dequant 273 MB + 546 MB write + 546 MB f16 read) and the mma
kernel is near the f16 read floor.

The cols=8 vec kernel is the problem, not the design goal. It has the
same grid (1x21x24, 504 CTAs) and the same KV traffic per CTA as the
batch-1 kernel, but each CTA does 8x the QK/PV work in an inner loop
that serializes the 8 columns. Measured 3.0 ms per layer vs 0.55 ms for
batch 1: 5.5x slower with the same memory load. Build check was clean
(255 registers, no local spills, 400 B stack), so the cost is the inner
loop structure, not spilling. The vec kernel is latency bound at
ncols = 8 and would need a register-tile rework to compete, which is a
kernel project, not a dispatch change.

### Verdicts

- Item A (dispatch `<= 8`): falsified. With the existing cols_per_block
  (1 or 2), verify batch 3-8 re-reads the KV per 2-token tile: 4x the
  batch-1 traffic at batch 8, ~35 ms per step. E2 A/B ran at 17.59-18.85
  ms/token vs the 15.79 baseline, within the wall noise (below). The
  simple relaxation helps nothing and can hurt.
- Item B cols=8: falsified. 48 ms per verify step, 2x the mma-f16 path
  it replaced.
- The mma-f16 + dequant path stays. It is within ~3 ms of its own
  traffic floor and beats both vec options at verify batch 3-8.
- Both changes were reverted. The service runs the baseline
  `libggml-cuda.so.0.24.0` (md5-verified) since 2026-09-23 17:47.

### Finding: the E2 wall at 131k is CPU ngram-map work, not GPU

The eval window of both captures (30056 kernel rows in 4.04 s, OLD)
shows the GPU silent for 0.4-1.3 s at a time, five gaps covering 81% of
the window. The GPU is 14% busy (OLD) to 21% busy (NEW). Only ~10 of
the ~69 spec steps run a main-model verify forward (160 mma attention
calls / 16 layers); the DFlash2 draft model forward also runs only a
handful of times (5 draft-attention calls in the whole trace). The
remainder of the 256 tokens comes from the ngram-map-k4v speculator
working on the CPU: the E2 prompt is repeated filler lines, which is
exactly what a k4v ngram map over a 131k context predicts.

Consequences:

- E2 t/s at 131k measures the CPU gap distribution plus acceptance, not
  the attention path. The 15.79 vs 17.59 vs 18.85 vs 19.92 spread
  across runs is this noise, not the kernel A/B.
- The A/B metric for GPU-path work is the per-step kernel sum in the
  eval window, sliced with the `eval time` from `print_timing` and
  checked against the trace span and the gap structure.
- On real (non-repetitive) sessions the ngram map hits less and the GPU
  verify step is a larger share of the wall. The 25 ms verify attention
  is then ~40% of the step. That is the remaining lever, and it is the
  GQA packing item, not a dispatch change.

## Realistic-context profile (2026-09-23, closes the spec)

A second capture used a non-repetitive 130438-token prompt (13 distinct
source and doc files, no repeated lines) to remove the ngram-map
shortcut the filler prompt gave. Same baseline library, same preset,
service stopped, single run.

- Wall: 5314 ms / 256 tokens = 20.84 ms/token. Draft acceptance 0.296
  (171 / 577), mean len 3.04.
- GPU kernel sum in the eval window: 255 ms total, 4.8% of wall. The
  GPU is silent for 4.6 of the 5.3 s (three short bursts).
- Only 5 main-model verify forwards ran (80 mma attention calls / 16
  layers, 160 dequants): 24.3 ms of attention each (dequant 14.2 + mma
  10.1), matching the filler measurement. About 93% of the 256 tokens
  came from the CPU ngram-map path with no GPU work at all.

The verify-step attention cost (~25 ms) is real, but at ~1 verify per
second it contributes ~25 ms of every second of wall time, 0.5% of the
felt decode speed. The GQA-packed vec kernel (the remaining lever, ~25
ms to ~12 ms per verify step) would save ~13 ms per second, well under
1%. It is not worth the kernel redesign. The felt long-context decode
speed on this stack is set by the CPU ngram-map speculator and the
server loop, not by the GPU attention path.

Verdict: keep the mma-f16 + dequant verify path. No code change. If
long-context decode still feels slow, the target is the CPU spec
machinery in the fork, outside the scope of this spec.
