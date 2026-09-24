# 001: Baseline measurements and kernel breakdown

Status: done. Matrix of 2026-09-23 recorded below. The D1L/D2L
`-pg` rows of 2026-09-23 (14:30) close the last gap: the true 131k
decode step is measured in the nd2l nsys capture.

## Goal

Record the pre-optimization numbers for the 27B target model on the 4090,
and get a per-kernel time breakdown for one prefill and one long-context
decode step. No code changes. Every later spec cites the numbers from this
file in its acceptance criteria.

## Setup

- The script reads slot state from the top-level
  `GET /slots?model=qwen3.8-27b-dflash`, which proxies to the model child
  server. This endpoint is enabled by default (`--no-slots` disables it),
  so no server change is needed.
- Optional: install Nsight Systems CLI (`nsys`) for the profile captures.
  If absent, the script skips the profile steps.

## Matrix

All runs: q8_0 KV (K and V), flash-attn on, ngl all, batch 2048, ubatch 512.
The script unloads the live model first and reloads it after.

| id | test | prefill | decode ctx | method |
|---|---|---|---|---|
| P1 | prefill small | 4096 | - | llama-bench |
| P2 | prefill large | 32768 | - | llama-bench |
| D1 | decode mid ctx | 1 | 32768 | llama-bench |
| D2 | decode long ctx | 1 | 131072 | llama-bench |
| D1L | decode mid ctx | 32767,1 | 32768 | llama-bench `-pg` |
| D2L | decode long ctx | 131071,1 | 131072 | llama-bench `-pg` |

`-p X -n 1` emits two tests: `ppX` (prompt only) and `tg1` (1 token
generated in a fresh near-empty context, `n_prompt = 0`). Only `-pg X,1`
(`ppX+tg1`) generates at the long context. D1L/D2L use `-pg`.

| E1 | production mid ctx | ~65536 | 256 gen | server e2e (dflash + ngram) |
| E2 | production long ctx | ~131072 | 256 gen | server e2e (dflash + ngram) |

E1 and E2 use distinct prompt prefixes so the prompt cache cannot reuse work
between them. Their numbers include the spec decode stack and are the
acceptance reference for decode specs.

## Commands

```sh
# full matrix on the 27B (default)
tools/bench/server-bench

# fast kernel A/B on the 9B dev model (llama-bench only, no e2e)
tools/bench/server-bench --model dev

# profile captures (requires nsys)
tools/bench/server-bench --profile
```

Results land in `bench-results/<timestamp>/`.

## Results

Run 2026-09-23, `bench-results/20260923-085223/`, build 2a34960d9 (11104).
All llama-bench rows: q8_0 KV, fa on, ngl all, batch 2048, ubatch 512.

| id | test | t/s | ms per token | notes |
|---|---|---|---|---|
| P1 | pp4096 | 3086.88 | 0.32 | |
| P1 | tg1 | 52.55 | 19.0 | small ctx decode floor |
| P2 | pp32768 | 2705.46 | 0.37 | |
| P2 | tg1 | 52.26 | 19.1 | small ctx decode floor |
| D1 | pp32767 | 2700.74 | 0.37 | |
| D1 | tg1 | 52.68 | 19.0 | small ctx decode floor |
| D2 | pp131071 | 1905.33 | 0.52 | |
| D2 | tg1 | 51.16 | 19.5 | small ctx decode floor |
| D1L | pp32767+tg1 | 2719.91 | 0.37 | combined t/s, decode invisible (note) |
| D2L | pp131071+tg1 | 1909.40 | 0.53 | combined t/s, decode invisible (note) |
| E1 | prompt 65585 | 2180.16 | 0.46 | eval 93.76 t/s, 10.67 ms/token |
| E2 | prompt 131088 | 1795.95 | 0.56 | eval 63.31 t/s, 15.79 ms/token |

The D1L/D2L rows (run 2026-09-23, `bench-results/20260923-143056/`) report
a combined t/s = (N+1)/(t_pp + t_tg). With N = 131071 and t_pp ~ 69 s,
the ~30 ms decode is 0.04% of the total and carries no resolution. They
confirm the prefill numbers are stable (1909.40 vs 1905.33); the decode
itself must be read from the nd2l nsys capture below.

E1/E2 details (dflash spec decode, ngram on):

- E1: draft acceptance 0.476, mean len 4.32, 256 tokens in 2719.8 ms
- E2: draft acceptance 0.426, mean len 3.88, 186 tokens in 2922.1 ms
  (the task stopped before 256; the 131k context left less generation
  budget)

Note: the `tg1` rows are the context-independent decode floor (fresh
context, 1 token). They are all ~51-53 t/s, confirming they do not scale
with the `pp` size in the same row. They are not the long-context decode
numbers the D1/D2 rows were meant to give. D1L/D2L are the corrected
measurements (pending).

E2 is the production acceptance reference for decode work: 15.79 ms/token
at 131k context with spec decode (mean 3.88 tokens per step, so about
61 ms per verify step).

### D2L (nd2l): the true 131k decode step (batch 1)

Capture: `-pg 131071,1 --no-warmup`, the first iteration's single decode
step. The capture was killed after the first iteration but the step is
fully inside it. Wall 30.0 ms, kernel sum 27.2 ms (nsys adds overhead).

| kernel | total | instances | notes |
|---|---|---|---|
| mul_mat_vec_q | 15.69 ms | 460 | weight GEMV, 13.3 GB, 800 GB/s |
| fattn vec (256,1,q8_0,q8_0) | 8.77 ms | 16 | 545 us each, grid 1x21x24 |
| rms_norm_f32 | 0.73 ms | 305 | |
| quantize_q8_1 | 0.56 ms | 461 | |
| gated_delta_net | 0.18 ms | 48 | SSM stacked |
| fattn combine_results | 0.05 ms | 16 | 3 us each, negligible |
| rest (scale, rope, conv, concat, ...) | ~1.3 ms | | |

Findings:

- The KV split IS engaged at 131k: grid 1x21x24 = 504 CTAs, each CTA
  reads ~6240 tokens of one kv-head (1.8 MB). The launch_fattn
  efficiency loop picks parallel_blocks = 21 (512 KV tiles / 21).
  The small-ctx decode (gridY = 1) is correct behavior, not a bug.
- Attention at 131k: 8.8 ms = 29% of the step. The KV floor is 4.35 ms
  (4.4 GB / 1008 GB/s), so the kernel runs at 2.0x the floor. The
  excess is the GQA redundancy: grid z = 6 q-groups x 4 kv-heads, so
  each kv-head chunk is read by 6 CTAs (903 MB of read requests per
  layer vs 150 MB unique; L2 absorbs about 40%). Per-CTA bandwidth is
  ~3.3 GB/s of the ~7.9 GB/s per-CTA share, so the 21-way split is
  already near DRAM peak once L2 reuse is counted.
- The split-KV combine pass costs 0.05 ms per step: free.
- Batch-1 decode step at 131k: 30.0 ms (29.2 t/s under nsys). The
  small-ctx floor step is 22.4 ms. Context adds 7.6 ms at 131k, of
  which 8.8 ms is attention (the SSM layers are O(1), as expected).

## Kernel breakdown

Type numbers in `mul_mat_q<T>` are ggml types: 23 = IQ4_XS, 21 = IQ3_S,
12 = Q4_K, 13 = Q5_K, 18 = IQ3_XXS, 22 = IQ2_S, 11 = Q3_K (UD mix).

### P2 (np2, pp32768; total GPU kernel time 71.4 s)

| kernel | share | total | instances | avg |
|---|---|---|---|---|
| mul_mat_q IQ4_XS | 28.9% | 20.6 s | 81024 | 255 us |
| gated_delta_net | 13.1% | 9.4 s | 18480 | 508 us |
| fattn mma-f16 (256,256,8,8) | 12.7% | 9.1 s | 6144 | 1.47 ms |
| mul_mat_q IQ3_S | 9.2% | 6.6 s | 17664 | 374 us |
| mul_mat_q Q4_K | 5.6% | 4.0 s | 21120 | 190 us |
| mul_mat_q Q5_K | 4.7% | 3.4 s | 19200 | 176 us |
| silu (unary_gated) | 3.6% | 2.5 s | 43084 | 59 us |
| mul_mat_q IQ3_XXS | 3.3% | 2.4 s | 6528 | 364 us |
| quantize_mmq_q8_1 | 2.2% | 1.5 s | 149760 | 10 us |
| mul_mat_q IQ2_S | 1.8% | 1.3 s | 2688 | 483 us |

Shares: GEMM (all mul_mat_q + quantize_mmq) ~57%, fattn 12.7%, SSM 13.1%,
elementwise (silu, concat, rms) ~8%.

### D2 (nd2, pp131071 x 6 passes + 1 small-ctx tg1; total 402 s)

| kernel | share | total | instances | avg |
|---|---|---|---|---|
| fattn mma-f16 (256,256,8,8) | 35.5% | 142.7 s | 24576 | 5.8 ms |
| mul_mat_q IQ4_XS | 20.7% | 83.1 s | 324096 | 256 us |
| gated_delta_net | 9.4% | 37.9 s | 73776 | 514 us |
| mul_mat_q IQ3_S | 6.6% | 26.6 s | 70656 | 376 us |
| mul_mat_q Q4_K | 4.0% | 16.1 s | 84480 | 191 us |
| mul_mat_q Q5_K | 3.4% | 13.6 s | 76800 | 177 us |
| dequantize q8_0 -> f16 | 2.8% | 11.2 s | 49152 | 229 us |
| silu (unary_gated) | 2.5% | 10.2 s | 172108 | 59 us |
| mul_mat_q IQ3_XXS | 2.4% | 9.6 s | 26112 | 367 us |
| quantize_mmq_q8_1 | 1.5% | 6.1 s | 599040 | 10 us |

Instance counts confirm the structure: 24576 = 6 passes x 256 ubatches
x 16 attention layers; 49152 dequants = 2 (K and V) per layer per
ubatch. Prefill attention always runs the mma-f16 kernel, which needs
the full layer KV dequantized to f16 first (273 MB -> 546 MB per layer
at full context). At the prefill tail, per layer per ubatch: 2 x 447 us
dequant + 11.5 ms mma fattn. mma fattn at 546 MB / 11.5 ms achieves
about 47 GB/s aggregate; the stream-K split is on (256 CTAs), so the
low bandwidth is a read-pattern/L2 problem, not a CTA-count problem.

### Decode step in the D2 capture (small context, tg1)

Wall 22.4 ms, kernel sum ~20.5 ms (well packed). This is the context-
independent decode floor, not a 131k decode:

| kernel | total | instances | notes |
|---|---|---|---|
| mul_mat_vec_q | 16.7 ms | 460 | weight GEMV, 13.3 GB read, ~800 GB/s |
| rms_norm_f32 | 0.71 ms | 304 | |
| quantize_q8_1 | 0.54 ms | 460 | activation quant for MMQ |
| scale_f32 | 0.40 ms | 190 | |
| gated_delta_net | 0.17 ms | 48 | SSM stacked |
| fattn vec (256,1,q8_0,q8_0) | 0.17 ms | 16 | ~10 us each, grid 1x1x24 |

The vec fattn at ~10 us per layer is consistent with a near-empty KV
(1-2 rows), not 131072. The GEMV floor is 13.3 GB / 1008 GB/s = 13.2 ms;
measured 16.7 ms is 79% of peak. A 131k decode step adds the full-KV
attention cost on top of this floor; that is what D2L must measure.

## Acceptance

- All six matrix rows filled: done (2026-09-23 runs)
- Breakdown table filled for P2 and D2: done (prefill + small-ctx decode)
- Numbers copied into the acceptance sections of 002 and 003: done
- D2L true 131k decode step recorded (nd2l): done 2026-09-23

Status: complete.

## Risks

- D2 needs ~20 GB VRAM (15.5 GB weights + 4.4 GB KV + buffers). Fits, but
  nothing else may hold VRAM, hence the unload step.
- E1/E2 each take several minutes (large cold prefills plus 256 generated
  tokens). Plan a bench window of ~20-30 min total.
- The D1L/D2L rows re-run the full prefill per iteration (llama-bench has
  no prompt cache), so one `-pg 131071,1` row costs about 70 s per
  iteration plus the decode.
- The combined t/s of a `ppN+tg1` row cannot resolve the decode: the
  prefill dominates the denominator. Use the nsys capture (nd2l) for the
  decode step; the row only re-confirms the prefill number.
