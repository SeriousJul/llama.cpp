# 003: FP8 block-scale GEMM for sm89 (prefill)

Status: done (2026-09-23, verdict: no change; premise falsified by
measurement, no code written)

## Problem (as originally framed)

Prefill of quantized weights was assumed to dequant to F16 and run
cuBLAS FP16 GEMM. Ada FP8 MMA (m16n8k32) is 2x the FP16 rate, so a
block-scale FP8 weight path could speed prefill. GEMM was measured
(001) at ~57% of P2 prefill.

## Why the premise was wrong

Checked the actual dispatch on this tree (build 11146):

- `ggml_cuda_should_use_mmq()` returns true for every MMQ-supported
  type on Turing-and-newer, at any batch size (`turing_mma_available`
  early-return in mmq.cu). F16 cuBLAS is only the fallback for types
  MMQ does not implement, or when smem < 48 KiB.
- IQ4_XS on Ada uses `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0` with stream_k
  (mmq-config-ampere.cuh): the GEMM already runs on INT8 tensor cores
  (mma.sync m16n8k32 s8), not FP16.
- MXFP4 and NVFP4 already exist as ggml types (39/40) with MMQ config
  entries on the Ampere/Ada table (`SRAM_LAYOUT_Q8_1`). Native FP4
  block-scale MMA is gated to Blackwell (`blackwell_mma_available`).

Ada dense tensor-core rates depend on the accumulator width, and
GGML needs fp32 accumulation for f32 output: FP16 (fp32 acc) 165
TFLOPS, FP8 e4m3 (fp32 acc) 330, INT8 660 TOPS. (The 661/1321-style
numbers in vendor sheets are fp16-accum and/or sparse.) The current
MMQ path therefore sits on the fastest 8-bit TC lane the hardware
has, and an FP8 kernel would run at half that rate. There is no
tensor-core headroom in any direction from INT8-with-fp32-accum;
only dequant ALU savings, which measurement below shows are not the
limiter.

## Measured (2026-09-23, dev model Qwen3.5-9B)

No new code: built a dense MXFP4 file from the existing tree with
`llama-quantize --allow-requantize --tensor-type-file` (all attn/ffn/
ssm_out 2D weights -> mxfp4, rest q8_0). Source: Q8_0 9B (the
--tensor-type override path needs no BF16 for a first-pass quality
call; requant delta is bounded by 8-bit source error). Files in
~/bench-fp8.

| metric (9B, ubatch 512, CUDA) | IQ4_XS | MXFP4 | delta |
|---|---|---|---|
| pp1024 t/s | 10574 | 10168 | -3.8% |
| pp2048 t/s | 10695 | 10407 | -2.7% |
| pp4096 t/s | 10660 | 10326 | -3.1% |
| tg128 t/s  | 152.3 | 147.7 | -3.0% |
| wikitext-2 PPL | 8.169 ± 0.055 | 8.921 ± 0.063 | +9.2% |
| file size | 4.80 GiB | 5.44 GiB | +13% |

Reference: Q8_0 9B PPL 8.196 ± 0.056 - IQ4_XS is already at the model
quality ceiling; MXFP4 is 9 sigma below it. The E2M1 grid (15 distinct
magnitudes per block) is the problem: at 4 bits it is strictly cruder
than the IQ4_XS codebook, and the block-32 e8m0 scale cannot recover
mantissa precision.

The 27B case is worse on size: e4m3 (true FP8) weights at 8.06 bpw
need 28 GB - the model does not fit 24 GB with any KV. W8A8 FP8 was
never an option for this model.

## Verdict

- No speed headroom exists on Ada from FP8/FP4 tensor cores: prefill
  GEMM already runs at the INT8 TC rate, which equals the FP8 rate.
- No new ggml type, quantizer, or kernel is justified.
- The existing MXFP4 path loses on speed, quality, and size against
  the IQ4_XS baseline. Nothing to adopt.
- The original 003 design (block-scale FP8 weights + Ada kernel) is
  closed without implementation. Cost of the finding: one download,
  one quantize run, one bench matrix, one perplexity pass.

## What would change this

- Blackwell-class hardware (native FP4 block-scale MMA at 2x FP8, and
  NVFP4/MXFP4 files that fit with quality via scaled microscaling
  recipes + imatrix) - a different spec on different hardware.
- A future 4-bit format with FP8-TC compatibility that beats IQ4_XS at
  equal bits on both quality and prefill speed. None exists today.
