# 012: Return prefill-phase bytes to the decode KV budget

Status: active (measured 2026-09-26; verdict: the reclaimable pp share is small,
see Problem Statement)
Label: research
Depends on: 001, 011 (shares the "bytes that exist but are idle during decode" question)
Scope: `ggml/src/ggml-cuda/ggml-cuda.cu` graph and pool lifecycle; possibly
`src/llama-context.cpp` graph reserve. No kernel work.

## Origin

Same fork analysis as 011. The fork's second idea ("phase arena", its commits `b086b04c`
through `11f8847f`) is orthogonal to streaming and simpler than it looks: prefill and
decode do not need their peak compute workspaces at the same time, so the bytes the
prefill CUDA graph and the pp-sized scheduler workspace hold should be spendable on KV
once decode starts. The fork reports that this makes the usable decode KV pool nearly
constant across `--ctx-size` settings.

We cannot reuse its code: it bypasses `-fit` and owns the whole allocation. We can
adopt the accounting idea inside our own allocator.

## Problem Statement (measured 2026-09-26)

27B IQ4_XS, ctx 160000, q8_0 KV, `-fa on`, single slot, no draft, from the
`-fit off` reference server runs (`specs/artifacts/011-sweep/012d-*.log`,
`012-server.log`):

| shape | CUDA0 self | model | context | compute | free | unaccounted |
|---|---|---|---|---|---|---|
| b2048 ub512 | 19050 MiB | 12726 | 5462 | **861.53** | ~4040 | 992-994 |
| b512 ub512 | 19050 MiB | 12726 | 5462 | **861.53** | 4109 | ~924 |
| b8 ub8 | - | - | - | **628.96** | - | - |

Findings:

1. The compute buffer is **insensitive to the logical pp batch** (b 512 vs 2048:
   identical 861.53 MiB, reserved worst case at context init) and sensitive only
   to the ubatch shape: the pp-shape workspace share is 861.53 - 628.96 =
   **232.6 MiB**, about 7K tokens of q8_0 KV. `nvidia-smi` during a live
   pp -> 2048-token decode transition moved 12 MiB (20033 -> 20045): the bytes
   are all reserved up front and none of them become "idle then reclaimable" in
   any amount that matters.
2. The fork's own arena numbers are consistent with this once you read them
   honestly: its headline is a ~2304 MiB arena, but the *ctx-size invariance* it
   advertises is bought by streaming (013), not by phase reuse alone. On our
   machine phase reuse of scheduler workspace is worth ~233 MiB.
3. What is not isolated: the ~924-994 MiB "unaccounted" band (CUDA graph
   captures, cuBLAS workspaces, VMM pool spares). There is no user-facing switch
   to disable CUDA graph capture in this tree (`use_cuda_graph` is set per
   context inside `ggml-cuda.cu:2558-4647`), and it is not constant across the
   two runs above, so splitting graph bytes from pool spares needs instrumented
   builds; until then 012's realistic ceiling is 233 MiB + an unknown graph
   share, not the fork's full arena.

Verdict: as a standalone spec, 233 MiB (~7K tokens at q8_0) does not justify an
opt-in trim subsystem with recapture jank. 012's idea survives only folded into
013's design (where the phase transition is already a real thing with real bytes
attached) or closed. The `-fit` accounting objection stands either way.

## Design sketch (freezes after baseline)

Candidate mechanisms, cheapest first:

1. **Graph eviction policy.** CUDA graph caching lives in `ggml_cuda_graph`
   (`ggml-cuda.cu:4514` capture start, `:4417` capture end). If a pp graph and a tg
   graph coexist and the pp graph is not replayed for N decode steps, release it
   (its captured allocations come from the pool, so freeing returns pool memory).
   This is a policy change, not a new subsystem: the graph is re-captured on the next
   pp burst, costing one capture.
2. **Pool trim between phases.** `ggml_cuda_pool_leg` / `ggml_cuda_pool_vmm`
   (`ggml-cuda.cu:413`, `:530`) keep freed blocks cached. The flush exists but only
   fires after an alloc failure (`ggml-cuda.cu:493-499`); firing the same flush
   deliberately at the pp-to-decode transition converts idle cache into `cudaMalloc`
   headroom for KV.
3. **VMM unmap, not free.** With the vmm pool (`:530-586`), physical handles can be
   unmapped while the virtual range stays reserved. That is the faithful form of a
   phase arena: same pointers, borrowed physical pages. Highest complexity; only if
   options 1 and 2 show the bytes are real but fragmentation blocks the return.

All three must be strictly opt-in under memory pressure (a threshold, or `-fit`
knowing the KV budget), never a default reshuffle, so short sessions do not pay
recapture costs.

## Acceptance Criteria

- Baseline table (steps 1-3) in this file before any patch.
- Measurable ctx ceiling gain, stated in tokens, on the 27B: e.g. max stable ctx at
  q8_0 KV rises by X K tokens at unchanged decode t/s (paired ABBA).
- No regression when the feature is inactive: production preset with everything
  fitting in VRAM must be within 1% on pp and tg, since no trim should ever fire
  there. If it fires, the pressure threshold is wrong.
- One-token-per-request resume cost bounded: after a trim, the first pp of the next
  request pays at most one graph capture; record the number in the log and put a
  ceiling in the file.

## Rollback and Risk

- Graph recapture per request-turn could make interactive use janky (every resumed
  prompt pays a capture). Mitigation: hysteresis, only trim when free VRAM is below
  the KV need, not on every decode transition.
- `cudaMalloc` after trim can fragment; the arena-style single allocation (option 3)
  is the fallback, not the entry.
- `-fit` (common/fit.cpp) models steady state only. Any behavior where memory
  changes with phase makes the estimator wrong in the optimistic direction unless fit
  is told to reserve the post-trim numbers. The fork simply bypassed fit; we cannot.
- Interaction with sleep/unload in the production router (`presets.ini`,
  `--sleep-idle-seconds`): phase trims during sleep must not race the unload path.

## Out of Scope

- Any per-layer KV residency management (013).
- Host staging of KV (013), UVM hints (011).
- Multi-GPU and pipeline-parallel accounting.
- Upstreaming anything that bypasses `-fit`.
