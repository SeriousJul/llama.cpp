# 012: Return prefill-phase bytes to the decode KV budget

Status: done (2026-09-26, verdict: no change. The phase-idle bytes are 16-22 MiB, not
233 MiB and not the fork's 2304 MiB; see Results)
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

## Results (measured 2026-09-26, windows 1-4)

The baseline table above freezes the split between the two things the band could be, and
finding 3 was left open because the instrument it assumed was missing did not exist in the
form it looked for: graph capture *does* have a runtime off switch, `GGML_CUDA_DISABLE_GRAPHS`
(read in `ggml_cuda_graph::is_enabled()`, `ggml/src/ggml-cuda/common.cuh:1288`, consulted at
`ggml-cuda.cu:4489` and `:4508`). So the graph share was measured two ways on one library:
the A/B on that switch, and a `cudaMemGetInfo` probe around capture and instantiate
(`specs/artifacts/012-graph-probe.patch`, built as `build-012`; windows 1-3 ran on
libggml-cuda md5 `8adde1895990`, window 4 on `3e4b4a2e137e`, which is the same probe plus the
capture key and node count).

Window 1 (`specs/artifacts/012-window1.sh`, `w1-*`): 27B IQ4_XS, ctx 160000, q8_0 KV, `-fa on`,
`-fit off`, one 1331-token prompt (`011-sweep/012-prompt.txt`) + n decode tokens:

| row | unaccounted | used_peak | graphs instantiated | tg t/s |
|---|---|---|---|---|
| b2048 ub512 graphs ON, n=8 | 1014 | 20045 | 1 | 51.88 |
| b2048 ub512 graphs OFF, n=8 | 998 | 20029 | 0 | 50.60 |
| b2048 ub512 graphs OFF, n=1024 | 998 | 20049 | 0 | 51.17 |
| b2048 ub512 graphs ON, n=1024 | 1014 | 20065 | 1 | 54.78 |
| b512 ub512 graphs ON, n=8 | 1014 | 20045 | 1 | 52.04 |
| b512 ub512 graphs OFF, n=8 | 998 | 20037 | 0 | 50.79 |
| b512 ub512 graphs OFF, n=1024 | 998 | 20049 | 0 | 51.15 |
| b512 ub512 graphs ON, n=1024 | 1014 | 20065 | 1 | 54.72 |
| b2048 ctx 262144 ON, n=8 | 1014 | 23935 | 1 | 51.83 |
| b2048 ctx 262144 OFF, n=8 | 998 | 23927 | 0 | 50.53 |

1. The graph share of the band is **16 MiB**, in every row, and it is the same 16 MiB in the
   `nvidia-smi` peak (20065 vs 20049). It does not depend on the batch shape, the ubatch
   shape or the context size, because there is always exactly **one** instantiated graph.
2. Prefill and decode do not hold separate graphs, so there is no pp graph to evict at the
   phase transition. Window 4 (`012-window4.sh`, log key + node count at each capture) shows
   prefill *does* get captured, and decode reuses the same slot:

   | row | captures | keys |
   |---|---|---|
   | 1331-token pp + 1 decode step | 1 | one key, captured inside pp (2nd identical 512-token ubatch completes `warmup_complete`) |
   | 1-token pp + 8 decode steps | 1 | one key |
   | 1331-token pp + 64 decode steps | 1 instantiate, 2 captures | `0x781057503980` at 0.01.887 (pp) and again at 0.02.485 (tg) |

   The slot is looked up by `ggml_cuda_graph_get_key(cgraph) = cgraph->nodes[0]`, and in the
   third row the pp capture and the tg capture print the same pointer, so the two phases share
   one `ggml_cuda_graph`: the phase whose shape changed re-captures in place (2.8 ms, 0.0 MiB)
   instead of adding a second graph. Design option 1 ("if a pp graph and a tg
   graph coexist and the pp graph is not replayed, release it") has no coexisting pair to act
   on, and the 16 MiB belongs to whichever phase is running, which after the transition is
   decode - the phase a trim must not tax.
3. Capture itself is free: pre-capture and post-capture `cudaMemGetInfo` agree to 0.0 MiB in
   all 5 ON rows (e.g. free 4057.3 -> 4057.3). The 16 MiB appears at `cudaGraphInstantiate`
   (4057.3 -> 4041.3), i.e. it is driver graph-exec overhead, not pool bytes pinned to a
   graph. Option 3 (unmap physical handles under a reserved virtual range) has nothing of
   ours to unmap.
4. What the graphs buy: +7.1% decode (54.75 avg vs 51.16 avg at n=1024; +6.9% b512, +7.1%
   b2048) at no pp cost (2768-2813 t/s either side). A trim that dropped the decode graph to
   reclaim 16 MiB (~500 tokens of q8_0 KV) would cost 7% of decode t/s. Bad trade by two
   orders of magnitude.
5. Side observation, unexplained and not 012's question: during the n=1024 rows the decode
   graph is *re-captured* about every 4.7 s (probe `pre capture`/`post capture` pairs at
   0.02.488, 0.06.225, 0.10.898, 0.15.569, 0.20.244 with `captures` staying 1). Each costs
   2.2 ms and 0 MiB, so 0.05% of the run.

Window 2 (`012-window2.sh`, `w2-*`, graphs off in all rows) isolates what is left of the band:

| row | self (model+ctx+compute) | unaccounted |
|---|---|---|
| idle, ctx 160000, no request at all | 19050 = 12726 + 5462 + 861 | 976 |
| idle, ctx 8192 | 13278 = 12726 + 421 + 130 | 976 |
| `-ngl 0` (nothing offloaded) | 1231 | 931 |
| one 1331-token pp + 1 token | 19050 | 996 |

The band is a fixed per-process init cost, not a function of how much was allocated: 976 MiB
at 19.0 GiB of buffers and 976 MiB at 13.3 GiB, 931 MiB with nothing offloaded. A full
prefill adds ~20 MiB on top of the idle floor (996 vs 976). So neither "VMM pool spares" nor
"freed pp cache" is hiding hundreds of MiB either.

`specs/artifacts/012-cuda-floor.cu` (device-wide NVML readings, run alone on the GPU) names
the fixed part: the CUDA context is **391 MiB**, a cuBLAS handle costs 12 MiB for the first
and ~8 MiB each thereafter (8 handles = 68 MiB; the context owns one handle per stream,
`cublas_handles[device][GGML_CUDA_MAX_STREAMS]` at `common.cuh:1452`, `GGML_CUDA_MAX_STREAMS = 8`),
and an instantiated 300-node graph costs 2.7 MiB. It also falsifies the
worst-case objection to option 2: `cudaMalloc` 12 GiB then `cudaFree` returns **0.0 MiB
sticky**, so a deliberate pool flush does hand bytes back to `cudaMemGetInfo`. The mechanism
works; there is simply nothing large enough to flush.

Window 3 (`012-window3.sh`, `w3-*`, production shape: dflash draft, `--spec-draft-n-max 7`,
verify mean length 3.23, ABBA) checks the multi-shape case that a single-graph session cannot:

| row | unaccounted | graphs instantiated | tg t/s |
|---|---|---|---|
| prod graphs ON | 2655 | 3 | 107.39 |
| prod graphs OFF | 2633 | 0 | 101.21 |
| prod graphs OFF (2nd) | 2633 | 0 | 101.32 |
| prod graphs ON (2nd) | 2655 | 3 | 107.95 |

Even with spec decode, where several batch shapes do recur, the whole graph set costs 22 MiB
(3 instantiated graphs, 4 + 18 + 0 MiB, and 0 MiB at every capture), and buys +6.2% decode.
Note also that this row's band is 1657 MiB larger than the
no-draft band (2633 vs 976), and 1080 MiB of that is the draft model's own buffer, which the
`memory breakdown` printout does not list under `model` (`load_tensors` reports `CUDA0 model
buffer size = 1079.61 MiB` for the draft, the table prints `model 12726`). Part of what a
future spec calls "unaccounted" with a draft loaded is a real, active, unattributed buffer.

### Verdict

Closed as no change. Reclaimable at the pp-to-decode transition, measured: 16 MiB (the one
graph slot, and after the transition it belongs to decode) + ~20 MiB (pool cache left by a
full prefill) + 233 MiB of pp-shape compute buffer that is reserved at init and never becomes
idle (finding 1, unchanged). The realistic ceiling is therefore ~36 MiB, ~1.1K tokens at
q8_0, against the fork's ~2304 MiB headline. No opt-in trim, no recapture hysteresis, no
arena.

What survives for 011/013:

- The `-fit` objection in finding 3 is now moot for graphs: there is no phase-varying graph
  footprint to make the estimator optimistic.
- The pool flush is mechanically safe (0 MiB sticky after a 12 GiB round trip), so 013 can
  use "free the idle cache, then `cudaMalloc` the KV" without a VMM unmap layer.
- The band is a fixed ~976 MiB init floor: 391 MiB CUDA context, tens of MiB of per-stream
  cuBLAS handles (68 MiB for 8 in the standalone probe), the rest driver-side. Any future
  "how much KV can I add" arithmetic on this box should treat it as a constant, not as slack.
- With a draft model loaded, `common_memory_breakdown_print` under-reports `model` by the
  draft's buffers. Say so before quoting `unaccounted` as reclaimable.

Of the acceptance criteria: the baseline table was frozen before any patch (met); the ctx
ceiling criterion was not met and cannot be, because the measurement that criterion needed
said the bytes are not there; the inactive-path and resume-cost criteria were not reached,
since no patch was written. The probes are archived (`012-graph-probe.patch`,
`012-cuda-floor.cu`, `012-window{1,2,3,4}.sh`) with the readings under `012-sweep/`:
`w*-key.txt` carries the `memory breakdown`, buffer-size, timing and `graph-probe` lines every
table here quotes, plus the per-row `.probe` and `nvidia-smi` `.sm` samples. The full `-v`
server logs were pruned (14 MB of them); re-run a window script to get them back.

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
