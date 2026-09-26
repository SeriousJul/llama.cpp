# 013: Pipelined block-granular KV streaming for contexts beyond VRAM

Status: deferred (gated on 011 and 012 results; needs a need-analysis baseline)
Label: research
Depends on: 011 (UVM hints, the competing simple answer), 012 (free-VRAM ceiling this
extends), 001, and the `-nkvo` scheduler path mapped in the session of 2026-10.
Scope: `src/llama-kv-cache.cpp`, `ggml/src/ggml-backend.cpp` (sched copy ring),
`ggml/src/ggml-cuda/`. New subsystem; do not start without design sign-off.

## Origin

`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`. Its full design: authoritative KV
in pinned host RAM, a bounded device pool split into resident pages plus one shared
transfer ring, prefetch of layer `il+1`'s staged span while layer `il` computes, and a
resident/ring partition that adapts to active context length. Attention stays exact.

Direct adoption is rejected: CUDA-only, single server slot, `fattn.cu` enlarged from
31 KB to 120 KB with model-specific staged-KV paths, and `-fit` bypassed. But our tree
already contains the same skeleton, discovered by tracing `-nkvo`:

- Placement switch: `src/llama-kv-cache.cpp:214-224` (`offload` picks `dev_layer(il)`
  buft vs CPU buft for every cache tensor).
- FA no longer pinned to CPU by `#19105`/`#19165`; the scheduler decides per node.
- Op claim: `ggml/src/ggml-backend.cpp:969-976` (cause `"1.off"`), gated by
  `ggml_backend_offload_op` (`ggml-cuda.cu:5662-5665`, env
  `GGML_OP_OFFLOAD_MIN_BATCH`, default 32).
- Copy insertion and execution: `ggml-backend.cpp:1389/1411` (`"4.cpy"`), ring of copy
  slots with events at `:1675-1792`, async via `cpy_tensor_async`.
- Store back: `ggml_set_rows` (`llama-kv-cache.cpp:1350/1385/1406`).

## What upstream does today vs what streaming needs

Upstream, `-nkvo` + FA stages the **whole active KV view** per ubatch through the copy
ring, synchronously per split: copy for layer `il` is issued when the scheduler reaches
layer `il`. There is no lookahead, no partial residency, no adaptation. Decode at long
ctx therefore pays one full KV H2D burst per layer per ubatch, serialized against the
compute it feeds.

The fork's delta over that, in mechanism terms:

| axis | upstream `-nkvo` path | fork |
|---|---|---|
| granularity | whole view, per ubatch | fixed KV blocks in a ring |
| overlap | copy issued at node order | prefetch layer il+1 under layer il |
| residency | all or nothing | adaptive resident set + streamed set |
| phase memory | untouched | arena borrows pp workspace |

## Gate zero: is this needed on this machine (computed 2026-09-26)

From the 012/011 measurements (`specs/artifacts/011-sweep/`):

- Steady state at ctx 160K, q8_0, no draft: CUDA0 self 19050 MiB, free ~4100 MiB.
- q8_0 `ctx_max` (cudaMalloc) ~285K tokens: verified live, OFF q8_0 c262144 with a
  209K-token prompt runs at pp 1611.8 / tg 31.1 t/s.
- f16 `ctx_max` ~150K tokens: verified OOM at 160K (`compute pp buffers` refused at
  P1) and at 262K (KV 16 GiB refused). Beyond that, stock UVM holds it at 0.6 t/s
  (see 011); streaming is the only alternative to that number.
- Production shape adds the dflash draft (~1 GB + draft KV) and `--models-max 1`
  reload headroom, putting the practical ceiling near the preset's 160K.

Verdict: for q8_0 KV at the production 160K preset this machine does not need
streaming; single-model runs fit with room. 013 earns its keep only for (a) f16
KV beyond ~150K, (b) q8_0 beyond ~280K, (c) co-locating a second model, or (d)
the draft + router + long-ctx combination if the 262K preset work ever lands.
Those are product decisions, not kernel ones; the spec stays deferred and the
numbers above are its re-open trigger.

## Design if ever started

Minimum viable form on top of existing infrastructure, ordered by preference:

1. **Lookahead in the sched copy ring.** While executing the split for node range
   `[a, b]` on a backend, issue the pending `"4.cpy"` copies for the next split's KV
   inputs one layer early. This is a reordering of the existing ring (`cur_copy`
   slots + events), not a new pool. Gets the overlap, keeps everything else.
2. **Partial residency in the cache buffer type.** A KV buffer that is device-resident
   for the newest R cells and host-backed for the rest, selected per layer inside the
   `llama_kv_cache` placement branch (`:214`), with streamed reads on path 1. This is
   where a second, larger copy-ring depth is justified.
3. Only after 1 and 2: adaptive resident/ring sizing. Explicitly out: custom staged-KV
   attention kernels. Upstream FA must be able to read the staged copy exactly as it
   reads a `"4.cpy"` result today; if it cannot, the change belongs in `fattn` input
   handling, discussed separately.

## Acceptance criteria (frozen at design, not now)

- Correctness: sampled tokens and perplexity bit-comparable to non-streaming runs of
  the same prompt at the same ctx, no drift; the KV store path (`set_rows`) exercised
  through context shift and slot reuse, not just a fresh prompt.
- Performance: decode t/s at ctx beyond `ctx_max` beats (a) `-nkvo` CPU attention and
  (b) hinted UVM from 011, paired ABBA, reported against the PCIe roof so the reader
  can see the efficiency, the same frame the fork's plots use.
- Multi-slot: any design that silently corrupts KV under `-np 2` is rejected at
  review; that is the fork's open weakness and ours would inherit it.
- `-fit` must model the streamed mode honestly; bypassing fit is not allowed here.

## Out of Scope

- Query-relevance eviction (approximate attention). Different product; see the KVMem
  project's approach, not adopted by the fork either.
- Weight streaming (a separate axis the fork leaves to UVM).
- Non-CUDA backends.
