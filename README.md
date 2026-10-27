# FAIE2-FlashNext (FAIE2-FN)

A from-scratch CUDA inference engine for **Qwen3.8-Flash-Next** (a DeltaNet-hybrid MoE), running split across two RTX 5090s on Windows. No llama.cpp, no PyTorch inference path: hand-written CUDA kernels, a ctypes host layer, and a correctness regime in which every optimization must prove it changed nothing.

**172.9 tok/s greedy decode** (159 tokens in 920 ms; 5.70 ms per forward), independently re-measured 2026-10-27 — see `VERDICT.md`. Part of the FAIE2 family; the Qwen3.8-27B engine, server and tool suite live in the sibling [`faie2`](../faie2) repository.

**Status: active work in progress.** This is a dated snapshot (2026-10-27) of a live project, published under ABUZ8's post-everything policy: results with their evidence, unfinished parts stated as unfinished.

## Measured progression (author's benches, DESIGN.md §9)

| Stage | Decode tok/s | ms/token |
|---|---|---|
| Split-K GEMV baseline | 75.1 | 13.3 |
| Tiled GEMV | 137.4 | 7.28 |
| Kernel fusion (PDL, early GEMVs, fused hyper-connections, single-kernel DeltaNet core) | 173.6 | 5.76 |
| 8-row routed-expert kernel (`k_tmoe_gu8`, current build) | 169.5–173.7 across runs | ~5.70–5.76 |

Kernels per token: 1502 → 1071. fp32 math throughout; no reduction order changes. Output is **bit-identical** across every step (logits at 54 positions + residual stream after every layer, compared build-over-build with `flashnext/bitcmp_fn.py`).

## Correctness gates (independently re-run 2026-10-27, all PASS)

| Gate | Checks | Result |
|---|---|---|
| QUANT (`test_fn_quants.py`) | All 13 GGUF quant formats dequantize on GPU exactly like gguf-py; GEMV vs float64 math | dequant error **0.00**; GEMV rel err ≤ 1.8e-07 |
| ORACLE (`gate_fn_ref.py`) | Full forward vs the official transformers implementation in float64 | top-1 **21/21**; max log-prob diff **0.0000**; layer-0 residual 1.02e-05 |

Raw verification logs: `verification/`. Full verdict: `VERDICT.md`. Complete engineering record (kernel-by-kernel measurements, dead ends included): `DESIGN.md` §9a–9c.

## Layout

- `engine/` — `faie2fn.cu` (the engine), `fn_tables.cuh`, `build_fn.ps1` (CUDA 13.1 + MSVC, sm_120).
- `flashnext/` — ctypes host wrapper (`fn.py`), the float64 oracle (`ref_fn.py`), gates, bench and profiling tools.
- `verification/` — the independent verifier's raw test logs for this snapshot.

## Not included

Model weights (the engine loads local GGUF shards) and the compiled `faie2fn.dll` — build from source.

## Not built yet (stated plainly)

Speculative decoding for this engine (it exists, lossless, in the `faie2` reference line), batched prefill, a serving load test, and the MTP drafter head (not downloaded). Tensor-core decode for IQ2/IQ3 is the largest remaining speedup and is pending a deliberate decision, because it changes the fp32-math rule. GPU↔GPU peer access is blocked by the closed Windows WDDM driver (peer access 0 both ways, topology PXB); the host-staged hand-off costs ~5 µs of a 5.70 ms token. The 3.5–4 ms/token target is not yet reached.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Ahmad A Odeh / ABUZ8 LLC.
