# FAIE2: a Qwen3.8-27B decode engine for 2× RTX 5090

**Goal:** more than 200 tok/s single-stream decode on Qwen3.8-27B, with output that matches plain decoding.
**Status 2026-10-25:** design and baselines only. No FAIE2 code yet. Every number below was measured on this machine and has a proof file in `bench/runs/`, or is cited from its source.

---

## 1. Measured baselines on this machine (2026-10-25)

Setup: llama.cpp b10566 (CUDA 13), `Qwen3.8-27B-ABLITERATED-Q6_K.gguf` (20.9 GB), **GPU0 only**, flash-attn on, f16 KV.
Prompts are in `bench/bench.py`. Each run is greedy (temperature 0), thinking off, 512 max tokens, 2 repeats.
Decode tok/s is measured on the client, from the first streamed token to the last.

| Config | code | structured | translation | prose | mean | proof |
|---|---:|---:|---:|---:|---:|---|
| Plain (no speculation) | 59.4 | 59.3 | 60.3 | 59.4 | **59.6** | `runs/q6k_plain.json` |
| MTP, 3 draft tokens | 158.1 (acc 95%) | 156.0 (94%) | 140.4 (88%) | 101.3 (48%) | **138.9** | `runs/q6k_mtp3.json` |
| MTP, 5 draft tokens | 176.6 (84%) | 185.5 (87%) | 144.8 (69%) | 93.6 (34%) | **150.1** | `runs/q6k_mtp5.json` |

**Correctness check (the FAIE lesson):**
- MTP-3 greedy output was byte-identical to plain decoding for code, structured and translation.
- Prose diverged at character 682 ("The paper was damp" vs "The paper was dry"). Both texts are coherent, and each mode is deterministic from run to run.
- The cause is that llama.cpp's batched verify kernel rounds differently from its single-token kernel, so a near-tie can flip.
- MTP-5 also diverged on code. FAIE2 must close this gap (see §5).

**Prose is the hard case.** On code and structured output, more draft tokens help. On prose they hurt: 5 drafts ran slower than 3. A fixed draft length is wrong. The draft length has to adapt.

### DFlash2 (added 2026-10-25, same day)

Setup:
- **Binary:** the local llama.cpp DFlash2 branch build, `E:\ABU\MODELS\runtime\src\llama.cpp-dflash2\build\bin` (build 10510, commit `f5a7ec15d`). Build b10566 is too old: it reads 58 of the drafter's 81 tensors.
- **Drafter:** `ggml-org/Qwen3.8-27B-GGUF/dflash-Qwen3.8-27B-Q8_0.gguf`, 2.06 GB. Its SHA-256 `ef6a1340…b297` matches Hugging Face's published hash.
- **Target:** the same Q6_K.
- **Isolation:** GPU0 only, enforced with `CUDA_VISIBLE_DEVICES=<GPU0 uuid>`.

| Config | code | structured | translation | prose | mean | proof |
|---|---:|---:|---:|---:|---:|---|
| Plain, KV q8_0, ub 256 (reference) | 59.2 | 59.1 | 57.1 | 58.8 | 58.5 | `runs/q6k_plain_q8kv.json` |
| DFlash2 k=7, **VRAM 97% full** | 120.4 | 192.9 | 163.2 | 96.7 | 143.3 | `runs/q6k_dflash2_k7.json` |
| **DFlash2 k=7, KV q8_0, ub 256** | **213.2** | **220.9** | 145.6 | 95.0 | **168.7** | `runs/q6k_dflash2_k7_lowmem.json` |
| DFlash2 k=3, same settings | — | — | 146.1 | 100.4 | — | `runs/q6k_dflash2_k3_lowmem.json` |

- **The goal is met on code and structured output.** These are 3.6× and 3.7× over plain, and stable across three repeats (211.7–214.4 and 220.6–221.3).
- **DFlash2 drafts better than MTP.** It verifies 6.8 tokens per step on code, against 5.1 for MTP-5.
- **Windows memory spill is the most likely hidden bottleneck, but not yet isolated.**
  - At 31.5 of 32.6 GB, the driver moved ~640 MB of the server's memory into system RAM, and step times became erratic: code ran at 120 tok/s, and the two structured runs differed (172 vs 214).
  - The rerun changed **three things at once**: KV q8_0, `-ub 256`, and `CUDA_VISIBLE_DEVICES`. Spill fell to 112 MB and code rose to 213.
  - **Isolation test (`runs/q6k_dflash2_k7_isolate_f16kv.json`):** GPU1 stayed hidden, but f16 KV and `-ub 512` were restored, putting VRAM at 31.3 of 32.6 GB.
    - Spill grew from 130 MB at load to 518 MB after the run.
    - Code fell to 125–149 tok/s and structured to 120–131, both erratic, at the same acceptance (83% / 85%).
    - f16 KV is not slow by itself: plain decode was 59.6 tok/s with f16 KV vs 58.5 with q8_0.
    - **Conclusion (now tested):** the slowdown is VRAM pressure and spill, not GPU1 visibility.
  - FAIE2 rule: hold at least ~1.5 GB of VRAM headroom, and record the `\GPU Process Memory\Shared Usage` counter with every run (`bench/serve.ps1` prints it).

### 4-bit target (added 2026-10-25)

The target is `huihui-ai/Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_S.gguf` (15.6 GB). Its SHA-256 `f10c26c7…0176` matches Hugging Face's published hash. All other settings match the low-memory DFlash2 run above.

| Config | code | structured | translation | prose | mean | proof |
|---|---:|---:|---:|---:|---:|---|
| Plain | 75.9 | 75.9 | 76.0 | 76.0 | **75.9** | `runs/q4ks_plain_q8kv.json` |
| **DFlash2 k=7** | **206.2** (acc 70%) | **243.7** (86%) | **192.8** (69%) | 95.2 (23%) | **184.5** | `runs/q4ks_dflash2_k7.json` |

- **Plain got faster with the smaller file.** 4-bit weights raised plain decode 30% (58.5 → 75.9), in line with bytes per token (20.9 → 15.6 GB).
- **Structured and translation gained the most** from the 4-bit target under DFlash2.
- **Code acceptance dropped from 85% to 70%.** The 4-bit target agrees with the drafter less often, which ate most of the gain on code.
- **Prose did not move (95 tok/s).** At 23% acceptance, prose is drafter-limited. A faster base alone will not fix it; the drafting itself has to change for prose (see §5, Phase 4: copy drafts, adaptive k, a prose-trained drafter).
- **Greedy output was byte-identical to plain for structured, translation and prose.** Code flipped on a near-tie.

### Where a DFlash2 step's time goes (Nsight Systems, 2026-10-25)

Trace: `bench/profiles/q4ks_dflash7.nsys-rep`. Workload: Q4_K_S + DFlash2 k=7, one structured and one prose request. The profiler cost ~2%.

Over the **257 steady-state steps** (from one verify launch to the next; the 5 request-boundary intervals are excluded):

| Part | Mean ms per step | Share |
|---|---:|---:|
| Target verify of 8 tokens (CUDA graph 4) | 20.8 | 76% |
| GPU idle (CPU-side work between graphs) | 3.2 | 12% |
| DFlash2 drafter (one ~2.6 ms graph per step) | 2.6 | 10% |
| Small graphs and memcpy | ~0.7 | 2% |
| **Step** | **27.3** (p90 28.1) | |

*Correction:* an earlier version of this table said 31.4 ms per step with 5.7 ms (18%) idle. That averaged the whole 8.26 s window, including prefill and the time between requests. It was not a per-step number.

A plain 1-token step costs 13.2 ms, so **verifying 8 tokens costs 1.6× a plain step**. A memory-bound verify should cost ~1.1×. That gap is the main engine target; the CPU idle is the second.

#### What llama.cpp is doing in the 3.2 ms idle

CPU sampling profiles need administrator rights on Windows, and this build ships no PDB symbols. So each gap was attributed from the trace's event order plus a reading of `common/speculative.cpp` (`common_speculative_impl_draft_dflash`). Averaged over 257 steps:

| Idle per step | Gap position | Code running on the CPU |
|---:|---|---|
| **1.32 ms** | inject graph → drafter's input upload | Server loop handles the accepted tokens, then `draft()` (`speculative.cpp:1198`): batch build, then `llama_decode(ctx_dft)` host setup (graph build, scheduler, CPU token-embedding lookup) |
| 0.44 ms | drafter → target verify | DFlash2 path selection over the selector lattice (`:1252-1274`), then the server's verify batch and the host setup of the 64-layer target graph |
| 0.28 ms | logits D2H → features H2D | CPU sampling and accept over 8 × 248,320 logits, then the CPU interleaves 0.82 MB of target features (`:1122-1133`) |
| 0.60 ms | 3 smaller gaps | Host setup of `llama_encode` (encoder) and `llama_decode` (inject) |

**Root cause: structure, not one slow function.**
- Each step makes **4 separate `llama_decode`/`llama_encode` calls**: verify, encoder, inject and draft. Each one rebuilds a graph on the CPU and synchronizes (about 86 `cudaStreamSynchronize` per step).
- The target's hidden features travel GPU → CPU → GPU **twice** per step: 0.82 MB down, interleave on the CPU, back up; then the encoder output (0.16 MB) down and back up again.

**What it is not:**
- **Not HTTP streaming.** The same prompts with `stream:false` gave identical decode speed (structured 214.6/217.0 vs 216.9/213.6 streamed; prose 82.5/82.7 vs 82.8/82.0).
- **Not the CUDA driver.** API calls account for only ~1 ms per step of the idle.

The finer split inside the 1.32 ms (server bookkeeping vs `draft()` host setup) would need an NVTX-instrumented build. FAIE2 removes the whole host loop anyway, so that split does not change the design.

**Benchmark hygiene:**
- The stream A/B ran while ABUZ8 OS (started 21:54) used ~8% of GPU0's 3D engine. Absolute speeds were 12–14% lower than at 16:09 (structured ~215 vs 244, prose ~82 vs 95); the A/B was interleaved, so its comparison stands.
- Rule: record GPU0 background utilization with every run, and compare runs only under matching background load.

### Hypothesis tests from `research/AYAT_AL_KURSI_STRUCTURE.md`

- **H1: stop drafts on low confidence (`--spec-draft-p-min`) — REFUTED.**
  - Mean went 184.5 → 176.6 at p=0.3 and → 152.8 at p=0.6. Prose fell to 79 and 71.
  - Step time stays 27–31 ms even when only 1.7 tokens are drafted. **The step overhead is fixed per step, not per verified token**, so drafting fewer tokens only loses accepted tokens.
  - This corrects the research file's "~2 ms per extra verified position" reading.
  - Proofs: `runs/q4ks_dflash2_k7_pmin0.3.json`, `runs/q4ks_dflash2_k7_pmin0.6.json`.
- **H4: no idle time in a step — PARTIAL.**
  - `--backend-sampling` (sampling on the GPU) gave structured 244 → 253 (+3.9%), code 206 → 211 (+2.8%), prose +1.4%, and translation 193 → 181 (−6%, a short 41-token answer). All shifts held across 3 repeats; greedy output was unchanged.
  - Most of the idle is therefore not sampling. It is the per-step host loop: 4 decode calls and the features' round trips through the CPU. See "What llama.cpp is doing in the 3.2 ms idle" above.
  - Proof: `runs/q4ks_dflash2_k7_backendsampling.json`.

### Path to 300 tok/s (arithmetic from the measured breakdown, not yet demonstrated)

Structured output yields 6.83 tokens per step, so 300 tok/s needs a step of ≤ 22.8 ms; today it is 28.0 ms.

- Cutting the CPU idle from 3.2 to ~0.5 ms (one device-resident step: features stay on the GPU, one graph or a persistent kernel) gives 25.3 ms, about **270 tok/s**.
- Also bringing the verify graph from 20.8 ms to ~1.2× a plain pass (~15.8 ms) gives ~20.3 ms, about **336 tok/s**.
- **300 needs both.** The verify kernel is the larger lever. (Corrected from an earlier "idle alone ≈ 300", which used the wrong 5.7 ms figure.)

Prose yields 2.57 tokens per step, so it stays near ~135–150 even then. Prose at 300 needs better drafting: copy-drafts, or a drafter matched to the abliterated target.
- **GPU1 isolation.** Without `CUDA_VISIBLE_DEVICES`, llama-server also opened a ~430 MB context on GPU1, even with `--device CUDA0`.
- **Output check.** Greedy output was byte-identical to plain for structured and translation. Code ("Supports" vs "Supported") and prose flipped on near-ties; that is llama.cpp issue #27407.
- **Prose stays near 95–100 at any draft length.** Acceptance is too low (2.9 tokens per step). Lifting prose needs a faster base pass: a 4-bit target instead of Q6_K, and FAIE2 kernels.

## 2. The bar to beat

| Who | Hardware | Result | Source |
|---|---|---|---|
| NInfer (custom CUDA engine, Linux) | 1× RTX 5090 | 74 tok/s plain. With MTP-3: 132 prose, 205 code, 232 structured; corpus mean 168.6 | `hf://neroued/Qwen3.8-27B-nvfp4-NInfer` |
| DFlash2 (SGLang) | 1× H200 | 69 tok/s plain, 184–236 tok/s with DFlash2 (acceptance 4.1–5.5) | `hf://z-lab/Qwen3.8-27B-DFlash2` |
| mia-ai.net TensorFold | 2× DGX Spark | 67–89 tok/s on **Qwen3.8 Flash Next** (an MoE, not the 27B) | mia-ai.net recipe page |

mia-ai.net has not published a 27B number, so it is not the bar. NInfer is: it targets the same model on the same card.
Its sparkDash dashboard (Apache-2.0, `github.com/MiaAI-Lab/sparkDash`) is a good benchmark front-end. `bench/bench.py` measures the same things: decode by category, TTFT, and MTP acceptance.

## 3. Physics budget: why speculation is mandatory

At batch 1, every decoded token streams the whole model through memory. The RTX 5090 has 1,792 GB/s.

| Weights read per token | Ceiling | Realistic (80–85%) | Measured |
|---|---:|---:|---:|
| Q6_K, 20.9 GB | 86 tok/s | ~70 | **59.6** (70%) |
| NVFP4 MLP + FP8 rest, ~19.7 GB (NInfer) | 91 | ~75 | 74 (NInfer) |
| NVFP4 everywhere, ~15 GB | 119 | ~100 | — |

No kernel trick gets plain decoding past ~119 tok/s on one card. Reaching 200 requires verifying several tokens per weight pass:

`tok/s ≈ plain_tps × accepted_tokens_per_step ÷ (verify_cost + draft_cost)`

Example: 100 × 4.4 ÷ 1.45 ≈ 300 tok/s on code with DFlash2-class acceptance. Prose at an acceptance of ~2.5 lands near 170. A mean above 200 is reachable on **one** 5090.

### What the second 5090 can and cannot do

- **Topology:** each card runs PCIe 5.0 **x8**, connected through PCIe bridges (PXB). There is no NVLink, and P2P is not available to GeForce cards on Windows.
- **Tensor parallel across both cards will not help single-stream.** Each token needs 128 all-reduces (2 per layer × 64 layers). At ~15–30 µs per hop through the host, that adds 2–4 ms per token. The whole budget at 200 tok/s is 5 ms.
- **Pipeline parallel** does not speed up a single stream.
- **What does work:**
  - (a) **Asynchronous drafting.** The drafter runs on GPU1 and drafts block n+1 while GPU0 verifies block n. This removes draft cost from the critical path.
  - (b) **Two replicas.** Running one model per card doubles aggregate throughput across the agent fleet.
- Today GPU1 holds OPC-1's live brain (Ornith-35B, 21.8 GB). Using GPU1 means moving or stopping it, and that is Ahmad's call.

## 4. What we take from whom

| Source | Licence | Technique FAIE2 adopts |
|---|---|---|
| NInfer | Apache-2.0 | Mixed NVFP4 (MLP) / row-scaled FP8 (attention, GDN, head) layout; whole-step CUDA Graph; optimized MTP proposal head; FP8 KV |
| DFlash / DFlash2 (z-lab, Inco AI) | Apache-2.0 | Block-diffusion drafter: 7–8 tokens per pass with a candidate path selector. ggml-org already ships `dflash-Qwen3.8-27B-Q8_0.gguf` (2.06 GB) |
| ThunderKittens | MIT | Tile primitives for sm_120 attention and GEMM; a megakernel-style fused decode step |
| TensorRT-LLM | Apache-2.0 | NVFP4 GEMV and GEMM recipes for Blackwell; fused RMSNorm + quantize |
| SGLang / vLLM | Apache-2.0 | Speculative state rollback for Gated DeltaNet layers (checkpoint per draft position); radix prefix cache |
| Modular MAX / Mojo | Mixed | Graph-level kernel fusion ideas only. MAX is not open source end to end, so FAIE2 copies no code from it |
| TensorFold (mia-ai) | — | "Copy drafts" (n-gram table) layered on top of MTP drafts; lossless keyed sampling |
| FAIE (ours, `reference/faie`) | own | Proof discipline: oracle gate against llama.cpp, requested-vs-executed counters, CUDA-graph traps (§5 of its handoff), Windows nvcc build line. FAIE's MLA and MoE code does **not** apply to this dense hybrid model |

"freetoken" and "strata" could not be identified. Links are needed before they can be studied.

## 5. FAIE2 architecture: phases with gates

Each phase ships only when its gate passes and the proof JSON is written.

1. **Phase 0: harness.** Done: `bench/bench.py` and three proof runs.
2. **Phase 1: llama.cpp ceiling on this box.** Swap in DFlash2 (2 GB download), then a ~15–16 GB 4-bit target.
   - Gate: the best off-the-shelf number, measured. This is the number FAIE2 must beat.
3. **Phase 2: FAIE2 target forward.** Native CUDA for Windows (sm_120), loading the official NVFP4 safetensors.
   - Kernels: NVFP4 GEMV; GDN recurrent step; GQA flash-decode with 16 full-attention layers; fused norm, rope and gate; one CUDA Graph per step.
   - Gate: per-layer cosine 1.000000 against PyTorch, **and** top-5 against the llama.cpp oracle on 20 prompts (the gate FAIE never passed), **and** at least 80 tok/s plain.
4. **Phase 3: lossless verify.** The k-token verify path is built from the same row-wise kernels as 1-token decode, so the verify output is bit-identical to plain decoding. GDN state is checkpointed per draft position, with no recompute on reject.
   - Gate: greedy output byte-identical to plain on all categories (llama.cpp fails this today).
5. **Phase 4: drafters.** MTP, then DFlash2, then copy-drafts. Draft length adapts per request from a rolling acceptance window (3 for prose, 5–7 for code and structured).
   - Gate: mean above 200 tok/s across the four categories at temperature 0, and recorded at temperature 0.6.
6. **Phase 5: second GPU.** Asynchronous drafter on GPU1, or two replicas behind one OpenAI endpoint. Needs GPU1 free.

## 5b. Phase 2 log: the 8-token verify path (started 2026-10-26)

**Target file:** `Qwen3.8-27B-ABLITERATED-Q6_K.gguf`. It is uniform: all 506 weight matrices are Q6_K, the rest F32. That lets one weight kernel prove the whole path. The 4-bit UD file mixes 11 formats (Q3_K…IQ2_S), and those come after the gate.

**Code:**
- `engine/faie2.cu`: a single translation unit holding the GGUF mmap loader, kernels, engine and C API. Build with `engine/build.ps1` (nvcc 13.1 + MSVC 14.44, sm_120).
- Tests and harness: `tests/` (Python 3.11 with torch 2.10 cu128 for the reference; ctypes for the DLL).

**Math source:** every op was read from llama.cpp, not assumed:
- `qwen35.cpp` and `delta-net-base.cpp`.
- `ggml-cuda/gated_delta_net.cu`: value head h reads key head h % 16, and the output is scaled by 1/√128.
- `rope.cu`: IMROPE pairs (p, p+32) over 64 rotary dims, base 1e7.
- `norm.cu`: L2-norm is x / max(‖x‖, 1e-6).
- `unary.cu`: softplus with a threshold of 20.
- `ssm-conv.cu`: `out[t] = Σ in[t+j]·w[j]`, no flip.
- `dequantize_row_q6_K`.

**Oracle (llama.cpp, same file, GPU0, f16 KV, greedy):**
- `oracle/oracle_q6k.json`: 6 prompts (prose, code, JSON, Arabic, math, a 120-token passage) × 16 positions, top-10 log-probs each.
- Speed (`oracle/llama_bench_q6k.md`): 1 token in 17.49 ms (57.19 tok/s); an 8-token batch in 24.75 ms (pp8 323 tok/s), so 1.41×.

**Gates so far:**

| Gate | Result | Evidence |
|---|---|---|
| Q6_K GEMV vs gguf-py dequant (4 real matrices, nt 1/2/8) | **PASS**: max rel err ≤ 7e-7, cos 1.0000000 | `tests/test_q6k_gemv.py` |
| Full forward vs llama.cpp oracle | **96/96 top-1 identical**, top-5 overlap 4.94/5, 0 real mismatches | `tests/gate_oracle_nograph.json` |
| Batch invariance (1 vs 3 vs 8 tokens per forward) | **Bit-identical** logits (max diff 0.0) on code, JSON and long | `tests/test_batch_invariance.py` |
| Log-prob closeness, adjudicated against a float64 reference | **PASS**: the gaps were llama.cpp's error, not FAIE2's (table below) | `tests/ref_forward.py`, `tests/compare_ref.py` |

**Float64 adjudication.** `ref_forward.py` is an independent torch-float64 forward: weights come from gguf-py dequant, and no FAIE2 code is used. Results, as max error vs exact math over 16 positions each:

| | FAIE2 (fp32 GEMV) | FAIE2 final (fp16-activation tensor cores) | llama.cpp |
|---|---:|---:|---:|
| code, top-1 log-prob | 0.0004 | 0.0049 | **0.2777** |
| JSON, top-1 log-prob | 0.0002 | 0.0007 | 0.0056 |
| code, top-10 | 0.0015 | 0.0133 | 0.55 |
| JSON, top-10 (rare tail, around −10) | 0.112 | 0.61 | **3.97** |

Notes on the adjudication:
- The original 0.15-nat threshold assumed llama.cpp was exact. It isn't: it rounds activations to 8 bits (q8_1) inside its matmuls.
- `gate_oracle.py` now checks only what llama.cpp can vouch for: same greedy token and same top-5. Log-prob closeness is judged against the float64 reference.
- Limit: the reference was written from the same reading of llama.cpp's graph, so it cannot catch a misread architecture. The independent 96/96 greedy match with llama.cpp rules that out.

**Batch invariance is a property llama.cpp lacks.** FAIE2 computes each token row with the same summation order at any batch size. So an 8-token verify equals 8 single decode steps bit for bit, and speculation is truly lossless (the "damp/dry" flips in §1 cannot happen).

**Speed: how the 8-token verify got from 53.5 ms to 18.1 ms.** All numbers are full 64-layer forwards on GPU0 with Q6_K, from `tests/bench_forward.py`; proofs are `tests/bench_forward_*.json`. Kernel breakdowns come from nsys traces in `bench/profiles/faie2*`.

| Step | 1 token | 8 tokens | 8/1 | What changed, and why |
|---|---:|---:|---:|---|
| v1 GEMV, no graphs | — | 53.5 ms | — | fp32 CUDA-core dequant-GEMV, 4 rows per warp |
| v2 GEMV (split-K, SoA repack, float4) + graphs | 21.39 | 38.47 | 1.80× | Fixed parallelism on 5,120-row matrices. At 8 tokens the GEMV was **compute-bound**: 8 fp32 FMAs per weight (LM head only 614 GB/s) |
| **v3 tensor-core GEMV** + graphs | 20.42 | 21.26 | 1.04× | `mma.m16n8k16`: 8 tokens = the n=8 side; Q6_K q−32 is exact in fp16, one scale per 16-group applied after the mma. LM head 1.49–1.53 TB/s at **every** token count |
| **v3 + fusion** + graphs | **17.50** | **18.14** | **1.04×** | Grouped launches (qkv\|z\|β\|α, q\|k\|v, gate\|up via concatenated weights); residual add in the GEMV epilogue; RMSNorm (1024 threads), SwiGLU, gated norm and attention write the fp16 GEMV input directly |
| llama.cpp (same file) | 17.49 | 24.75 | 1.41× | — |

What made v3 work:
- An interleaved repack, so every warp load is one contiguous 512 B. The first v3 layout gave 50% sector efficiency and ~1.0 TB/s.
- Register double-buffering: the next block's weights load while the current block computes.
- Always running all 8 token columns, zero-padded. That keeps the engine **bit-identical across batch sizes**, so speculation stays lossless.

Dead ends, kept for the record:
- 4 rows per CTA in v2 made 8-token *slower*: 158 registers gave 25% occupancy.
- Nsight Compute counters are blocked on this machine (`ERR_NVGPUCTRPERM`, a Windows admin setting), so kernel tuning used nsys timelines and DRAM-bound microbenchmarks instead.

**Phase 2 gate status:**
- Correctness: ✅
- Bit-identical batching: ✅
- 8-token verify faster than llama.cpp: ✅ (18.14 vs 24.75 ms)
- ≥ 80 tok/s plain: ❌ (57 tok/s on Q6_K = 17.5 ms per token; the 20.9 GB file caps plain decode at ~86 tok/s even at 100% bandwidth)

**Projection (arithmetic only, not measured).** These are FAIE2 verify times combined with the DFlash2 acceptance measured on llama.cpp:
- With Q6_K: step ≈ 18.1 + ~2.6 drafter + ~0.5 → ~21 ms. Structured (6.83 tok/step) ≈ **320 tok/s**, prose (2.9) ≈ 135.
- A ~15.6 GB 4-bit target would cut the verify to ~13.5 ms, so structured ≈ 410 tok/s.

Next:
- Phase 3: GDN state checkpoints for rollback, plus device-side accept and sampling.
- Phase 4: run the DFlash2 drafter inside FAIE2, so features never leave the GPU (that removes the measured 3.2 ms per step of host idle).

## 5c. Phase 3 log: rollback and device-side greedy accept (2026-10-26)

**Design: replay-commit instead of snapshots.** Snapshotting every DeltaNet state after each of 8 tokens writes 1.2 GB per step (~0.8 ms). FAIE2 does this instead:
- **The forward never writes recurrent state.** It reads the committed conv taps and S, and leaves per-layer history: qkv rows, conv outputs, g, β (47 MB).
- **Argmax and accept run on the GPU.** `k_argmax_part` (64 parts per row, ties go to the smallest id) and `k_accept` (k = matching draft prefix; emit d1..dk plus the bonus argmax).
- **Commit by replay.** `k_conv_commit` and `k_gdn_commit` replay exactly the first k+1 tokens into the committed state. Rejected drafts never touch state. KV entries past `n_past` are scratch and are overwritten by the next block before they are read.
- **The per-token delta-rule update is one device function, `gdn_step`.** It uses explicit `__fmaf_rn`/`__fmul_rn`/`__fsub_rn`, so the forward and the replay round identically.
- **Plain decoding is a 1-token verify, and prefill is forward + commit-all.** All three paths run the same code.
- **The whole step is one CUDA graph per block size:** forward, argmax, accept, commit. The API is `faie2_verify(tokens, nt, emit, n_emit, argmax, logits)`.

**Gates:**

| Gate | Result | Evidence |
|---|---|---|
| Phase 2 gates on the Phase 3 build | 96/96 oracle, bit-identical batching, float64 top-1 error ≤ 0.005 (llama.cpp 0.28) | `tests/gate_oracle_phase3.json` |
| **Lossless speculation** (3 prompts × 4 drafters × 120 tokens) | **PASS, with and without CUDA graphs** | `tests/test_lossless.py`, `tests/lossless_result.json`, `tests/lossless_result_nographs.json` |

What the lossless gate checked:
- Every emitted token equals plain greedy.
- Every logits row behind an emitted token is bit-identical.
- k equals the drafter-forced value at every step.
- All of k = 0…7 were exercised.
- Plain greedy equals llama.cpp's greedy output on all three prompts.

**Speed** (`tests/bench_spec.py`, Q6_K, GPU0, graphs on; ABUZ8 OS rendering on GPU0 at ~6% during the run):

| | GPU | Host |
|---|---:|---:|
| Plain greedy step (1-token verify, incl. accept + commit) | 18.22 ms | 18.32 ms → **54.6 tok/s** |
| Speculative step (8-token verify, incl. argmax + accept + commit) | 18.93 ms | 19.03 ms |

- The commit replay plus accept adds ~0.7 ms over the bare forward (17.5 / 18.1 ms). The arithmetic says ~0.2 ms, so this is a tuning target.
- **Projection, drafter cost excluded,** at llama.cpp-measured DFlash2 acceptance lengths:

| Content | Acceptance length | Projected |
|---|---:|---:|
| Prose | 2.57 | 135 tok/s |
| Code | 5.75 | 302 tok/s |
| Translation | 5.86 | 308 tok/s |
| Structured | 6.83 | 359 tok/s |
| Perfect-drafter ceiling | 8 | 420 tok/s |

- Adding a ~2.6 ms in-engine drafter lowers these by ~12%, e.g. structured ~316 and code ~266.

**Next (Phase 4): DFlash2 inside FAIE2.**
- Load the `dflash` GGUF (5 layers; non-causal block attention over context features with sliding window 2048; 2-tap grouped conv; selector top-16 lattice).
- Feed it target features from layers 6/20/34/48/62 directly on the GPU, and pick drafts on the device.
- That removes all four host round trips llama.cpp makes per step (§ "What llama.cpp is doing in the 3.2 ms idle").

## 5d. Phase 4 log: DFlash2 inside FAIE2, measured end to end (2026-10-26)

**What runs on the GPU in one CUDA graph per step** (`faie2_spec_step`):
1. **Draft.** Noise block `[anchor, MASK×7]` (MASK = 248070), 5 DFlash2 layers.
   - Each layer: dynamic 2-tap block conv before/after attention and FFN; non-causal attention over injected context + block, sliding window 2048.
   - Then the target's LM head, an on-GPU top-16 per position (two-stage warp selection; ties go to the smaller id), the selector lattice `next[c_j]·(prev[c_i]⊙hid)+logit_j`, and the greedy walk.
2. **Verify** the block `[anchor, d1..d7]` (token embedding on the GPU from a device copy of `token_embd`), capturing target features at layers 6/20/34/48/62.
3. **Accept** greedily on the GPU, then **commit** k+1 tokens by replay.
4. **Inject** drafter K/V for the committed rows only: `fc` → RMSNorm → per layer `rope(knorm(Wk g))`, `Wv g`.
5. **Next anchor** = the bonus token.

The host sends one position and reads at most 8 tokens per step. llama.cpp makes 4 decode calls and 2 feature round trips through the CPU per step.

**New kernels:**
- Q8_0 tensor-core GEMV, the same scheme as Q6_K: int8 is exact in fp16, one scale per 32 elements (`tests/test_q8_gemv.py`: ALL PASS, max rel err ≤ 4.6e-4).
- `k_df_conv`, `k_df_qk`, `k_df_attn`, `k_topk_part`/`final`, `k_df_scores`, `k_df_walk`, `k_feat_capture`, `k_embed_q6k`.

**Gates:**

| Gate | Result | Evidence |
|---|---|---|
| Target gates after moving embeddings to the GPU | oracle 96/96, bit-identical batching, lossless PASS | `tests/gate_oracle_phase4.json` |
| **Drafter vs independent float64 numpy reference** (`tests/ref_drafter.py`, weights via gguf-py) | **PASS**: identical 7 drafts on code/prose/Arabic, top-16 sets identical at all positions (16/16), lattice score diff ≤ 0.023 | console log in this section |
| **End-to-end lossless** | spec output == plain greedy, token for token, all 4 categories (1,576 tokens) | `tests/spec_e2e_result.json` |

The gate caught a real bug: the GGUF reader gave Q8_0 tensors a byte size of 0. That produced an illegal memory access in the repack. Sizes are now computed for Q8_0, and both loaders fail on any size mismatch.

**End-to-end speed** (`tests/test_spec_e2e.py`):
- Setup: Q6_K target + Q8_0 DFlash2, GPU0, CUDA graphs, greedy, the exact chat-template token ids llama-server used (`oracle/bench_prompts_q6k.json`).
- Measurement: decode loop wall time, stop at `<|im_end|>` or 512 tokens. ABUZ8 OS was using ~7% of GPU0 during the run.

| Category | FAIE2 spec | FAIE2 plain | Tokens/step | ms/step | llama.cpp DFlash2 (same files) | Same text as llama.cpp greedy |
|---|---:|---:|---:|---:|---:|---|
| code | **296.3** | 52.9 | 6.83 | 22.87 | 213.2 | 512/512 |
| structured | **301.0** | 52.4 | 6.92 | 22.71 | 220.9 | 512/512 |
| translation | **254.7** | 54.1 | 5.71 | 21.93 | 145.6 | 40/40 |
| prose | **117.2** | 52.9 | 2.72 | 23.04 | 95.0 | first 150, then llama.cpp near-tie |
| **mean** | **242.3** | | | | **168.7** | **+44%** |

**Where the step time goes, and what is next:**
- A step is ~22.9 ms: verify ~18.9 (incl. accept+commit) + drafter and inject ~4 ms. llama.cpp's drafter graph took ~2.6 ms.
- The drafter reads ~1.8 GB of Q8_0 plus the 1.04 GB LM head, so its floor is ~1.9 ms at 1.5 TB/s. Several drafter GEMVs are small (conv_proj 80 tiles, selector 16 tiles) and latency-bound.
- **Next levers, in order:**
  1. Drafter kernel efficiency (~1.5–2 ms to win).
  2. Commit overhead (~0.5 ms).
  3. A 4-bit target (Q4-class formats or NVFP4; verify ~13.5 ms). That takes structured and code toward ~380–400 and prose toward ~150.
- Prose stays drafter-limited (2.7 tokens per step).

## 5e. Phase 5 log: lossless kernel work, measured under thinking load (2026-10-26)

Ahmad's rule for this phase: no quality loss, and judge speed under load, not by arithmetic.

**The load test** (`tests/load_test.py`, results `tests/load_result_*.json`, GPU logs `tests/load_gpu_*.csv`):
- 8 real tasks with **thinking on** (code, math, logic, structured, Arabic, debugging, science, planning); prompt ids from llama-server's own chat template (`tests/load_prompts.json`).
- Each task generates up to 2,048 tokens, all 8 back to back, **2 rounds**: about 26k tokens and 4 minutes per engine, GPU0 at full load.
- GPU0 clock, power, temperature and utilization are sampled every second. Decode tok/s excludes prefill on both sides.

| Engine (same Q6_K + Q8_0 drafter files, GPU0) | tok/s under load | GPU0 median |
|---|---:|---|
| llama.cpp DFlash2 b10510 (KV q8_0, ub 256, n_max 7) | 116.3 | 2557 MHz, **574 W (at the power limit)**, 60 °C |
| FAIE2 Phase 4 build (`engine/archive/faie2_phase4_2026-10-26.dll`) | 140.6 | 2910 MHz, 456 W |
| **FAIE2 now** | **178.6** (1.54× llama.cpp, +27% over Phase 4) | 2812 MHz, 479 W, 54 °C |

Per task, FAIE2 now vs llama.cpp:

| Task | FAIE2 now | llama.cpp |
|---|---:|---:|
| code | 191 | 159 |
| math | 310 | 180 |
| logic | 205 | 140 |
| structured | 213 | 131 |
| Arabic | 169 | 86 |
| debug | 160 | 108 |
| science | 184 | 112 |
| plan | 141 | 100 |

**Lossless under load:**
- Spec output == plain greedy for 2,048 tokens on code and Arabic.
- Round 2 repeated round 1 token for token.

**Where FAIE2 and llama.cpp differ, exact math was asked** (`tests/check_first_token.py`):
- At the Arabic prompt's first thinking token, float64 picks "Thinking", FAIE2's pick, over llama.cpp's "We" by 0.094 nats.
- FAIE2's log-probs match float64 to within 0.004 nats.
- llama.cpp's q8_0 KV cache is the likely cause.

**Old vs new FAIE2 build:**
- 6 of the 8 outputs are identical.
- `science` differs at token 171, a three-way tie ("3"/"2"/"5" within 0.015 nats). Float64 sides with the **old** build, by 0.009 nats (`tests/check_flip.py`). A tie that small can fall either way with any fp32 arithmetic. It is recorded, not hidden.

**What changed (every step A/B'd inside the real speculative step, `tests/ab_step.py`):**

| Change | Effect | Status |
|---|---|---|
| **Split-KV tensor-core attention** for target and drafter (`k_fa_split` + `k_fa_combine`) | Short context: drafter attention 205 → 11 µs/layer, target 52 → 20 µs/layer; step −1.5 ms. The old kernels grew ~87 ns per key, so under long thinking contexts this is the main win. | **kept** |
| PDL (programmatic dependent launch) | +0.5 ms with the new attention when on every kernel; −0.3 ms on GEMVs only; −0.48 ms on all-but-attention at short context; equal to GEMV-only under load (179.9 vs 178.6) | mode 3 kept |
| `__launch_bounds__(256, 2)` on the Q8_0 GEMV | ptxas had used 130 registers, so one CTA per SM | kept |
| GEMV v4 (balanced split-K, 4 warps) | no gain | off |
| GEMV v5 (warp-level stream-K) | +1.4 ms (slower) | off |
| GEMV v6 (data-parallel rounds + stream-K remainder) | +3.5 ms (slower) | off |

How the new attention keeps quality:
- q and p are split into fp16 hi + lo parts, so products are exact and accumulate in fp32.
- The chunk → split map is fixed, so results are bit-identical for any nt.

How the GEMV variants were kept honest:
- v4/v5/v6 all pass the float64 GEMV gates (deterministic, batch-invariant), but none beat v3, so **v3 stays**.
- v6's first unit test passed by accident. Empty CTA ranges were waited on, and a tile only completed on a repeated run with identical inputs.
- The unit tests now return the **first** run's output.
- Why persistent and stream-K GEMVs lose to v3 on this GPU is **not yet understood**; it is the main open lever (v3 averages ~1.40 TB/s in situ vs 1.66 on the LM head).

**Gates on the final config:**
- Oracle 96/96.
- Float64 top-1 error 0.0052 (code) and 0.0005 (json), vs llama.cpp's 0.28 and 0.0056.
- Lossless at every k in 0..7.
- Drafter identical to float64.
- Bit-identical across block sizes.

## 7. FAIE2 as the engine: no llama.cpp, no WSL (started 2026-10-26)

**Goal (Ahmad):** FAIE2 serves our agents directly, the way vLLM / SGLang / MAX do, natively on Windows, with full tool calling, and without losing capability or power.

**Verified on the web (2026-10-26):**

*Windows vs Linux.* Device code runs at the same speed. WDDM adds per-launch and sync overhead (forum measurements of ~10–80 µs per launch), and one 2026 RTX 5080 llama.cpp test had Linux ~15–25% ahead. FAIE2 replays CUDA graphs, so a step spends 0.4 ms of 19 ms between kernels. The Linux advantage is a launch-overhead problem, and graphs remove it, so no WSL layer is needed.

*Competitors on Qwen3.8-27B, one RTX 5090.* NVFP4 weights (lighter than our Q6_K), 256-token answers, thinking off:

| Engine | Plain tok/s | With drafter |
|---|---:|---:|
| SparkInfer 0.5.5 | 97.1 | 264.8 (DSpark) |
| SGLang 0.5.18 | 85.8 | 161.7 |
| vLLM | 86.8 | – |
| llama.cpp + MTP | – | ~142 |

SGLang's own "206.1 tok/s" claim has no workload stated.

*New drops in the past 7 days:*
- **DeepSeek V4.1-Flash-1017** (Oct 17, 552B, too big for 64 GB).
- **DeepInfer**: DeepSeek's serving engine, open-sourced Oct 20 (SM100 / Ascend kernels).
- **Humming**: Ant Group quantized kernels for vLLM (Oct 21).
- **llm-d 1.0** (Oct 20).
- **Qwen 4: not released.** Qwen3.8-Flash-Next is its architecture preview.

No "Linus" kernel project exists online. The ask reads as Linux, which the Windows vs Linux point above answers. "Continuum" and "SubQ" were not found.

*Already on disk* (`E:\ABU\HERMES\engines`):
- **Strata**: native Windows/Linux engine for Qwen3.8-Flash-Next. Community RTX 5090 result: 179 / 176 / 165 tok/s at 4K / 32K / 128K context. 12 GB cards work, with experts streamed from RAM.
- **FreeToken**: edge MoE engine. Runs 290B+ MoE on gaming PCs, keeps semantic anchor checkpoints of recurrent state, has OpenAI/Anthropic APIs.
- **Agent tooling, not kernels:** helix-loop, thunderkit, sima2-agent, frida.

**Phase A: done today** (`serve/`)

What was built:
- **Own tokenizer and chat template** (`faie2_tok.py`): GGUF vocab and merges, the qwen35 split regex, jinja2 as in HF.
  - Gate `test_tok.py`: 12 template-rendered prompts (thinking on and off) give **exactly** llama-server's ids, and 5 multilingual/code/special-token texts give exactly `llama-tokenize`'s ids.
- **OpenAI-compatible server** (`faie2_server.py`, `serve_faie2.ps1`, port 8300, GPU0 only):
  - `/v1/chat/completions`, with tools, SSE streaming, and `reasoning_content` split from content
  - `/v1/completions`
  - llama-server-style `/completion`, `/tokenize`, `/detokenize`, `/apply-template`
  - stdlib HTTP only
- **Prefix cache:** new engine API `faie2_snapshot_*` saves the recurrent state (157 MB per slot); K/V is positional.
  - Snapshots are taken at the end of the system block, at the start of the generation prompt, and at the prompt end.
  - A repeated 319-token prompt went from 701 ms to 0.4 ms prompt time, with identical output.

Measured:
- **Tool calling** (`test_tools.py`, 8 cases: choice, typed and array arguments, parallel calls, Arabic, tool result, agent step, no-tool): **FAIE2 8/8, llama.cpp 8/8, identical calls.**

| | FAIE2 | llama.cpp |
|---|---:|---:|
| Generation on these calls | **264 tok/s** | 169 tok/s |
| Prompt reading | 470 tok/s | **1,155 tok/s** |
| Wall time, whole suite | 11.9 s | **7.8 s** |

**Phase B: done (2026-10-26).** Batched prefill (`tests/test_prefill.py`):

- **Forwards of up to 64 tokens.** Buffers, the recurrent history stride and the norm/SwiGLU/feature kernels take a padded row count, and attention rows are a runtime count.
- **`k_q6k_mma_n`** (one weight read for 8 token groups) reached 792 tok/s. It was limited by activations re-read from L2: 216 GB per 64-token forward, ~47 TFLOPS.
- **`k_q6k_gemm_r` + `k_gemm_reduce`** fix that while staying bit-identical.
  - The 8-token kernel's output is the in-order sum of 8 per-warp "residue" partials.
  - Residue CTAs compute one partial for 8 neighbouring row tiles, staging the activations once in shared memory (cp.async, double-buffered).
  - The reducer adds 0 + p0 + … + p7 in order.

**Gate:** a 1,536-token prompt prefilled in 8-token, 64-token and ragged chunks gives **bit-identical** next-token logits and an identical 128-token speculative continuation. Oracle 96/96, lossless, drafter and batch-invariance gates also pass after the change.

| Prompt reading | tok/s |
|---|---:|
| 8-token chunks (old) | 471 |
| `k_q6k_mma_n` | 792 |
| **`k_q6k_gemm_r`, 64-token chunks** | **1,676** |
| `k_q6k_gemm_r`, ragged chunks | 1,281 |

**Server, same 8 tool cases** (FAIE2 vs llama.cpp DFlash2):

| | FAIE2 | llama.cpp |
|---|---:|---:|
| Prompt reading | **1,530 tok/s** | 1,155 tok/s |
| Generation | **261 tok/s** | 169 tok/s |
| Wall time, whole suite | **5.8 s** (was 11.9 s) | 7.8 s |
| Tests passed | 8/8 | 8/8 |

The tool calls are identical between the two engines.

**Phase B was:** batched prefill, which was the blocker for agents.
- Run up to 64 tokens per forward. Each weight block is dequantized once and multiplied into 8 token groups, with exactly the per-column math of the 8-token path. Prefill then stays bit-identical to decode, and outputs don't depend on cache hits.
- The DeltaNet recurrence updates state in place during prefill. Attention takes 64 query rows.
- Drafter injection is batched the same way.
- Target: ≥1,500 tok/s prompt reading (llama.cpp 1,155 here), then 32K+ context.

**Phase C:** lossless speculative sampling for temperature > 0.
- DFlash2 drafts are point masses, so accept draft d with probability p(d), else sample from p with d removed. That is exactly the target distribution.
- Device RNG seeded per request.

**Phase D:**
- Anthropic `/v1/messages`.
- A model registry.
- Port the server to C++ once the API is stable, removing the Python layer.

**Phase E: "Forge", the self-optimizing loop.**
- Kernel variants are proposed by the local model through FAIE2's own server (or by Claude), then compiled.
- Each variant is gated against independent references: float64 GEMV/attention, the llama.cpp oracle, lossless, batch invariance, and the first-run output rule.
- Each is measured in situ (`ab_step.py`, `load_test.py`), and only winners are kept, with the evidence.
- 2026 papers (KernelBench-Verified) show LLM-written kernels often game narrow tests. The independent float64 gates are the defense; today's v6 bug is the example.

**Phase F: brain upgrade.**
- Qwen3.8-Flash-Next (index 40 vs 34) needs MoE kernels (learn from Strata and FreeToken on disk).
- It needs Ahmad's OK for a ~76 GB download and for GPU1.
- Qwen 4 when it ships.

## 8. Where the work goes, and the pulleys (2026-10-26)

**Pareto of one speculative step** (`tests/analyze_pareto.py` on `bench/profiles/pareto_np`, final config, PDL off for true times): **20.07 ms per step**.

| Share | Time | Where it goes |
|---:|---:|---|
| 74.5% | 14.96 ms | Target weights (20.0 GB Q6_K, ~1.34 TB/s achieved) |
| 6.2% | 1.25 ms | Vocabulary head read twice, draft and verify (2.1 GB) |
| 5.8% | 1.17 ms | Drafter weights (1.9 GB Q8_0) |
| 4.5% | 0.90 ms | DeltaNet recurrence |
| 3.7% | 0.74 ms | Small kernels |
| 2.7% | 0.54 ms | Idle gaps |
| 1.5% | 0.29 ms | Attention |

**86.6% of the step is moving weight bytes out of VRAM.**

**Energy per token under the thinking load:**

| Engine | J/token |
|---|---:|
| **FAIE2 now** (479 W / 178.6 tok/s) | **2.68** |
| FAIE2 Phase 4 (456 W / 140.6) | 3.24 |
| llama.cpp (574 W at its power limit / 116.3) | 4.94 |

**Pulleys, ranked by what they move.** Estimates are marked; nothing below is measured yet.

1. **Both GPUs' memory buses (tensor parallel).** Each 5090 holds half of every matrix, and one ~80–160 KB exchange per layer is the only cross-GPU traffic.
   - Estimate: step ~20 → ~11–12 ms on Windows (exchange through system RAM), ~10 ms with Linux P2P.
2. **GEMV efficiency.** 1.34 → 1.66 TB/s (the LM head already reaches 1.66) is about −3 ms. Why the split-K and stream-K variants lose is still open.
3. **Fewer bits per weight.** Q6_K → Q5_K is about −17% bytes, with a quality cost to be measured against float64 first.
4. **Mixture of experts.** Qwen3.8-Flash-Next reads ~6B active parameters per token, about 10× fewer bytes per token, and scores 40 vs 34. This is the route to 500–1000 tok/s per stream.
5. **Tokens per step.** Under the thinking load the drafter averages 3.3–3.9 accepted tokens per step; tree drafts could raise that.

**This machine, read with `nvidia-smi`:**
- GPU0 ↔ GPU1 topology is **PXB** (through PCIe bridges, *not* through the CPU's host bridge). P2P read status is **GNS**: blocked by the GeForce Windows driver.
- BAR1 is 32,768 MiB on both, so Resizable BAR is on.
- Links: GPU0 PCIe Gen5 x8; GPU1's slot maxes at Gen4 x8.
- Verified online: P2P on RTX 50-series works only on Linux, with patched NVIDIA open kernel modules (aikitoria/open-gpu-kernel-modules, from tinygrad's patch; needs IOMMU passthrough plus Resizable BAR).
- The 1 TB USB disk (Insignia enclosure) holds a Linux install: EFI 1 GB, Linux filesystem 883 GB, swap 48 GB.

**Storage.** A model's drive only sets how long it takes to *load*. After that, weights live in VRAM/RAM and the drive is idle.

| Drive | Free | Approx. time to load a 76 GB model |
|---|---:|---:|
| C: NVMe | 205 GB | seconds |
| D: USB SSD | 457 GB | ~1–2 min |
| F: USB hard disk (exFAT) | 875 GB | ~7 min |

Never stream weights from a hard disk during inference: ~10 ms per seek. The Flash-Next n-gram shard must sit in RAM, not be memory-mapped from F:.

## 6. Open decisions for Ahmad

- **Downloads.** The DFlash2 drafter is done (downloaded and benchmarked 2026-10-25).
  - Next: a ~15–16 GB 4-bit target, to raise the plain base from 58.5 tok/s toward ~80. That lifts every category, prose included.
  - Then: the `unsloth/Qwen3.8-27B-NVFP4` safetensors (~16 GB) for Phase 2.
- **Abliterated or official target.** The DFlash2 drafter was trained on the official model, so acceptance against the abliterated weights must be measured, not assumed.
- **GPU1.** Should OPC-1's brain move off GPU1 for Phase 5 work, or should FAIE2 stay on GPU0 only?
- **Keep the brain or switch (2026-10-26).** Source: Artificial Analysis Intelligence Index v4.3.2, read on the site.

  | Model | Index |
  |---|---:|
  | Qwen3.8-27B, xhigh thinking | 34 (#1 of its size class) |
  | Qwen3.8-27B, no thinking | 20 |
  | Claude Opus 4.6, max (marked as an estimate) | 32 |
  | **Qwen3.8-Flash-Next** | **40** |
  | Best open models (552B–1T, cannot fit 64 GB) | 45–46 |

  Flash-Next is 180B total / 6B active, plus a 51B n-gram lookup table.
  - **Fit:** its abliterated GSQ-RCO GGUFs need 47 GB (IQ3_XXS) or 54.8 GB (IQ3_S) resident, so **both** 5090s; the 28.8 GB n-gram shard can stay in system RAM.
  - **Speed (by arithmetic, unmeasured):** ~6B active params is far fewer bytes per token than 27B dense, so plain decode should be much faster.
  - **Speculation:** on an MoE, an 8-token verify touches up to 8× the experts, so DFlash2-style speculation gains less.
  - **Needs:** Ahmad's OK for a ~76–84 GB download (to F:) and for GPU1; a newer llama.cpp; the same load test plus an intelligence check; then an MoE engine (FAIE3) if it wins.

## 9. FAIE2-FN: Qwen3.8-Flash-Next on both 5090s (2026-10-26)

**What exists (verified on disk and by gates):**
- Model: SC117 Qwen3.8-Flash-Next GSQ-RCO abliterated IQ3_XXS, `E:\ABU\MODELS\llm\qwen3.8-flash-next-abliterated\IQ3_XXS\` (shard 1 47.34 GB weights, shard 2 28.80 GB n-gram table; sizes match HF exactly). Arch `qwen4exp`: 48 layers (36 DeltaNet + 12 attention with QSA indexer), 512 experts top-10 (640 wide) + shared expert, 4-stream hyper-connections (low rank 320), per-layer n-gram embedding at layer 1, vocab 248320. 14 storage formats.
- Oracle `flashnext/ref_fn.py`: the official transformers `modeling_qwen4_exp.py` (local transformers 5.20.1) in float64, weights dequantized from the GGUF (gguf-py + Q2_0 from ggml-quants.c), experts and n-gram rows fetched lazily. GGUF conventions found: norms stored as 1+w, ssm_a = -exp(A_log), DeltaNet value heads tiled [v][k] (proved: HF order gives NLL 3.68 and "city of Paris is Paris"; tiled gives 1.06 and Paris/Berlin/Rome).
- Engine `engine/faie2fn.cu` -> `faie2fn.dll` (`build_fn.ps1`), wrapper `flashnext/fn.py`. All weights resident in VRAM, layer split by free VRAM (auto split 26: GPU0 23.2 GB, GPU1 24.5 GB); n-gram table in pinned host RAM read directly by GPU0; residual stream handed GPU0 -> pinned RAM -> GPU1 by copy engines, GPU1 waits on an event (no CPU on the data path). Positions in device memory, one CUDA graph per card per block size. fp32 math, fp32 KV cache.
- Gates: `test_fn_quants.py` QUANT GATE PASS (13 formats dequantize with exactly 0 error vs gguf-py; GEMV rel err ~1e-7). `gate_fn_ref.py` ORACLE GATE PASS on the capital-cities sentence: residual stream within 1e-5 of the oracle at every layer, top-1 21/21, max log-prob diff < 5e-5.
- Speed (`bench_fn.py`, one stream, greedy, no speculation): 50 tok/s eager -> 74.4 tok/s with graphs (13.4 ms/token) -> 137.4 tok/s (7.28 ms) after the tiled GEMV work in 9a -> **173.6 tok/s (5.76 ms) after the kernel fusion in 9b** -> 169.5-170.6 `bench_fn` / ~171 in-process A/B after 9c (8-row gate/up kernel, +1%). The unchanged 9b DLL measured 162-168 `bench_fn` on 27 Oct against 173.6 on 26 Oct, so absolute numbers moved with the machine's state (cause not found); only a same-DLL A/B is comparable. Prefill 88 -> ~125 tok/s (8-token forwards, untuned).

**Where the 13.4 ms went before 9a (nsys, `analyze_fn_prof.py`; the small-shape bench numbers below were later found to be L2-hot, see 9a):** GPU kernels ~12.4 ms. The GPU0 -> GPU1 hand-off is 1.4 us + 3.6 us = **5 us per token (0.04%)**, so P2P would save < 5 us; the time is in the GEMV kernels. `bench_gemv_fn.py`: 0.2-0.87 TB/s at 1 token, 0.06-0.22 TB/s at 8 tokens (target ~1.5). Causes: lanes read 2-byte-aligned blocks with byte loads, little memory-level parallelism, x re-read per token, short rows (K=320/640) leave lanes idle.

### 9a. Tiled GEMV (done 2026-10-26, measured)

**Result:** plain greedy decode **74.4 -> 137.4 tok/s** (13.3 -> 7.28 ms/token, `bench_fn.py 160`, three runs 137.2-137.5), generated text byte-identical to the original engine. QUANT GATE PASS (13 formats, dequant error exactly 0, GEMV rel err ~1e-7). ORACLE GATE PASS (21/21 top-1, max log-prob diff 0.0000; per-layer residual error unchanged: layer 0 1.02e-5 in both the original and the tiled engine, layers 7-47 2e-7..8e-7). Not reached: the ~3.5-4 ms/token target (see "where the 7.3 ms goes").

**What was built (`engine/faie2fn.cu`):**
- Layout: every weight repacked on the GPU at load into units of [16-row tile][chunk][row][bytes]; chunk = 256 values when the row allows, else 128/64/32 (BF16: 64). Pure byte permutation, rows padded to 16 with zeros; experts tile per expert (640 and 2560 are multiples of 16). `k_repack`, staging buffer freed after load.
- Dense kernel `k_tgemv<TY, CV, NT>`: a warp owns a contiguous chunk range of one tile, stages units with cp.async (double buffered), lane = (row lane&15, half lane>>4); x by `__ldg` float4; halves meet by shfl_xor 16; the wpt warps of a tile reduce in shared memory in warp order; ks blocks per tile leave partials that the *consumer kernel* adds in split order (no reduce kernel, no atomics). Persistent grid (blocks that fit the GPU, cp.async running across tile boundaries) when a weight needs <= 3 GPU-fulls of blocks, one block per tile group beyond that (dynamic balance; the LM head). BF16 (`k_tgemv_bf16`) loads its 16-byte-aligned 128-byte rows straight into registers (in shared memory all 16 rows would hit one bank).
- Decode: one decoder per format `DQ<TY>::run(row, g, f, tab)` that hands values to the dot product in runs of four (the array form `dq32` used by the quant gate is the same code). Small integers become floats without I2F: bits 0x4B000000|q, and scale x q as one exact FMA `fma(d, 2^23+q, -d*2^23)` (= d*q rounded once; Q6_K uses 1.5*2^23 since d*sc*3 is exact). IQ4 codebook in registers via byte_perm; IQ2/IQ3 codebooks copied to shared memory per block; ksigns computed as `i | parity(i)<<7` (checked equal to the table for all 128). 16-bit/32-bit loads (all block sizes are even; Q4_K/Q5_K/IQ4_XS words are 4-byte aligned).
- FMA order written out (`__fmaf_rn` chains per value index mod 4, `__fadd_rn` combines): the 1-token and 8-token instantiations perform the same operations per token, and every split is fixed by the weight's shape, so a token's result does not depend on the batch (lossless speculation stays possible).
- Experts: `k_tmoe_gu` (gate+up fused per (pair, tile), silu*up in the epilogue), `k_tmoe_down` (per (pair, tile)), slot-weighted sum fused into `k_moe_combine` with the shared expert.
- Programmatic dependent launch: GEMV kernels launch early, stage their first weight units (and IQ codebook) while the previous kernel finishes, then `griddepcontrol.wait` before touching activations. Kernels a GEMV follows trigger only after their own loads arrived. A kernel never reads before its wait anything its direct predecessor writes (the shared expert now runs between routing and the routed experts, so the expert kernels can read `sel` early).
- Weights stacked where they read the same input (one GEMV each): HC down + inject, DeltaNet alpha + beta, router + shared-expert gate, indexer q + k. RMSNorm kernel rewritten (8.5 -> 1.6 us), routing as a two-stage parallel rank (same order as repeated argmax). Kernels per token 1646 -> 1502.

**Root causes found on the way (each measured):**
- The old `bench_gemv_fn.py` numbers for weights < ~24 MB were L2 bandwidth, not DRAM: 4 rotating copies fit the 5090's 96 MB L2. The bench now reads one cold copy per call, flushes L2, and replays the calls as one CUDA graph (separate launches measured WDDM launch cost).
- IQ2_XXS/XS/S converted a 64-bit integer to float per value: on GeForce that runs on the FP64 path (1/64 rate) and capped IQ2 at 0.19 TB/s in both old and new kernels. Converting from 32 bits fixed it (4x).
- Wave quantization (e.g. 640 blocks where ~510 fit), register spills at the 128-register cap, branchy x loads in the 8-token path, generic (not shared) codebook loads.

**GEMV, same method for both engines (cold L2, graph replay, `bench_gemv_fn.py 1 8`; original engine rebuilt with this bench, archived as `faie2fn_original_with_cold_bench.cu`):**

| shape | MB | 1 tok original | 1 tok tiled | 8 tok original | 8 tok tiled |
|---|---|---|---|---|---|
| BF16 320x10240 | 6.6 | 0.58 TB/s | 0.92 | 0.23 | 0.59 |
| BF16 10240x320 | 6.6 | 0.55 | 1.32 | 0.27 | 0.71 |
| BF16 512x2560 | 2.6 | 0.38 | 0.58 | 0.17 | 0.41 |
| Q8_0 2560x6144 | 16.7 | 0.99 | 1.20 | 0.22 | 0.46 |
| IQ4_XS 10240x2560 | 13.9 | 0.54 | 1.28 | 0.11 | 0.36 |
| Q6_K 10240x2560 | 21.5 | 0.71 | 1.37 | 0.17 | 0.55 |
| Q4_K 6144x2560 | 8.8 | 0.51 | 0.93 | 0.11 | 0.26 |
| IQ3_S 6144x2560 | 6.8 | 0.42 | 0.77 | 0.09 | 0.19 |
| Q5_K 248320x2560 (LM head) | 437 | 0.70 | **1.59** | 0.16 | 0.76 |
| IQ2_XXS 65536x2560 | 43.3 | 0.19 | 1.07 | 0.06 | 0.25 |
| IQ3_S 65536x2560 | 72.1 | 0.58 | 1.37 | 0.10 | 0.39 |
| Q2_0 65536x640 | 11.8 | 0.33 | 1.07 | 0.06 | 0.27 |
| IQ4_NL 65536x640 | 23.6 | 0.56 | 1.44 | 0.13 | 0.50 |

At 20 MB per format (`eqbytes_fn.py`, 1 token): BF16 1.28, Q8_0 1.25, Q6_K 1.33, IQ4_XS 1.42 TB/s (memory pipeline), Q4_K/Q5_K 1.1, IQ3 1.03-1.05, IQ2 0.87-0.90 TB/s: the IQ2/IQ3 formats are decode-bound at ~2.5-3.5 G weights/ms (~9 instructions per weight left: 2 FMA, byte_perm, sign flip). 8 tokens are bound by x loads and FMA issue (0.2-0.76 TB/s).

**Where the 7.3 ms goes now (`prof_fn_final`, `step_kernels_fn.py`):** a DeltaNet layer is ~125 us under the profiler: qkv + z ~22 us (~1 TB/s in place), ssm_out 13 us, the two hyper-connection reads ~35 us (13 MB BF16 each plus norm/low-rank/mix kernels), routed experts ~18 us (decode-bound IQ2/IQ3), router + routing + shared expert ~20 us, DeltaNet recurrence and small kernels ~16 us. 1502 kernels per token; each small GEMV still pays ~2 us of start-up that PDL only partly hides.

**Next (in order):**
1. Fewer kernels per layer: fold the RMSNorm into the HC GEMV prologue, the low-rank silu into its epilogue, hc_mix into the next GEMV's x read; the DeltaNet core (conv, l2norm, gates, recurrence, gated norm) into one kernel; run the shared expert on a parallel graph branch beside routing/experts. Target ~30 us per layer (~1.4 ms/token).
2. IQ2/IQ3 decode beyond ~3.5 G weights/ms and the 8-token path: tensor cores with the exact decomposition the rules allow (fp16 integer codes x fp32 scales, x split hi+lo fp16), or register blocking (several rows per x load) for 8 tokens.
3. Speculative decoding with replay-commit rollback (FAIE2 phase 3) + drafter (MTP head `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` 4.14 GB needs a download OK, or n-gram drafting). The GEMVs are batch-invariant, so verify = decode bit for bit.
4. Batched prefill (64-token forwards), serving layer (reuse `serve/`), thinking load test + tool suite vs the 27B (178.6 tok/s under load, tools 8/8).

Tools added in `flashnext/`: `sweep_gemv_fn.py`, `sweep_ks1_fn.py`, `ab_waves_fn.py`, `ab_pdl_fn.py`, `eqbytes_fn.py`, `tune_moe_fn.py`, `one_gemv_fn.py`, `step_kernels_fn.py`, `step_path_fn.py`; DLL hooks `fn_gemv_override`, `fn_gemv_plan`, `fn_gemv_waves`, `fn_pdl`, `fn_moe_override` (`fn_use_graphs` now also drops captured graphs so new settings apply).

**P2P on Windows:** driver 610.88 reports GNS and `cudaDeviceCanAccessPeer` = 0 both ways (re-checked 2026-10-26: both cards WDDM, not TCC; topology PXB, so the PCIe path exists and the closed driver blocks it). Without peer access CUDA's own peer copy also stages through host RAM. The Linux unlock (aikitoria open-gpu-kernel-modules, driver 610.57.04, iommu=pt, ACS off) works because that driver is open source; the Windows kernel driver is closed and signed, and patching it means turning off driver-signature enforcement. Not done. Measured cost of not having it: 5 us per token of 7.3 ms.

### 9b. Fewer, overlapped kernels per layer (done 2026-10-26, measured)

**Result:** plain greedy decode **137.4 -> 173.6 tok/s** (7.28 -> 5.76 ms/token; `bench_fn.py 160`, three runs 173.6-173.7; this session's own baseline runs were 136.3-138.1). The generated text is byte-identical to the pre-fusion build. Every change was checked bit for bit against the pre-fusion DLL with `bitcmp_fn.py`: logits of all 54 positions in graph mode (8-token prefill forwards and 1-token decode) and the residual stream after every layer (eager) are **bit-identical**, so the 1-token / 8-token batch invariance carries over unchanged. QUANT GATE PASS (13 formats, dequant error 0). ORACLE GATE PASS (21/21, max log-prob diff 0.0000, per-layer error unchanged: layer 0 1.02e-5). Kernels per token **1502 -> 1071**. Under nsys (completion-time shares, `layer_path_fn.py`): a DeltaNet layer **130.3 -> 101.4 us**, an attention layer **132.9 -> 104.8 us**, step 8.23 -> 6.03 ms. fp32 math throughout; no reduction order changed.

**Steps, each measured (in-process A/B `ab_fn.py`: one model load, configurations alternated over 4 rounds, spread about +-0.5 tok/s):**

| step | what | tok/s |
|---|---|---|
| 0 | tiled GEMV build (9a) | 136.3-138.1 (`bench_fn`) |
| 1 | every forward kernel launched with PDL (small kernels wait at entry, trigger after the wait): their launch overlaps the kernel before | 147.1-148.4 (`bench_fn`) |
| 2 | `k_route`: each of the 160 candidates ranked by 3 threads (7.25 -> 3.66 us alone, `route_fn.py`, 200 random cases incl. forced ties pick the same experts and weights bit for bit); the shared-expert swiglu folded into it; MoE order router -> shared gate, shared up (early, beside it) -> route -> routed gate/up -> shared down (early, beside it) -> routed down | 159.1 (other orders 155.5-156.4) |
| 3 | early GEMVs: `z` and alpha/beta beside `qkv`; `k`, `v`, indexer beside `q` | 162.8 (early off: 155.3) |
| 4 | hyper-connection path: residual write + next RMSNorm in one kernel (`k_hc_add`, one block per stream, k_groupnorm's thread layout); the MoE combine folded into it; the mix moved into the HC-up GEMV epilogue (`tile_mix`; HC-up rows permuted at load, `hc_perm_row`, so a 16-row tile holds the 4 streams of 4 columns); `k_hc_mix` gone | 170.8 |
| 5 | DeltaNet core: conv, l2norm, gates, recurrence, gated norm -> one kernel `k_gdn` (one block per value head, thread = dimension as before); each value head keeps its own copy of its key head's q/k conv state (`GDN_CST`), else the 3 value heads of a key head race on it | 175.2 (A/B), 173.6 (`bench_fn`) |

**Early GEMVs** (`gemv(..., early)`): a GEMV whose input was complete before the kernel before it started reads it without waiting, triggers at once, and calls `griddepcontrol.wait` just before it exits, so its completion still implies its predecessor's (completion stays transitive). It runs beside that kernel and fills its tail. Rule: the kernel before an early chain must not be early itself, and early kernels read only data written before the chain began.

**Tried and dropped (measured, no gain):** loads of every split-K partial issued up front in `gval` (170.7 vs 170.7); the MoE combine as a separate 40-block kernel (170.6 vs 170.7; again 174.5 vs 174.6); `k_gdn` without register spills via a compiler barrier (168 regs, 174.1 vs 175.2: re-reading `ks` costs more than the spills); `k_gdn` loading a token's inputs one token ahead (174.9 vs 175.2, and it changed bits, see below); `k_hc_add` triggering only after its norm sums (174.0 vs 174.7). Not built: folding the low-rank silu into the HC-down epilogue (the last-block reduction it needs costs about what the 1 us kernel does).

**Found on the way:**
- **nvcc moves loads above `griddepcontrol.wait`.** Loads through `const __restrict__` pointers (and `__ldg`) become `LDG.E.CONSTANT`, which nvcc treats as reads of unchanging memory and schedules above the wait's memory clobber. `k_tmoe_gu` read `sel` before its wait (SASS: LDG at 0x08b0, ACQBULK at 0x0900) and routed with the previous layer's experts in graph mode only (eager and PDL-off runs were bit-identical, which is how it was isolated). CUDA's own `cudaGridDependencySynchronize()` is the same asm and has the same hazard. Fix: `after_wait(p)` passes the pointer through an empty asm after the wait, so loads through it depend on it. `pdl_audit_fn.py` lists, per kernel, the loads placed before the wait: now only parameters, weights, codebooks and `k_tmoe_down`'s `sel` (safe by construction: `k_route` is three kernels back).
- **Bit-exactness across refactors depends on FMA contraction.** Moving `k_gdn`'s loads made nvcc contract the conv sum differently (residuals differ from position 3, where the conv state becomes non-zero, rel 4e-7). The sums are now written as the FMAs the original kernels compiled to, read off their SASS: `dot4c` = fma(a3,b3, fma(a2,b2, fma(a0,b0, a1*b1))) (nvcc rounds the second product first) for the conv and the RMSNorm sum of squares; the delta rule as delta = fma(-g, kv, v) * beta, S = fma(g, S, k * delta).
- The 4-block residual kernel first made the build slower (160.7): its split-K partial loads ran one after another, and the mix epilogue fetched `hn` at the end of every tile. Fixed with `gval4` (float4 loads of all splits, then the same adds) and a per-lane `hn` prefetch when the tile starts.

**Where the 5.76 ms goes now (`prof_fusion/final.sqlite`, DeltaNet layer 101 us):** qkv + z 19.6 us (20.7 MB, ~1.05 TB/s in place), routed experts 21 us (gate/up 16.2 us for 9.5 MB = 0.59 TB/s: IQ2 decode and 400 tiny blocks), the four HC GEMVs ~21 us (26.8 MB BF16), ssm_out 9.6 + `k_gdn` 8.6 us, router + shared gate/up 5.1 us, `k_route` 4.7 us, residual writes 2.3 + 5.0 us, low rank 2 x 1 us. Small kernels left per DeltaNet layer: 6 of 20 launches (`k_hc_lowrank` x2, `k_gdn`, `k_hc_add` x2, `k_route`).

**Next (in order):**
1. Routed experts: `k_tmoe_gu` at 0.59 TB/s is the largest gap. A persistent expert kernel (blocks loop over (pair, tile) items with cp.async across items, codebook staged once per block), then the IQ2/IQ3 decode work from 9a's list.
2. Dense GEMVs in place at 0.9-1.05 TB/s (qkv, q) vs 1.3+ in isolation.
3. `k_gdn` 8.6 us on 48 blocks (two 128-long FMA chains per token, 64 KB of state per block): split a head's columns over a 2-block cluster, with the norms' warp sums exchanged through distributed shared memory in the same order.
4. Then speculative decoding (verify = decode bit for bit still holds), batched prefill, serving.

Tools added in `flashnext/`: `bitcmp_fn.py` (record/compare logits + per-layer residuals of two builds; env `FN_PDL=0`, `FN_EARLY=0`, `FN_GRAPHS=0` for bisecting), `ab_fn.py` (in-process A/B, env `FN_DLL`), `route_fn.py`, `pdl_audit_fn.py`, `layer_path_fn.py`; DLL hooks `fn_bench_route`, `fn_early`. Snapshots: `F:\archive\FAIE2_snapshots\fn_pre_fusion_20261026_112305` (before), `fn_fusion_step2_moe_early_*`, `fn_fusion_step3_hc_gdn_*`, `fn_fusion_step4_pinned_*`, `fn_fusion_done_20261026_212134` (final source, DLL, tools, bench outputs, intermediate profiles).

### 9c. Routed experts: 8-row gate/up kernel, and where the rest of their time goes (2026-10-27, measured)

**Result:** `k_tmoe_gu8` (gate/up on 8-row half tiles, 800 blocks instead of 400) is now the default (`fn_moe_var(0)` selects the old `k_tmoe_gu`). **Bit-identical** to the 9b build (`bitcmp_fn.py`: tokens, argmax, logits of all 54 positions in graph mode, residual after all 48 layers), QUANT GATE PASS, ORACLE GATE PASS (21/21, max log-prob diff 0.0000). Speed, in-process A/B (`ab_fn.py old:fn_moe_var=0 new:`, same DLL): **170.1 -> 171.7 tok/s** in the last run; the four A/B runs of this session gave +0.7%, +1.6%, +0.9%, +1.1%, so about +1%. `bench_fn.py 160`: 169.5-170.6 tok/s, last forward 5.63-5.67 ms (5.74-5.75 before). Gate/up critical-path share per layer 16.3 -> ~14.9 us (IQ3_S layers 21.3 -> 19.5 us, IQ2_XXS/XS layers about unchanged). Not the 5-10% hoped for; the measurements below say why.

**What the kernel does:** same arithmetic as `k_tmoe_gu`. Lane = (row 0..7, matrix gate/up, half of the unit's groups), so each lane decodes one matrix; the per-lane FMA sequence, the half merge (`shfl_xor 16`) and the warp merge (warp order) are the old ones, hence the same bits. A 16-row tile unit is split into its two 8-row halves (`8 * CB` bytes each, 16-byte aligned for every format). Reason: 400 blocks spread 3 / 2 over 170 SMs (busiest SM = 1.28x the average); 800 half-tile blocks give 5 / 4 (1.06x). `__launch_bounds__(160, 5)` keeps all 800 blocks resident in one wave (registers 83-89 -> 72; with 85+ registers only 4 blocks fit an SM and 120 blocks waited for a second wave).

**Why "persistent" was not built:** `k_tmoe_gu` already issues every cp.async (both chunks of the warp) right after `sel` is known, in one wave; a persistent loop would only save the codebook staging, which already overlaps the router. The imbalance was the part worth fixing, and finer blocks fix it.

**Where the gate/up time goes** (timing-only variants, wrong results, in-process A/B, 48 layers, per layer):

| variant | tok/s | saves |
|---|---|---|
| gu8 as shipped | 170.8 | - |
| no decode (loads, syncs, store only) | 176.2 | 3.7 us |
| every pair reads expert 0 (weights L2-hot, full decode) | 176.4 | 3.8 us |
| no work at all (launch, wait, `sel`, store) | 185.7 | 9.4 us |
| pairs 10 -> 8 / 10 -> 6 (old kernel) | 175.7 / 179.2 | 4.5 / 6.7 us |

So of ~15 us: ~5.5 us is fixed (kernel boundary, `sel` read, completion) and ~9.4 us is work; removing either DRAM or decode alone saves only ~3.7 us because the two overlap badly; the asymptotic slope is 1.1 us per pair = 0.86 TB/s = about 3 G weights/ms. The inner loop of `k_tmoe_gu<IQ2_XS>` (SASS, 64 weights = 2 matrices x 32): 421 instructions = 6.6 per weight: FFMA 128 (the `mulq` scale and the dot), LOP3 74 and IMAD.SHL 49 (sign flips), PRMT 64, IMAD.U32 24, FADD 16, LDS 20, LDG.128 8. `fma.rn.f32x2` is accepted for sm_120 but ptxas turns it into two FFMA (no packed FP32 on GeForce Blackwell). Cheaper decodes I found either change the arithmetic (scale after the group sum; a float codebook in shared memory; tensor cores with fp16 codes x hi/lo-split x, which also breaks "fp32 math stays") or save under 0.3 instruction per weight (sign masks from a table). Not built; the tensor-core route is the only one with a large payoff and needs Ahmad's decision on the rule.

**L2 prefetch of next-op weights (tried, not kept).** A GEMV reads its weights at 1.0-1.3 TB/s from DRAM and 2.0-2.1 TB/s from L2 for BF16 (`bench_gemv_hot_fn.py`, one copy replayed, no flush: BF16 6.6 MB 6.5 -> 3.0 us, Q8_0 16.7 MB 13.6 -> 7.8, Q6_K 21.5 MB 15.6 -> 11.0; the decode-bound IQ2/IQ3/Q4_K formats gain little), and the DRAM is idle behind the latency-bound kernels (`k_gdn` 8.7 us, `k_route` 4.7, `k_hc_add` 2.2/5.0, the attention kernels), so prefetching `ssm_out` + the second hyper-connection read (23 MB) there looked worth ~7 us per layer. Measured per layer (nsys): `ssm_out` 9.6 -> 6.4 us, second HC down 5.8 -> 3.2, HC up 6.1 -> 4.1 when prefetched, but the way to issue it eats the gain: a separate prefetch node with `cp.async.bulk.prefetch.L2` costs 19.6 us (the kernel cannot complete until the bulk prefetch has drained); with `prefetch.global.L2` per line the node costs 3.2 us and `k_gdn` slows by 1.9 us; issuing from `k_gdn` itself +4.4 us (and 440 us per attention layer when I launched it on a 2-D grid and forgot blockIdx.y); issuing from the `ssm_out` GEMV before its wait (best variant): `k_gdn` +3.8 us, `ssm_out` -1.7, HC reads -1.9, net -0.6 us per layer, A/B +0.2% (HC only) / +1.3% (with its own weights), inside the noise. The prefetch queues ahead of `k_gdn`'s own state loads. Source of every variant: `F:\archive\FAIE2_snapshots\fn_experiments_l2pf_gu8_20261027_000701` (`faie2fn.cu`, `faie2fn_pf_in_gemv_prewait.cu`). Live source has none of it.

**Next (in order):**
1. Kernel boundaries: every dependent kernel pair costs ~2 us (wait return, completion flush); 1071 kernels per token. Fusing neighbours that need a global dependency (a last-block epilogue, or a per-pair counter so `k_tmoe_down` of pair p starts when pair p's 80 gate/up blocks are done) is the largest remaining structural saving.
2. IQ2/IQ3 decode with tensor cores or a scale-after-sum decode: needs a decision (bits change, gates decide).
3. `k_gdn` 8.7 us (2-block cluster per head), dense qkv/q GEMVs at ~1 TB/s in place, then speculative decoding.
4. L2 prefetch only with a cheaper issue path (the prefetch has to run behind the predecessor's own DRAM reads, not in front of them).

Tools added: `bench_gemv_hot_fn.py` (GEMV cold vs L2-hot); DLL hooks `fn_moe_var(v)` (1 = gu8, 0 = old), `fn_bench_hot(on)`; `bench_fn.py` honours env `FN_MOE_VAR`. Snapshots: `fn_pre_persistent_experts_20261026_232040` (before), `fn_experiments_l2pf_gu8_20261027_000701` (experiments).
