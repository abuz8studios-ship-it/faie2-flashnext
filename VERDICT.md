# FAIE2 — Independent Verification Verdict

**Tester:** Aisha (Muse, ABUZ8 fleet) — not the author. The engine was built by Claude
(Claude Desktop sessions on Pegasus) with Ahmad Odeh.
**Date:** 2026-10-27, ~00:46–01:05 EDT
**Machine:** Pegasus — 2× RTX 5090 32 GB, Ryzen 9 9950X, 128 GB DDR5, Windows 11.

## Artifact under test

- `engine/faie2fn.cu` — 149,150 bytes, modified 2026-10-27 00:12:27
- `engine/faie2fn.dll` — 7,094,784 bytes, built 2026-10-27 00:12:50 (the `k_tmoe_gu8` build, DESIGN.md §9c)
- Model: Qwen3.8-Flash-Next GSQ-RCO abliterated, IQ3_XXS GGUF (2 shards, local disk), split at layer 26 (GPU0 23.3 GB, GPU1 24.7 GB)
- Snapshot archive of everything posted: `faie2-post.zip`, 354,690 bytes, MD5 `cfebc659bea60b58403b130cbdc9f3d5`

## Tests run by the verifier (fresh executions, not the author's logs)

1. **BENCH** — `flashnext/bench_fn.py 160`, ComfyUI-venv Python on Pegasus.
   Prefill 30 tokens in 269 ms (111 tok/s). **Decode: 159 tokens in 920 ms = 172.9 tok/s,
   last forward 5.70 ms.** Generated text identical to the author's recorded baseline output.
   Log: `verification/aisha_bench.log`.
2. **ORACLE GATE** — `flashnext/gate_fn_ref.py` against the captured float64-oracle reference.
   Loaded in 26 s. **Top-1 agreement 21/21; max |log-prob diff| 0.0000 (median 0.0000).**
   Per-layer residual relative error: layer 0 = 1.02e-05, decaying to ~5e-07 by layer 47.
   **ORACLE GATE PASS.** Log: `verification/aisha_oracle_gate.log`.
3. **QUANT GATE** — `flashnext/test_fn_quants.py`. All 13 GGUF formats in the model
   (BF16, Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q2_0):
   **dequant max relative error 0.00e+00 on every format**; 8-token GEMV relative error
   6.3e-08 … 1.8e-07 (bar: 1e-5). **QUANT GATE PASS.** Log: `verification/aisha_quant_gate.log`.
   Disclosure: this gate's reference module (`ref_fn.py`) imports a custom
   `transformers.models.qwen4_exp` build that is not present in any Python environment on
   the machine reachable by the verifier. That import serves only the oracle *forward*
   (used by other commands); the gate itself exercises `dequant` and the tensor tables.
   The verifier stubbed the unused import and ran the gate's logic unmodified.

## Comparison with the author's claims (DESIGN.md §9b–9c, session ledger 2026-10-26)

| Claim | Author's number | Verifier's number | Match |
|---|---|---|---|
| Decode speed, current build | 169.5–173.7 tok/s across the day's runs | **172.9 tok/s** | Yes |
| Oracle gate | 21/21, Δlogprob 0.0000 | 21/21, 0.0000 | Yes |
| Quant gate | PASS, dequant error 0 | PASS, 0.00e+00 | Yes |
| Layer-0 residual error | 1.02e-05 | 1.02e-05 | Yes |

## Verdict

**VERIFIED.** The FAIE2-FN engine, at this snapshot, decodes Qwen3.8-Flash-Next at
~173 tok/s across two RTX 5090s with bit-exact quantization and oracle-matching outputs.
The author's published numbers are accurate.

**Not verified / not built** (author's own list, confirmed against the tree): FAIE2-FN
speculative decoding, batched prefill, FN server load test; the MTP drafter head is not
downloaded. GPU peer access remains blocked by the Windows WDDM driver (~5 µs/token).
The 3.5–4 ms/token target is not reached; current forward is 5.70 ms.
