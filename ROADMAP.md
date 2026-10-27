# FAIE2-FlashNext — Roadmap

Verified state (Aisha, 2026-10-27): 172.9 tok/s decode, oracle gate 21/21 exact,
quant gate 13/13 exact. See VERDICT.md for the evidence and the honest not-built list.

## Next builds, in order
1. **Server shim (ORDER FILED 2026-10-27)** — `serve/fn_server.py`: OpenAI-compatible
   `/v1/chat/completions` + `/v1/models` + `/health` on port 8310, patterned on the
   FAIE2 (27B) server, registered as the named service `faie2-fn`. This is what lets
   Hermes, OpenClaw, and the agent harness call Flash-Next — they consume endpoints,
   not DLLs. Greedy-first; host-side sampling from returned logits; single-request lock.
2. **Batched prefill** — decode is 172.9 tok/s but prefill is ~111 tok/s in 8-token
   forwards; long agent system prompts need 64-token prefill forwards.
3. **Tool-call parsing** — structured tool calls in chat responses (the 27B server
   already passes an 8/8 tool suite; this line reaches parity).
4. **Speculative decoding for FN** — lossless by construction, same discipline as the
   27B line (verify = decode bit-for-bit).
5. **Kernel-boundary fusion** — the largest remaining structural saving per DESIGN.md.

## The universal direction (FAIE3)
One loader reading GGUF metadata, one runtime, architectures as backends keyed by the
GGUF `general.architecture` string (`qwen4exp` here, `qwen3` in FAIE2). A new model
family = one layer block + one tensor mapping + one oracle gate. First proof target:
serve soup_90_10 (ABUZ8's own Qwen3-arch brain) through the same face.
