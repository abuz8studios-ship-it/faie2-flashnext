"""FAIE2-FN speed: prefill in 8-token forwards, then greedy decode one token per forward.
usage: bench_fn.py [n_decode] [split]"""

import sys
import time

import numpy as np

sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
sys.path.insert(0, r"E:\ABU\FAIE2\serve")
from fn import FN, SH1  # noqa: E402
from faie2_tok import QwenTokenizer  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
n_dec = int(sys.argv[1]) if len(sys.argv) > 1 else 128
split = int(sys.argv[2]) if len(sys.argv) > 2 else 0
tok = QwenTokenizer(SH1)
msgs = [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, with a short explanation."}]
ids = tok.encode(tok.render(msgs, add_generation_prompt=True, enable_thinking=False))
fn = FN()
fn.load(max_ctx=8192, split=split)
import os  # noqa: E402
if os.environ.get("FN_MOE_VAR"):  # routed-expert kernel variant (A/B)
    fn.d.fn_moe_var(int(os.environ["FN_MOE_VAR"]))
    fn.d.fn_use_graphs(1)
print(f"split {fn.d.fn_split()}, GPU0 {fn.d.fn_dev_gb(0):.1f} GB, GPU1 {fn.d.fn_dev_gb(1):.1f} GB, prompt {len(ids)} tokens", flush=True)
t0 = time.perf_counter()
for p in range(0, len(ids), 8):
    am = fn.forward(ids[p:p + 8])
t_pf = time.perf_counter() - t0
nxt = int(am[-1])
out = [nxt]
t0 = time.perf_counter()
for _ in range(n_dec - 1):
    nxt = int(fn.forward([nxt])[0])
    out.append(nxt)
    if nxt in (tok.eos, tok.im_end):
        break
t_dec = time.perf_counter() - t0
print(f"prefill {len(ids)} tok in {t_pf * 1000:.0f} ms ({len(ids) / t_pf:.0f} tok/s); "
      f"decode {len(out) - 1} tok in {t_dec * 1000:.0f} ms ({(len(out) - 1) / t_dec:.1f} tok/s, last forward {fn.d.fn_last_ms():.2f} ms)")
print(repr(tok.decode(out)))
