"""A/B decode speed of launch configurations in one process (one model load), alternating configurations over several
rounds so drift hits all of them alike. Each configuration is a list of (dll function, value) calls; graphs are
re-captured after every switch.
usage: ab_fn.py <config> [<config> ...]   config = name:fn=val,fn=val  (e.g. base:  early0:fn_early=0)"""

import sys
import time

import numpy as np

sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
sys.path.insert(0, r"E:\ABU\FAIE2\serve")
from fn import FN, SH1  # noqa: E402
from faie2_tok import QwenTokenizer  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
configs = []
for a in sys.argv[1:]:
    name, _, rest = a.partition(":")
    calls = [(kv.split("=")[0], int(kv.split("=")[1])) for kv in rest.split(",") if kv]
    configs.append((name, calls))
tok = QwenTokenizer(SH1)
msgs = [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, with a short explanation."}]
ids = tok.encode(tok.render(msgs, add_generation_prompt=True, enable_thinking=False))
import os
fn = FN(os.environ.get("FN_DLL") or None) if os.environ.get("FN_DLL") else FN()
fn.load(max_ctx=8192)
defaults = {}
res = {n: [] for n, _ in configs}
texts = {}
for rnd in range(4):
    for name, calls in configs:
        for f, v in defaults.items():
            getattr(fn.d, f)(v)
        for f, v in calls:
            getattr(fn.d, f)(v)
            defaults.setdefault(f, {"fn_early": 1, "fn_pdl": 1, "fn_moe_var": 1}.get(f, 1))
        fn.d.fn_use_graphs(1)
        fn.reset()
        for p in range(0, len(ids), 8):
            am = fn.forward(ids[p:p + 8])
        nxt = int(am[-1])
        fn.forward([nxt])  # capture the 1-token graph outside the timing
        out = [nxt]
        t0 = time.perf_counter()
        for _ in range(127):
            nxt = int(fn.forward([nxt])[0])
            out.append(nxt)
        dt = time.perf_counter() - t0
        res[name].append(127 / dt)
        texts.setdefault(name, tok.decode(out))
for name, _ in configs:
    v = res[name]
    same = texts[name] == texts[configs[0][0]]
    print(f"{name:12s} median {np.median(v):6.1f} tok/s  (runs {', '.join(f'{x:.1f}' for x in v)})  text {'same' if same else 'DIFFERENT'}")
