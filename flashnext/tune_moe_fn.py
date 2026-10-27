"""Decode speed under different routed-expert launch shapes (gate+up wpt/tb, down wpt/tb), one model load.
usage: tune_moe_fn.py [n_decode]"""
import ctypes as C, sys, time
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
sys.path.insert(0, r"E:\ABU\FAIE2\serve")
from fn import FN, SH1
from faie2_tok import QwenTokenizer
n_dec = int(sys.argv[1]) if len(sys.argv) > 1 else 64
tok = QwenTokenizer(SH1)
ids = tok.encode(tok.render([{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, with a short explanation."}],
                            add_generation_prompt=True, enable_thinking=False))
fn = FN()
fn.d.fn_moe_override.argtypes = [C.c_int] * 4
fn.load(max_ctx=8192)

def run():
    fn.reset()
    for p in range(0, len(ids), 8):
        am = fn.forward(ids[p:p + 8])
    nxt = int(am[-1]); out = [nxt]
    fn.forward([nxt])  # warm the 1-token graph
    t0 = time.perf_counter()
    for _ in range(n_dec):
        nxt = int(fn.forward([nxt])[0]); out.append(nxt)
    return n_dec / (time.perf_counter() - t0), out

base = None
for cfg in [(5, 1, 1, 8), (8, 1, 1, 8), (2, 4, 1, 8), (5, 1, 5, 1), (5, 1, 1, 4), (5, 1, 2, 4), (8, 1, 5, 1), (5, 1, 3, 2), (2, 4, 5, 1), (1, 8, 5, 1)]:
    fn.d.fn_moe_override(*cfg)
    fn.check(fn.d.fn_use_graphs(1))
    tps, out = run()
    if base is None: base = out
    print(f"gu wpt {cfg[0]} tb {cfg[1]}  down wpt {cfg[2]} tb {cfg[3]}: {tps:6.1f} tok/s  {'same tokens' if out == base else 'TOKENS DIFFER'}", flush=True)
