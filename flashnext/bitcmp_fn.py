"""Bit-exactness check between two builds of faie2fn.dll.

record: run the bench prompt as 8-token forwards, then greedy decode one token per forward, with CUDA graphs (logits of
every position) and again eagerly with the residual stream after every layer; save everything to an .npz.
compare: report whether two recordings are bit-identical, and where they first differ.
usage: bitcmp_fn.py record <out.npz> [dll] [n_decode]
       bitcmp_fn.py compare <a.npz> <b.npz>"""

import sys

import numpy as np

sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
sys.path.insert(0, r"E:\ABU\FAIE2\serve")
sys.stdout.reconfigure(encoding="utf-8")


def record(out, dll, n_dec):
    from fn import FN, SH1, HCD, NL, DLL
    from faie2_tok import QwenTokenizer
    tok = QwenTokenizer(SH1)
    msgs = [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, with a short explanation."}]
    ids = tok.encode(tok.render(msgs, add_generation_prompt=True, enable_thinking=False))
    fn = FN(dll or DLL)
    fn.load(max_ctx=4096)
    import os
    if os.environ.get("FN_PDL") == "0":  # serialize every kernel (bisecting launch-order races)
        fn.d.fn_pdl(0)
        fn.d.fn_use_graphs(1)
    if os.environ.get("FN_MOE_VAR"):
        fn.d.fn_moe_var(int(os.environ["FN_MOE_VAR"]))
        fn.d.fn_use_graphs(1)
    if os.environ.get("FN_GRAPHS") == "0":
        fn.d.fn_use_graphs(0)
    if os.environ.get("FN_EARLY") == "0":
        fn.d.fn_early(0)
        fn.d.fn_use_graphs(1)
    blocks = [ids[p:p + 8] for p in range(0, len(ids), 8)]
    # pass 1: graphs, logits of every position, greedy continuation
    lg, am = [], []
    for b in blocks:
        a, l = fn.forward(b, logits=True)
        lg.append(l); am.append(a)
    nxt = int(am[-1][-1])
    dec = [nxt]
    for _ in range(n_dec):
        a, l = fn.forward([nxt], logits=True)
        lg.append(l); am.append(a)
        nxt = int(a[0]); dec.append(nxt)
    blocks += [[t] for t in dec[:-1]]
    # pass 2: eager, residual stream after every layer, same token blocks
    fn.reset()
    hid = []
    for b in blocks[:len(blocks) - n_dec + 8]:
        buf = np.zeros((NL, len(b), HCD), np.float32)
        fn.debug_layers(buf)
        fn.forward(b)
        hid.append(buf)
    fn.debug_layers(None)
    np.savez(out, logits=np.concatenate(lg), argmax=np.concatenate(am), hidden=np.concatenate(hid, axis=1),
             tokens=np.array(sum(blocks, []), np.int32))
    print(f"recorded {sum(len(b) for b in blocks)} positions ({len(blocks)} forwards) to {out}; decode text: {tok.decode(dec)!r}")


def compare(a, b):
    A, B = np.load(a), np.load(b)
    same = True
    for k in ("tokens", "argmax", "logits", "hidden"):
        x, y = A[k], B[k]
        if k != "hidden" and x.shape[1:] == y.shape[1:]:  # recordings of different lengths: compare the common part
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
        if x.shape != y.shape:
            print(f"{k}: shapes differ {x.shape} vs {y.shape}")
            same = False
            continue
        eq = x.view(np.uint32) == y.view(np.uint32) if x.dtype == np.float32 else x == y
        if eq.all():
            print(f"{k}: bit-identical {x.shape}")
            continue
        same = False
        if k == "hidden":
            bad = [(il, p) for il in range(x.shape[0]) for p in range(x.shape[1]) if not eq[il, p].all()]
            il, p = min(bad, key=lambda q: (q[1], q[0]))
            rel = np.abs(x - y).max() / np.abs(x).max()
            print(f"hidden: {len(bad)} (layer, position) rows differ; first at position {p} layer {il}; max rel diff {rel:.2e}")
        elif k == "logits":
            rows = np.where(~eq.all(-1))[0]
            print(f"logits: {len(rows)} positions differ, first {rows[0]}; max abs diff {np.abs(x - y).max():.3e}")
        else:
            print(f"{k}: {int((~eq).sum())} entries differ, first at {int(np.where(~eq)[0][0])}")
    print("BIT-IDENTICAL" if same else "DIFFERENT")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None,
               int(sys.argv[4]) if len(sys.argv) > 4 else 24)
    else:
        compare(sys.argv[2], sys.argv[3])
