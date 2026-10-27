"""Gate: FAIE2-FN forward vs the official-code float64 oracle (ref_fn.py logits), teacher forced.

Compares the residual stream after every layer and the next-token log-probs at every position.
usage: gate_fn_ref.py <ids.json> <ref.npy> [chunk]"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN, HCD, NL  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ids = json.load(open(sys.argv[1]))
ref_lp = np.load(sys.argv[2])
ref_h = np.load(sys.argv[2].replace(".npy", "_hidden.npy"))
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 8

fn = FN()
t0 = time.time()
fn.load(max_ctx=4096)
print(f"loaded in {time.time() - t0:.0f} s, split at layer {fn.d.fn_split()}, GPU0 {fn.d.fn_dev_gb(0):.1f} GB, "
      f"GPU1 {fn.d.fn_dev_gb(1):.1f} GB", flush=True)

n = len(ids)
hid = np.zeros((NL, n, HCD), np.float32)
lg = np.zeros((n, ref_lp.shape[1]), np.float32)
am = np.zeros(n, np.int32)
for p in range(0, n, chunk):
    blk = ids[p:p + chunk]
    buf = np.zeros((NL, len(blk), HCD), np.float32)
    fn.debug_layers(buf)
    a, l = fn.forward(blk, logits=True)
    hid[:, p:p + len(blk)] = buf
    lg[p:p + len(blk)] = l
    am[p:p + len(blk)] = a
fn.debug_layers(None)

print("layer  max|h_fn - h_ref| / max|h_ref|   (residual stream after each layer)")
for il in range(NL):
    err = np.abs(hid[il] - ref_h[il]).max() / np.abs(ref_h[il]).max()
    if il < 4 or il % 8 == 7 or err > 1e-2:
        print(f"  {il:2d}  {err:.2e}")
lp = lg.astype(np.float64)
lp -= lp.max(-1, keepdims=True)
lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
top_ref = ref_lp.argmax(-1)
agree = int((am == top_ref).sum())
dmax = np.abs(lp - ref_lp).max(-1)
# log-prob error on the oracle's own top token and the realized next token
print(f"top-1 agreement {agree}/{n}; max |logprob diff| per position: median {np.median(dmax):.4f}, max {dmax.max():.4f}")
for i in range(n):
    if am[i] != top_ref[i]:
        r = ref_lp[i]
        print(f"  pos {i}: fn {am[i]} vs ref {top_ref[i]}  ref margin {r[top_ref[i]] - r[am[i]]:.4f} nats")
print("ORACLE GATE", "PASS" if agree == n and dmax.max() < 0.05 else "CHECK")
