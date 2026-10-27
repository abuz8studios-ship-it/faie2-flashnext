"""Routing kernel check: every variant picks the same experts and weights as k_route (variant 0), bit for bit, on
random router outputs with ties forced in, and the time per call of each.
usage: route_fn.py [variants...]"""

import ctypes as C
import sys

import numpy as np

sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN  # noqa: E402

RSG, MAXT, NEU = 528, 8, 10
variants = [int(v) for v in sys.argv[1:]] or [1]
fn = FN()
fn.d.fn_bench_route.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
rng = np.random.default_rng(1)


def run(v, part, nt, ks, iters=1):
    sel = np.zeros(nt * NEU, np.int32)
    w = np.zeros(nt * NEU, np.float32)
    ms = np.zeros(1, np.float32)
    fn.check(fn.d.fn_bench_route(v, nt, ks, part.ctypes.data, iters, sel.ctypes.data, w.ctypes.data, ms.ctypes.data))
    return sel, w, float(ms[0])


ok = True
for trial in range(200):
    ks = int(rng.integers(1, 6))
    nt = int(rng.integers(1, 9))
    part = (rng.standard_normal((ks, MAXT, RSG)) * rng.choice([0.1, 1, 4])).astype(np.float32)
    if trial % 3 == 0:  # ties: copy some logits onto others (one split, so the sums tie exactly)
        part[1:] = 0
        for t in range(nt):
            src = rng.integers(0, 512, 20)
            dst = rng.integers(0, 512, 20)
            part[0, t, dst] = part[0, t, src]
    if trial % 7 == 0:  # many exact ties
        part[:, :, :512] = np.round(part[:, :, :512])
    ref = run(0, part, nt, ks)
    for v in variants:
        got = run(v, part, nt, ks)
        if not (np.array_equal(ref[0], got[0]) and np.array_equal(ref[1].view(np.uint32), got[1].view(np.uint32))):
            ok = False
            print(f"trial {trial} variant {v}: differs (nt {nt}, ks {ks})")
print("ROUTE CHECK", "PASS" if ok else "FAIL", f"(200 random cases, variants {variants})")
part = rng.standard_normal((5, MAXT, RSG)).astype(np.float32)
for v in [0] + variants:
    t1 = min(run(v, part, 1, 5, 200)[2] for _ in range(3))
    t8 = min(run(v, part, 8, 5, 200)[2] for _ in range(3))
    print(f"variant {v}: {t1 * 1e3:.2f} us per call (1 token), {t8 * 1e3:.2f} us (8 tokens)")
