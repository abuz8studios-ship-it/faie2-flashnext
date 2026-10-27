"""Sweep the tiled-GEMV launch plan per shape: automatic plan vs forced (wpt, ks, tb) and target warps per SM.
usage: sweep_gemv_fn.py [nt]"""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN

BYTES = {30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 22: (256, 82),
         18: (256, 98), 21: (256, 110), 20: (32, 18), 23: (256, 136), 42: (64, 18)}
NAMES = {30: "BF16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 22: "IQ2_S", 18: "IQ3_XXS",
         21: "IQ3_S", 20: "IQ4_NL", 23: "IQ4_XS", 42: "Q2_0"}

fn = FN()
d = fn.d
d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
d.fn_gemv_override.argtypes = [C.c_int] * 4
d.fn_gemv_plan.argtypes = [C.c_uint32, C.c_int, C.c_int, C.POINTER(C.c_int)]
nt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
shapes = [(30, 320, 10240), (30, 10240, 320), (30, 512, 2560), (30, 48, 2560), (30, 16, 10240), (8, 2560, 6144), (8, 2560, 640),
          (23, 10240, 2560), (14, 10240, 2560), (12, 6144, 2560), (21, 640, 2560), (13, 248320, 2560),
          (16, 65536, 2560), (17, 65536, 2560), (22, 65536, 2560), (18, 65536, 2560), (21, 65536, 2560)]


def run(ty, rows, cols, ov):
    d.fn_gemv_override(*ov)
    try:
        plan = (C.c_int * 7)()
        if d.fn_gemv_plan(ty, rows, cols, plan) != 0:
            return 1e9, [0] * 7
        ms = C.c_float()
        fn.check(d.fn_bench_gemv(ty, rows, cols, nt, 50, C.byref(ms)))
        return ms.value * 1000, list(plan)
    finally:
        d.fn_gemv_override(0, 0, 0, 0)


for ty, rows, cols in shapes:
    qk, b = BYTES[ty]
    mb = rows * cols // qk * b / 1e6
    us, plan = run(ty, rows, cols, (0, 0, 0, 0))
    print(f"{NAMES[ty]:8s} {rows:6d} x {cols:5d} {mb:7.1f} MB  auto cv/wpt/ks/tb/nbuf/tiles/nch={plan}: {us:7.1f} us {mb / 1e3 / us * 1e3:5.2f} TB/s")
    best = None
    for warps in (4, 8, 16, 32, 64):
        us, plan = run(ty, rows, cols, (0, 0, 0, warps))
        if best is None or us < best[0]:
            best = (us, f"warps/SM {warps} plan {plan}")
    for wpt in (1, 2, 4, 8, 16):
        for ks in (1, 2, 4, 8, 16):
            for tb in (0,):
                us, plan = run(ty, rows, cols, (wpt, ks, tb, 16))
                if plan[1] != wpt or plan[2] != ks:
                    continue
                if us < best[0]:
                    best = (us, f"forced plan {plan}")
    print(f"{'':31s} best {best[0]:7.1f} us {mb / 1e3 / best[0] * 1e3:5.2f} TB/s  ({best[1]})", flush=True)
