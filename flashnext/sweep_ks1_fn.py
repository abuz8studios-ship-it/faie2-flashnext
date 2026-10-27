"""Small-shape GEMVs: best plan with no split-K (one kernel) vs best plan overall (+ reduce kernel).
usage: sweep_ks1_fn.py [nt]"""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN

fn = FN()
d = fn.d
d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
d.fn_gemv_override.argtypes = [C.c_int] * 4
d.fn_gemv_plan.argtypes = [C.c_uint32, C.c_int, C.c_int, C.POINTER(C.c_int)]
nt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# (type, rows, cols, what)
shapes = [(30, 336, 10240, "hc down+inject"), (30, 10240, 320, "hc up"), (30, 528, 2560, "router+shexp gate"),
          (30, 96, 2560, "alpha+beta"), (30, 640, 2560, "indexer q+k"), (8, 2560, 6144, "ssm_out / attn o"),
          (8, 2560, 640, "shexp down"), (23, 640, 2560, "shexp gate/up IQ4_XS"), (21, 640, 2560, "shexp up IQ3_S"),
          (14, 512, 2560, "attn k/v Q6_K"), (23, 6144, 2560, "attn z IQ4_XS"), (14, 10240, 2560, "qkv Q6_K"),
          (23, 10240, 2560, "qkv IQ4_XS"), (12, 10240, 2560, "qkv Q4_K"), (14, 12288, 2560, "attn q Q6_K"),
          (30, 10240, 2560, "ple key"), (13, 248320, 2560, "lm head")]


def run(ty, rows, cols, ov):
    d.fn_gemv_override(*ov)
    try:
        plan = (C.c_int * 7)()
        if d.fn_gemv_plan(ty, rows, cols, plan) != 0:
            return 1e9, None
        ms = C.c_float()
        fn.check(d.fn_bench_gemv(ty, rows, cols, nt, 50, C.byref(ms)))
        return ms.value * 1000, list(plan)
    finally:
        d.fn_gemv_override(0, 0, 0, 0)


for ty, rows, cols, what in shapes:
    auto = run(ty, rows, cols, (0, 0, 0, 0))
    b1 = b = (1e9, None)
    for wpt in (1, 2, 3, 4, 5, 6, 8):
        for ks in (1, 2, 3, 4, 6, 8, 12, 16, 20):
            for tb in (1, 2, 4, 8):
                if tb * wpt > 8:
                    continue
                r = run(ty, rows, cols, (wpt, ks, tb, 16))
                if r[1] is None or r[1][1] != wpt or r[1][2] != ks or r[1][3] != tb:
                    continue
                if ks == 1 and r[0] < b1[0]:
                    b1 = r
                if r[0] < b[0]:
                    b = r
    print(f"{what:22s} {rows:6d} x {cols:5d}  auto {auto[0]:6.1f} us {auto[1]}  | ks=1 best {b1[0]:6.1f} us {b1[1]}"
          f"  | best {b[0]:6.1f} us {b[1]}", flush=True)
