"""A/B the persistent GEMV grid (waves = resident blocks x N, 0 = one block per tile group) on bench shapes."""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN
fn = FN()
d = fn.d
d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
d.fn_gemv_waves.argtypes = [C.c_int]
shapes = [(30, 320, 10240), (30, 10240, 320), (8, 2560, 6144), (23, 10240, 2560), (14, 10240, 2560), (12, 6144, 2560),
          (13, 248320, 2560), (16, 65536, 2560), (21, 65536, 2560)]
for ty, rows, cols in shapes:
    line = f"type {ty:2d} {rows:6d} x {cols:5d}"
    for nt in (1, 8):
        for wv in (0, 1, 2, 4):
            d.fn_gemv_waves(wv)
            ms = C.c_float()
            fn.check(d.fn_bench_gemv(ty, rows, cols, nt, 50, C.byref(ms)))
            line += f" | nt{nt} w{wv} {ms.value * 1000:6.1f}"
    print(line, flush=True)
