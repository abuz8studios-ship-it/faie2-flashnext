"""A/B programmatic dependent launch on the bench GEMV shapes (50 back-to-back calls in one graph, cold weights)."""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN
fn = FN()
d = fn.d
d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
d.fn_pdl.argtypes = [C.c_int]
BYTES = {30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 22: (256, 82),
         18: (256, 98), 21: (256, 110), 20: (32, 18), 23: (256, 136), 42: (64, 18)}
shapes = [(30, 320, 10240), (30, 10240, 320), (30, 512, 2560), (8, 2560, 6144), (23, 10240, 2560), (14, 10240, 2560),
          (12, 6144, 2560), (21, 6144, 2560), (13, 248320, 2560), (16, 65536, 2560), (21, 65536, 2560), (42, 65536, 640), (20, 65536, 640)]
for ty, rows, cols in shapes:
    qk, b = BYTES[ty]
    mb = rows * cols // qk * b / 1e6
    line = f"type {ty:2d} {rows:6d} x {cols:5d} {mb:6.1f} MB"
    for nt in (1, 8):
        for on in (0, 1):
            d.fn_pdl(on)
            ms = C.c_float()
            fn.check(d.fn_bench_gemv(ty, rows, cols, nt, 50, C.byref(ms)))
            line += f" | nt{nt} pdl{on} {ms.value * 1000:6.1f} us {mb / ms.value / 1e3:5.2f}"
    print(line, flush=True)
