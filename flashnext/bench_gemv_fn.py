"""Achieved DRAM bandwidth of the FAIE2-FN GEMV per format and shape (random bytes, 4 rotating copies)."""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN
fn = FN()
fn.d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
BYTES = {30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 22: (256, 82),
         18: (256, 98), 21: (256, 110), 20: (32, 18), 23: (256, 136), 42: (64, 18)}
NAMES = {30: "BF16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 22: "IQ2_S", 18: "IQ3_XXS",
         21: "IQ3_S", 20: "IQ4_NL", 23: "IQ4_XS", 42: "Q2_0"}
shapes = [(30, 320, 10240), (30, 10240, 320), (30, 512, 2560), (8, 2560, 6144), (23, 10240, 2560), (14, 10240, 2560),
          (12, 6144, 2560), (21, 6144, 2560), (13, 248320, 2560), (16, 65536, 2560), (21, 65536, 2560), (42, 65536, 640),
          (20, 65536, 640)]
nts = [int(a) for a in sys.argv[1:]] or [1, 8]
for ty, rows, cols in shapes:
    qk, b = BYTES[ty]
    mb = rows * cols // qk * b / 1e6
    line = f"{NAMES[ty]:8s} {rows:6d} x {cols:5d} {mb:8.1f} MB"
    for nt in nts:
        ms = C.c_float()
        fn.check(fn.d.fn_bench_gemv(ty, rows, cols, nt, 50, C.byref(ms)))
        line += f" | nt={nt}: {ms.value * 1000:7.1f} us {mb / 1e3 / ms.value:6.2f} TB/s"
    print(line, flush=True)
