"""Every format at ~20 MB (2560 columns): does time track bytes (memory pipeline) or format (decode)?"""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN
fn = FN()
d = fn.d
d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
fmts = [(30, "BF16", 5120), (8, "Q8_0", 2720), (14, "Q6_K", 2100), (13, "Q5_K", 1760), (12, "Q4_K", 1440), (23, "IQ4_XS", 1360),
        (21, "IQ3_S", 1100), (18, "IQ3_XXS", 980), (22, "IQ2_S", 820), (17, "IQ2_XS", 740), (16, "IQ2_XXS", 660)]
for ty, nm, rowb in fmts:
    rows = int(20e6 / rowb) // 16 * 16
    line = f"{nm:8s} {rows:6d} x 2560 {rows * rowb / 1e6:5.1f} MB"
    for nt in (1, 8):
        ms = C.c_float()
        fn.check(d.fn_bench_gemv(ty, rows, 2560, nt, 50, C.byref(ms)))
        line += f" | nt={nt} {ms.value * 1000:6.1f} us {rows * rowb / ms.value / 1e9:5.2f} TB/s {rows * 2560 / ms.value / 1e9:5.2f} Tw/s"
    print(line, flush=True)
