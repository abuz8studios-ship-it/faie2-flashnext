"""Run one bench GEMV shape (for profilers).  usage: one_gemv_fn.py type rows cols nt"""
import ctypes as C, sys
sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from fn import FN
fn = FN()
fn.d.fn_bench_gemv.argtypes = [C.c_uint32, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
ty, rows, cols, nt = (int(a) for a in sys.argv[1:5])
ms = C.c_float()
fn.check(fn.d.fn_bench_gemv(ty, rows, cols, nt, 4, C.byref(ms)))
print(f"{ms.value * 1000:.1f} us")
