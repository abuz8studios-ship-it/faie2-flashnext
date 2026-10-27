"""ctypes wrapper for faie2fn.dll (FAIE2 engine for Qwen3.8-Flash-Next)."""

import ctypes as C
import os

import numpy as np

DLL = r"E:\ABU\FAIE2\engine\faie2fn.dll"
SH1 = r"E:\ABU\MODELS\llm\qwen3.8-flash-next-abliterated\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
SH2 = SH1.replace("00001-of", "00002-of")
VOCAB = 248320
HCD = 10240
NL = 48

P = np.ctypeslib.ndpointer
F32 = P(np.float32, flags="C_CONTIGUOUS")
I32 = P(np.int32, flags="C_CONTIGUOUS")
U8 = P(np.uint8, flags="C_CONTIGUOUS")


class FN:
    def __init__(self, dll=DLL):
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin")
        d = self.d = C.CDLL(dll)
        d.fn_error.restype = C.c_char_p
        d.fn_load.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int]
        d.fn_forward.argtypes = [I32, C.c_int, C.c_void_p, C.c_void_p]
        d.fn_debug_layers.argtypes = [C.c_void_p]
        d.fn_test_dequant.argtypes = [C.c_uint32, U8, C.c_int, C.c_int, F32]
        d.fn_test_gemv.argtypes = [C.c_uint32, U8, C.c_int, C.c_int, F32, C.c_int, F32]
        d.fn_dev_gb.restype = C.c_double
        d.fn_dev_gb.argtypes = [C.c_int]
        d.fn_last_ms.restype = C.c_double

    def check(self, rc):
        if rc != 0:
            raise RuntimeError(self.d.fn_error().decode())

    def load(self, max_ctx=32768, split=0):
        self.check(self.d.fn_load(SH1.encode(), SH2.encode(), max_ctx, split))

    def reset(self):
        self.check(self.d.fn_reset())

    def forward(self, tokens, logits=False):
        t = np.ascontiguousarray(tokens, dtype=np.int32)
        am = np.zeros(len(t), np.int32)
        lg = np.zeros((len(t), VOCAB), np.float32) if logits else None
        self.check(self.d.fn_forward(t, len(t), lg.ctypes.data if logits else None, am.ctypes.data))
        return (am, lg) if logits else am

    def debug_layers(self, buf):
        self.check(self.d.fn_debug_layers(buf.ctypes.data if buf is not None else None))

    def dequant(self, qtype, raw, rows, cols):
        out = np.zeros((rows, cols), np.float32)
        self.check(self.d.fn_test_dequant(int(qtype), np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1), rows, cols, out))
        return out

    def gemv(self, qtype, raw, rows, cols, x):
        nt = x.shape[0]
        y = np.zeros((nt, rows), np.float32)
        self.check(self.d.fn_test_gemv(int(qtype), np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1), rows, cols,
                                       np.ascontiguousarray(x, dtype=np.float32), nt, y))
        return y
