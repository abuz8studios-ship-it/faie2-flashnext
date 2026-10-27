"""Gate: every GGUF format Flash-Next uses dequantizes on the GPU exactly like gguf-py / ggml, and the GEMV matches
float64 math on the dequantized weights. Uses real tensors from the model (a slice of rows of each type)."""

import sys

import numpy as np

sys.path.insert(0, r"E:\ABU\FAIE2\flashnext")
from ref_fn import T, NG, dequant  # noqa: E402  (also puts gguf-py on the path)
from fn import FN  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
fn = FN()
seen = {}
for name, t in list(T.items()) + [(NG.name, NG)]:
    if t.tensor_type.name in ("F32", "F16") or t.tensor_type.name in seen:
        continue
    seen[t.tensor_type.name] = name
rng = np.random.default_rng(0)
ok = True
for tname, name in sorted(seen.items()):
    t = NG if name == NG.name else T[name]
    data = np.asarray(t.data)
    cols = int(t.shape[0])
    rows2d = data.reshape(-1, data.shape[-1])
    pick = rng.choice(rows2d.shape[0], size=min(64, rows2d.shape[0]), replace=False)
    raw = np.ascontiguousarray(rows2d[pick])
    if raw.dtype != np.uint8:
        raw = raw.view(np.uint8)
    ref = dequant(rows2d[pick], t.tensor_type).reshape(len(pick), cols)
    got = fn.dequant(t.tensor_type.value, raw, len(pick), cols)
    dq_err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    x = rng.standard_normal((8, cols)).astype(np.float32)
    yref = x.astype(np.float64) @ ref.T
    y = fn.gemv(t.tensor_type.value, raw, len(pick), cols, x)
    gv_err = np.abs(y - yref).max() / np.abs(yref).max()
    good = dq_err == 0.0 and gv_err < 1e-5
    ok &= good
    print(f"{tname:8s} {name:42s} cols {cols:5d}  dequant max rel err {dq_err:.2e}  gemv(8 tok) rel err {gv_err:.2e}  {'OK' if good else 'FAIL'}")
print("QUANT GATE", "PASS" if ok else "FAIL")
