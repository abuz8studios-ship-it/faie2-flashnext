"""List Flash-Next GGUF metadata and tensors (layout for the FAIE2 port)."""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"E:\ABU\FAIE2\tests")
from faie2 import gguf_reader  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SH1 = r"E:\ABU\MODELS\llm\qwen3.8-flash-next-abliterated\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
SH2 = SH1.replace("00001-of", "00002-of")
for path in (SH1, SH2):
    r = gguf_reader(path)
    print("=====", path.rsplit("\\", 1)[-1])
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or "tokenizer.ggml.tokens" in k or "merges" in k or "token_type" in k:
            continue
        try:
            v = f.contents()
        except Exception:
            v = "?"
        s = str(v)
        print(f"  {k} = {s[:200]}")
    types = Counter()
    bytes_by = defaultdict(int)
    for t in r.tensors:
        types[t.tensor_type.name] += 1
        bytes_by[t.tensor_type.name] += int(t.n_bytes)
    print("  types:", {k: (types[k], round(bytes_by[k] / 1e9, 2)) for k in types})
    for t in r.tensors:
        n = t.name
        if not n.startswith("blk.") or n.startswith(("blk.0.", "blk.1.", "blk.3.")):
            print(f"  {n:55s} {t.tensor_type.name:8s} {list(t.shape)} {int(t.n_bytes)/1e6:.2f} MB")
    per_layer = defaultdict(int)
    for t in r.tensors:
        if t.name.startswith("blk."):
            per_layer[int(t.name.split(".")[1])] += int(t.n_bytes)
    if per_layer:
        print("  per-layer MB:", {k: round(v / 1e6) for k, v in sorted(per_layer.items())})
