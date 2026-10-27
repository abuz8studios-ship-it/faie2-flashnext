"""Probe GGUF storage conventions for the Flash-Next port: norm offsets, ssm_a sign, PLE multipliers."""
import sys
import numpy as np
sys.path.insert(0, r"E:\ABU\FAIE2\tests")
from faie2 import gguf_reader  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")
SH1 = r"E:\ABU\MODELS\llm\qwen3.8-flash-next-abliterated\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
r = gguf_reader(SH1)
T = {t.name: t for t in r.tensors}
for n in ["blk.0.hc_attn_norm.weight", "blk.0.hc_ffn_norm.weight", "blk.0.ssm_norm.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
          "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight", "blk.3.indexer.q_norm.weight", "blk.3.indexer.k_norm.weight",
          "blk.1.ple_norm_key.weight", "blk.1.ple_norm_query.weight", "blk.1.ple_norm_conv.weight", "output_hc_norm.weight"]:
    a = np.asarray(T[n].data, dtype=np.float64)
    print(f"{n:34s} mean {a.mean():+.4f} min {a.min():+.4f} max {a.max():+.4f}")
# PLE multipliers as the HF code derives them
MASK = (1 << 64) - 1
def splitmix(v):
    v = (v + 0x9E3779B97F4A7C15) & MASK
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK
    return (v ^ (v >> 31)) & MASK
half = max(1, (((1 << 63) - 1) // 248320) // 2)
base = 1234 + 10007 * 0
print("HF multipliers:", [2 * (splitmix((base + 0x9E3779B97F4A7C15 * (i + 1)) & MASK) % half) + 1 for i in range(3)])
