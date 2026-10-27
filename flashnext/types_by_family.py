import sys, re
from collections import defaultdict
sys.path.insert(0, r"E:\ABU\FAIE2\tests")
from faie2 import gguf_reader
SH1 = r"E:\ABU\MODELS\llm\qwen3.8-flash-next-abliterated\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
r = gguf_reader(SH1)
fam = defaultdict(lambda: defaultdict(int))
for t in r.tensors:
    f = re.sub(r"^blk\.\d+\.", "", t.name)
    fam[f][t.tensor_type.name] += 1
for f in sorted(fam):
    print(f"{f:34s} {dict(fam[f])}")
