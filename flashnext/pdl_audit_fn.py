"""PDL audit of faie2fn.dll: for every kernel, the global loads the compiler placed before the grid-dependency wait
(ACQBULK = griddepcontrol.wait). A load of data the previous kernel writes must come after the wait; nvcc may hoist
loads through const __restrict__ pointers (LDG.E.CONSTANT) above it.
usage: pdl_audit_fn.py [dll]"""
import re
import subprocess
import sys

dll = sys.argv[1] if len(sys.argv) > 1 else r"E:\ABU\FAIE2\engine\faie2fn.dll"
CUO = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cuobjdump.exe"
FILT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cu++filt.exe"
sass = subprocess.run([CUO, "-sass", dll], capture_output=True, text=True).stdout
funcs = re.split(r"\n\s*Function : ", sass)[1:]
for f in funcs:
    name = f.split("\n", 1)[0].strip()
    dem = subprocess.run([FILT, name], capture_output=True, text=True).stdout.strip()
    if not dem.startswith(("void k_", "k_")) or "k_repack" in dem or "k_dequant" in dem:
        continue
    lines = f.split("\n")
    pre = []
    waited = False
    for ln in lines:
        m = re.search(r"/\*([0-9a-f]{4})\*/\s+(.*?);", ln)
        if not m:
            continue
        ins = m.group(2)
        if "ACQBULK" in ins:
            waited = True
            break
        if re.search(r"\bLD(G|C)?\.|\bLDG\b|\bLD\b", ins) and "LDGSTS" not in ins and "LDS" not in ins.split()[0]:
            pre.append(f"{m.group(1)} {ins}")
    short = dem.split("(")[0].replace("void ", "")
    if not waited:
        print(f"{short:40s} no wait")
    elif pre:
        print(f"{short:40s} {len(pre)} loads before the wait: " + "; ".join(p.split(" ", 1)[1].split(",")[0] for p in pre[:6]))
    else:
        print(f"{short:40s} ok")
