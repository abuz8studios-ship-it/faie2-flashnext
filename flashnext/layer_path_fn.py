"""Per-layer critical path of decode steps: each kernel's share is end(this kernel) - end(previous kernel) on its device
(kernels overlap under programmatic dependent launch, so durations double-count; a kernel that starts early and waits
is charged only for the time its completion adds). Layers end at the second residual
write (k_hc_add*) of the layer. Prints the median layer time by kind (DeltaNet / attention) and the median share of
each kernel position within a DeltaNet and an attention layer.
usage: layer_path_fn.py prof.sqlite [steps]"""
import sqlite3, sys
from collections import defaultdict
import numpy as np

con = sqlite3.connect(sys.argv[1])
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()


def short(n):
    nm = names[n].split("(")[0].replace("void ", "")
    if "<" in nm:
        nm = nm.split("<")[0] + "<" + nm.split("<")[1].split(",")[0].split(">")[0] + ">"
    return nm


starts = [i for i, r in enumerate(rows) if "k_embed" in names[r[2]]]
kinds = defaultdict(list)          # kind -> [layer time us]
pos = defaultdict(lambda: defaultdict(list))  # kind -> (index, name) -> [share us]
counts = []
walls = []
for a, b in zip(starts[-nsteps - 1:-1], starts[-nsteps:]):
    counts.append(b - a)
    walls.append((rows[b][0] - rows[a][0]) / 1e6)
    for dv in (0, 1):
        ks = [r for r in rows[a:b] if r[3] == dv]
        # layer boundaries: every second k_hc_add* closes a layer
        cur, nadd = [], 0
        layers = []
        for i, r in enumerate(ks):
            cur.append(i)
            if short(r[2]).startswith("k_hc_add"):
                nadd += 1
                if nadd % 2 == 0:
                    layers.append(cur)
                    cur = []
        for L in layers:
            nm = [short(ks[i][2]) for i in L]
            if any("ple" in n for n in nm):
                continue
            kind = ("attn" if any("attn" in n for n in nm) else "gdn", tuple(nm))
            nxt = L[-1] + 1
            if nxt >= len(ks) or L[0] == 0:  # a layer's share starts at the end of the kernel before it
                continue
            kinds[kind].append((ks[L[-1]][1] - ks[L[0] - 1][1]) / 1e3)
            for j, i in enumerate(L):
                pos[kind][(j, nm[j])].append((ks[i][1] - ks[i - 1][1]) / 1e3)
print(f"steps {len(walls)}: wall median {np.median(walls):.3f} ms, kernels per step {int(np.median(counts))}")
tot = sum(sum(v) for v in kinds.values()) / len(walls)
print(f"layers sum {tot / 1e3:.3f} ms per step")
for k0 in ("gdn", "attn"):
    ks_ = [k for k in kinds if k[0] == k0]
    if not ks_:
        continue
    kind = max(ks_, key=lambda k: len(kinds[k]))
    print(f"{k0}: most common kernel sequence, {len(kinds[kind]) // len(walls)} layers/step, median {np.median(kinds[kind]):.1f} us per layer "
          f"(all {k0}: {sum(len(kinds[k]) for k in ks_) // len(walls)} layers, median {np.median([x for k in ks_ for x in kinds[k]]):.1f} us)")
    for (j, n), v in sorted(pos[kind].items()):
        print(f"   {j:2d} {np.median(v):6.2f} us  {n}")
