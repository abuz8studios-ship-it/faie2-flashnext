"""Critical-path share of each kernel in one decode step: start(next kernel) - start(this kernel), per device,
summed by kernel name over the step (kernels overlap under programmatic dependent launch, so durations double-count).
usage: step_path_fn.py prof.sqlite [steps]"""
import sqlite3, sys
from collections import defaultdict
import numpy as np
con = sqlite3.connect(sys.argv[1])
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, deviceId, gridX, gridY FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
starts = [i for i, r in enumerate(rows) if "k_embed" in names[r[2]]]
tot = defaultdict(list)
walls = []
for a, b in zip(starts[-nsteps - 1:-1], starts[-nsteps:]):
    per = defaultdict(float)
    for dv in (0, 1):
        ks = [r for r in rows[a:b] if r[3] == dv]
        for k0, k1 in zip(ks, ks[1:] + [None]):
            nm = names[k0[2]].split("(")[0].replace("void ", "")
            if "<" in nm:
                nm = nm.split("<")[0] + "<" + nm.split("<")[1].split(",")[0].split(">")[0] + ">"
            dt = ((k1[0] if k1 else k0[1]) - k0[0]) / 1e3
            per[(dv, nm)] += dt
    walls.append((rows[b][0] - rows[a][0]) / 1e6)
    for k, v in per.items():
        tot[k].append(v)
print(f"steps {len(walls)}: wall median {np.median(walls):.2f} ms")
med = sorted(((float(np.median(v)), k) for k, v in tot.items()), reverse=True)
s = sum(m for m, _ in med)
for m, k in med[:32]:
    print(f"  dev{k[0]} {m / 1e3:7.3f} ms {100 * m / s:5.1f}%  {k[1]}")
