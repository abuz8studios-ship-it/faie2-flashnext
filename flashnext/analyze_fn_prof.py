"""Per-kernel GPU time per decode step (median over steps), per device, plus wall time per step."""
import sqlite3, sys
from collections import defaultdict
import numpy as np
con = sqlite3.connect(sys.argv[1])
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, deviceId, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
starts = [i for i, r in enumerate(rows) if "k_embed" in names[r[2]]]
steps = []
walls = []
busy = []
for a, b in zip(starts[-25:-1], starts[-24:]):
    per = defaultdict(float); bz = [0.0, 0.0]
    for s, e, n, dv, gx in rows[a:b]:
        nm = names[n]
        key = nm.split("(")[0].replace("void ", "")
        if "k_gemv<" in nm: key = f"gemv {nm.split('<')[1].split('>')[0]}"
        per[(dv, key)] += (e - s) / 1e6
        bz[dv] += (e - s) / 1e6
    steps.append(per); walls.append((rows[b][0] - rows[a][0]) / 1e6); busy.append(bz)
print(f"steps {len(steps)}: wall median {np.median(walls):.2f} ms, GPU busy dev0 {np.median([b[0] for b in busy]):.2f} ms, dev1 {np.median([b[1] for b in busy]):.2f} ms")
keys = set().union(*steps)
med = {k: float(np.median([s.get(k, 0.0) for s in steps])) for k in keys}
tot = sum(med.values())
for k, v in sorted(med.items(), key=lambda kv: -kv[1])[:30]:
    print(f"  dev{k[0]} {v:7.3f} ms {100 * v / tot:5.1f}%  {k[1]}")
n_launch = np.median([b - a for a, b in zip(starts[-25:-1], starts[-24:])])
print(f"kernels per step: {n_launch:.0f}")
