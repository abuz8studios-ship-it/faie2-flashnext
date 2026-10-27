"""Kernels of one decode step in launch order (device 0, layers 0..3 by default) with durations and the gap before each.
usage: step_kernels_fn.py prof.sqlite [first] [count]"""
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 200
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, deviceId, gridX, gridY, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL "
                   "ORDER BY start").fetchall()
starts = [i for i, r in enumerate(rows) if "k_embed" in names[r[2]]]
a = starts[-3]
prev_end = None
tot = 0.0
for s, e, n, dv, gx, gy, bx in rows[a + first:a + first + count]:
    nm = names[n].split("(")[0].replace("void ", "")
    gap = (s - prev_end) / 1e3 if prev_end else 0.0
    prev_end = e
    tot += (e - s) / 1e3
    print(f"dev{dv} {(e - s) / 1e3:7.2f} us  gap {gap:5.2f}  grid {gx:6d}x{gy:<3d} blk {bx:4d}  {nm[:70]}")
print(f"sum {tot:.1f} us")
