#!/usr/bin/env python3
"""T265 scale from lidar map sharpness: shift each 10 s lidar segment map by (s-1) * (T265 position at the segment
time) and score how well segments agree (fraction of points within 6 cm of other segments' points).
The true scale makes walls seen from different places coincide. usage: scale_sharp.py SESSION_DIR"""
import sys, numpy as np
from scipy.spatial import cKDTree

S = np.load(sys.argv[1] + "/session.npz")
segs = np.split(S["seg_pts"], np.cumsum(S["seg_len"])[:-1])
tt, bt = S["base_traj_t"], S["base_traj"]
pos = np.stack([np.interp(S["seg_t"], tt, bt[:, i]) for i in range(2)], 1)
segs = [s[(s[:, 2] > S["base_z"] + 0.3) & (s[:, 2] < S["base_z"] + 2.0)][::2, :2] for s in segs]
o = pos.mean(0)


def score(sc):
    P = [s + (sc - 1) * (p - o) for s, p in zip(segs, pos)]
    tot, n = 0.0, 0
    for i in range(len(P)):
        far = [j for j in range(len(P)) if abs(j - i) > 1]
        if not far:
            continue
        tree = cKDTree(np.concatenate([P[j] for j in far]))
        d, _ = tree.query(P[i], distance_upper_bound=0.06)
        tot += np.isfinite(d).sum(); n += len(P[i])
    return tot / n


grid = np.arange(0.84, 1.061, 0.02)
sc = np.array([score(s) for s in grid])
b = grid[sc.argmax()]
fine = np.arange(b - 0.02, b + 0.0201, 0.005)
sf = np.array([score(s) for s in fine])
print(sys.argv[1].split("/")[-1], "coarse:", " ".join(f"{g:.2f}:{v:.3f}" for g, v in zip(grid, sc)))
print("   best scale (true/T265) %.3f  (score %.3f vs %.3f at 1.00)" % (fine[sf.argmax()], sf.max(), sc[np.argmin(abs(grid - 1))]))
