#!/usr/bin/env python3
"""Occupancy grid of observed surfaces (lidar walls/floor/ceiling + SfM points), dilated, for the trainer's
--surface-grid constraint. usage (gs env): make_surface_grid.py <dataset> <out.npz> [voxel=0.05] [dilate_m=0.15]"""
import sys, numpy as np
from scipy.ndimage import binary_dilation
from pycolmap import SceneManager
data, out = sys.argv[1], sys.argv[2]
vox = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
dil = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
Ps = []
for d in data.split(","):                        # several datasets (e.g. all partitions) can be combined
    m = SceneManager(d + "/sparse/0/"); m.load_points3D(); Ps.append(m.points3D)
P = np.concatenate(Ps)
org = P.min(0) - 1.0
idx = np.floor((P - org) / vox).astype(int)
shape = idx.max(0) + 3
g = np.zeros(shape, bool); g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
r = int(round(dil / vox))
st = np.zeros((2 * r + 1,) * 3, bool)
zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]; st[(xx ** 2 + yy ** 2 + zz ** 2) <= r * r] = True
g = binary_dilation(g, structure=st)
np.savez_compressed(out, origin=org.astype(np.float32), voxel=vox, grid=g)
print(f"{out}: grid {g.shape}, {g.sum()} occupied ({g.mean()*100:.1f}%), from {len(P)} points, dilation {dil} m")
