#!/usr/bin/env python3
"""Split every place of a floor dataset into two partitions along its main corridor axis (gs env).
Image -> partition label (p1a, p1b, ...) written as JSON for subset_dataset.py / merge_partitions.py (PART_MAP).
usage: make_part_map.py <floor_dataset> <out.json>"""
import sys, json, numpy as np
from pycolmap import SceneManager
m = SceneManager(sys.argv[1] + "/sparse/0/"); m.load_images()
names = [m.images[i].name for i in m.images]
C = np.array([(-m.images[i].R().T @ m.images[i].tvec)[:2] for i in m.images])
pl = np.array([n[:2] for n in names]); out = {}
for p in sorted(set(pl)):
    s = pl == p; X = C[s] - C[s].mean(0)
    ax = np.linalg.svd(X, full_matrices=False)[2][0]; t = X @ ax
    cut = np.median(t)
    for n, v in zip(np.array(names)[s], t): out[n] = p + ("a" if v < cut else "b")
    print(f"{p}: axis {np.round(ax, 2)}, length {np.ptp(t):.1f} m, split at the median -> {int((t < cut).sum())} / {int((t >= cut).sum())} images")
json.dump(out, open(sys.argv[2], "w"))
