#!/usr/bin/env python3
"""Copy a dataset keeping only images whose names also exist in a reference dataset (gs env).
usage: filter_names.py <src> <ref> <dst>"""
import os, sys, numpy as np
from pycolmap import SceneManager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B
src, ref, dst = sys.argv[1:4]
keep_names = set(os.listdir(os.path.join(ref, "images")))
m = SceneManager(src + "/sparse/0/"); m.load_cameras(); m.load_images(); m.load_points3D()
cids = sorted(m.cameras)
cams = [(m.cameras[c].width, m.cameras[c].height, np.array([[m.cameras[c].fx, 0, m.cameras[c].cx], [0, m.cameras[c].fy, m.cameras[c].cy], [0, 0, 1]])) for c in cids]
names, c2ws, coi = [], [], []
for i in sorted(m.images, key=lambda i: m.images[i].name):
    im = m.images[i]
    if im.name not in keep_names: continue
    T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
    names.append(im.name); c2ws.append(np.linalg.inv(T)); coi.append(cids.index(im.camera_id))
os.makedirs(dst + "/images", exist_ok=True)
for n in names:
    p = os.path.join(dst, "images", n)
    if not os.path.lexists(p): os.symlink(os.path.realpath(os.path.join(src, "images", n)), p)
B.write_colmap(dst + "/sparse/0", cams, coi, names, np.array(c2ws), m.points3D.astype(np.float32), m.point3D_colors.astype(np.uint8), [[] for _ in names])
print(f"{dst}: {len(names)} images (of {len(m.images)}), {len(m.points3D)} points")
