#!/usr/bin/env python3
"""Subset of a gsplat/colmap dataset: images whose name starts with PREFIX, points within MARGIN m of their cameras."""
import sys, os, numpy as np
from pycolmap import SceneManager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B
src, dst, prefix = sys.argv[1], sys.argv[2], sys.argv[3]; margin = 8.0
expand = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # also take other images within this distance (m) of own cameras
VIS_R = float(os.environ.get("VIS_R", "0")); VIS_DEG = float(os.environ.get("VIS_DEG", "35"))   # + images looking at us
m = SceneManager(src + "/sparse/0/"); m.load_cameras(); m.load_images(); m.load_points3D()
cids = sorted(m.cameras)
cams = [(m.cameras[c].width, m.cameras[c].height, np.array([[m.cameras[c].fx, 0, m.cameras[c].cx], [0, m.cameras[c].fy, m.cameras[c].cy], [0, 0, 1]])) for c in cids]
def c2w_of(im):
    T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
    return np.linalg.inv(T)
own = np.array([c2w_of(m.images[i])[:2, 3] for i in m.images if m.images[i].name.startswith(prefix)])
from scipy.spatial import cKDTree
tree = cKDTree(own)
names, c2ws, coi = [], [], []
for i in sorted(m.images, key=lambda i: m.images[i].name):
    im = m.images[i]
    if not im.name.startswith(prefix):
        T_ = c2w_of(im); cpos = T_[:2, 3]
        d_, j_ = tree.query(cpos)
        near = expand > 0 and d_ <= expand
        # visibility: within VIS_R of this partition's cameras and looking towards them (forward within VIS_DEG)
        seen = False
        if VIS_R > 0 and d_ <= VIS_R:
            idx_ = tree.query_ball_point(cpos, VIS_R)
            dirs = own[idx_] - cpos; dist = np.linalg.norm(dirs, axis=1); ok = dist > 1.0
            if ok.any():
                fwd = T_[:2, 2] / max(np.linalg.norm(T_[:2, 2]), 1e-9)
                cosang = (dirs[ok] / dist[ok, None]) @ fwd
                seen = (cosang > np.cos(np.radians(VIS_DEG))).mean() > 0.2
        if not (near or seen): continue
    T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
    names.append(im.name); c2ws.append(np.linalg.inv(T)); coi.append(cids.index(im.camera_id))
c2ws = np.array(c2ws); C = c2ws[:, :3, 3]
lo, hi = C.min(0) - margin, C.max(0) + margin
P = m.points3D.copy(); keep = ((P >= lo) & (P <= hi)).all(1)
if len(sys.argv) > 6:   # move the synthetic floor plane (placed at z=FROM, wrongly at sensor-base height) to z=TO
    zf, zt = float(sys.argv[5]), float(sys.argv[6])
    fl = np.abs(P[:, 2] - zf) < 0.006
    P[fl, 2] = zt
    print(f"moved {int((fl & keep).sum())} synthetic floor points from z={zf} to z={zt}")
os.makedirs(dst + "/images", exist_ok=True)
for n in names:
    os.symlink(os.path.realpath(os.path.join(src, "images", n)), os.path.join(dst, "images", n))
B.write_colmap(dst + "/sparse/0", cams, coi, names, c2ws, P[keep].astype(np.float32), m.point3D_colors[keep].astype(np.uint8), [[] for _ in names])
print(dst, len(names), "images", keep.sum(), "points")
