#!/usr/bin/env python3
"""Partition dataset -> depth-supervised dataset (gs env).

* z-buffered projection of the lidar/SfM init cloud into every training image (bag_to_colmap.color_and_tracks)
* init points: only those seen by at least one image (drops lidar-only room interiors behind walls, which no
  image ever supervises and which otherwise stay in the model as grey clutter)
* per-image tracks = visible points at 1.0-25 m (sparse metric depth targets for --depth-loss; near points are
  excluded because the disparity loss over-weights them)
usage: make_depth_dataset.py <src_dataset> <dst_dataset>
"""
import os, sys, numpy as np
from pycolmap import SceneManager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
MIN_D, PER_IMG = float(os.environ.get("MIN_D", "0.5")), 4000
m = SceneManager(os.path.join(src, "sparse", "0") + "/"); m.load_cameras(); m.load_images(); m.load_points3D()
cids = sorted(m.cameras)
cams = [(m.cameras[c].width, m.cameras[c].height,
         np.array([[m.cameras[c].fx, 0, m.cameras[c].cx], [0, m.cameras[c].fy, m.cameras[c].cy], [0, 0, 1]])) for c in cids]
order = sorted(m.images, key=lambda i: m.images[i].name)
names, c2ws, coi = [], [], []
for i in order:
    im = m.images[i]
    T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
    names.append(im.name); c2ws.append(np.linalg.inv(T)); coi.append(cids.index(im.camera_id))
W, H = cams[0][0], cams[0][1]
P = m.points3D.astype(np.float64)
rgb, seen, tracks = B.color_and_tracks(P, c2ws, [cams[c][2] for c in coi], W, H,
                                       [os.path.join(src, "images", n) for n in names], PER_IMG * 3, 0.04, print)
# keep seen points only, re-index, and drop near points from each image's track
keep = np.where(seen)[0]
remap = -np.ones(len(P), np.int64); remap[keep] = np.arange(len(keep))
out_tracks, nd = [], []
for c2w, tr in zip(c2ws, tracks):
    if len(tr) == 0:
        out_tracks.append(np.zeros(0, np.int64)); nd.append(0); continue
    z = (P[tr] - c2w[:3, 3]) @ c2w[:3, 2]
    far = tr[z >= MIN_D][:PER_IMG]
    tr = far if len(far) >= 50 else tr[:PER_IMG]       # very close views: keep the near points too
    out_tracks.append(remap[tr]); nd.append(len(tr))
# the trainer needs >= 1 depth point per image: drop images that see no map point at all
ok = [k for k, t in enumerate(out_tracks) if len(t) > 0]
if len(ok) < len(names):
    print(f"dropping {len(names) - len(ok)} images without any visible map point")
names = [names[k] for k in ok]; c2ws = [c2ws[k] for k in ok]; coi = [coi[k] for k in ok]
out_tracks = [out_tracks[k] for k in ok]; nd = [nd[k] for k in ok]
os.makedirs(os.path.join(dst, "images"), exist_ok=True)
for n in names:
    p = os.path.join(dst, "images", n)
    if not os.path.lexists(p):
        os.symlink(os.path.realpath(os.path.join(src, "images", n)), p)
# keep the original point colours (from the merged lidar/SfM cloud)
B.write_colmap(os.path.join(dst, "sparse", "0"), cams, coi, names, np.array(c2ws), P[keep].astype(np.float32),
               m.point3D_colors[keep].astype(np.uint8), out_tracks)
print(f"{dst}: {len(names)} images, init points {len(keep)}/{len(P)} (seen by an image), depth points per image "
      f"median {int(np.median(nd))}, min {int(np.min(nd))}")
