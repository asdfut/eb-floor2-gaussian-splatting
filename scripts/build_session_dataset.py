#!/usr/bin/env python3
"""COLMAP dataset from ONE floor-frame session (bag_to_colmap.py --odom_dir lio/floor, e.g. a whole-floor re-recording).
gs env.
  stage "init":    <session_dir> <out_dir> --prefix n
                   images <prefix>_<tag>_<name>, lidar map (cleaned) + synthetic floor plane at z=--floor_synth within
                   --floor_radius of the path (subset_dataset.py later moves it to each place's real floor, like v2).
  stage "refined": <init_dir> <out_dir> --refined
                   applies <init_dir>/refined.npz (refine_colmap.py, colmap env): BA poses, strict image filter
                   (nobs>=80, rot<3 deg, pos<0.3 m, same as build_floor2.py --strict), adds SfM points (err<2 px).
usage: build_session_dataset.py init <session_dir> <out_dir> [--prefix n --tag h3]
       build_session_dataset.py refined <init_dir> <out_dir>
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/gsplat/examples"))
import bag_to_colmap as B

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["init", "refined"])
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--prefix", default="n"); ap.add_argument("--tag", default="h3")
ap.add_argument("--floor_real", type=float, default=-0.19, help="real floor z in the floor frame (for mirror cleanup)")
ap.add_argument("--floor_synth", type=float, default=-0.0252, help="z of the synthetic floor plane (v2 convention)")
ap.add_argument("--floor_radius", type=float, default=3.0)
ap.add_argument("--floor_grid", type=float, default=0.04)
B.add_args(ap)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
log = B.make_logger(a.out, "build_log.txt")

if a.stage == "init":
    S = np.load(os.path.join(a.src, "session.npz"))
    traj = S["base_traj"].astype(np.float64)
    log(f"session {a.src}: {len(S['names'])} keyframes, base z median {np.median(traj[:, 2]):.2f}, "
        f"cam z median {np.median(S['c2ws'][:, 2, 3]):.2f}")
    pts = B.voxelize(S["lidar_map"].astype(np.float64), a.voxel)
    pts = B.clean_map(pts, a.floor_real, [(traj, float(np.median(traj[:, 2])))], log)
    pts = pts[pts[:, 2] > a.floor_real + 0.03]          # drop the real floor/mirror: replaced by the synthetic plane
    from scipy.spatial import cKDTree
    xy = traj[:, :2]; lo, hi = xy.min(0) - a.floor_radius, xy.max(0) + a.floor_radius
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], a.floor_grid), np.arange(lo[1], hi[1], a.floor_grid))
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    d, _ = cKDTree(xy[::4]).query(g, distance_upper_bound=a.floor_radius); g = g[d < a.floor_radius]
    fl = np.column_stack([g, np.full(len(g), a.floor_synth)])
    log(f"  {len(pts)} lidar points + {len(fl)} synthetic floor points at z={a.floor_synth}")
    pts = np.concatenate([pts, fl]).astype(np.float32)
    img_dir = os.path.join(a.out, "images"); os.makedirs(img_dir, exist_ok=True)
    names, paths = [], []
    for n in S["names"]:
        name = f"{a.prefix}_{a.tag}_{n}"; src = os.path.abspath(os.path.join(a.src, "images", str(n)))
        dst = os.path.join(img_dir, name)
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(src, dst); names.append(name); paths.append(src)
    cams = [(int(S["W"]), int(S["H"]), S["K"])]
    B.finalize(a.out, paths, names, S["c2ws"].astype(np.float64), cams, [0] * len(names), pts, a, log)
else:
    from pycolmap import SceneManager
    m = SceneManager(a.src + "/sparse/0/"); m.load_cameras(); m.load_images(); m.load_points3D()
    z = np.load(os.path.join(a.src, "refined.npz"))
    good = (z["nobs"] >= 80) & (z["dang"] < 3) & (z["dpos"] < 0.3)
    log(f"refined: keep {good.sum()}/{len(good)} images; mount fix {np.round(z['mount_fix_deg'], 2)} deg; "
        f"pose change median {np.median(z['dpos'])*100:.1f} cm / {np.median(z['dang']):.2f} deg")
    cid = sorted(m.cameras)[0]; c = m.cameras[cid]
    cams = [(c.width, c.height, np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]]))]
    img_dir = os.path.join(a.out, "images"); os.makedirs(img_dir, exist_ok=True)
    names, c2ws = [], []
    for n, c2w, g in zip(z["names"], z["c2ws"], good):
        if not g: continue
        n = str(n); dst = os.path.join(img_dir, n)
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(a.src, "images", n)), dst)
        names.append(n); c2ws.append(c2w)
    sel = z["sfm_err"] < 2.0
    P = np.concatenate([m.points3D, z["sfm_xyz"][sel]]); C = np.concatenate([m.point3D_colors, z["sfm_rgb"][sel]])
    key = np.floor(P / 0.04).astype(np.int64); _, keep = np.unique(key, axis=0, return_index=True)
    P, C = P[keep].astype(np.float32), C[keep].astype(np.uint8)
    B.write_colmap(os.path.join(a.out, "sparse", "0"), cams, [0] * len(names), names, np.array(c2ws), P, C,
                   [[] for _ in names])
    log(f"done -> {a.out}: {len(names)} images, {len(P)} points ({len(m.points3D)} lidar/floor + {sel.sum()} SfM, 4 cm dedup)")
