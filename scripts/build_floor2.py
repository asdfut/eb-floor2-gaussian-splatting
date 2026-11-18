#!/usr/bin/env python3
"""Merge all places (each already = all heights, BA-refined) into one floor dataset.

1. rigid place->floor transforms from place_icp.py (data/floor2/place_T.npz)
2. non-rigid: a few sweeps of local lidar ICP (x, y, yaw) of each place against the others at control nodes
   along its path, applied as a smooth Gaussian-blended deformation (gain < 1 so places meet in the middle)
3. writes data/EB_floor2_merged: images (p1_h1_00000.jpg ... symlinks), sparse/0 with the deformed BA poses as
   priors (only poses that build_final_dataset would trust) and the deformed lidar/floor + SfM points
usage (gs env): build_floor2.py [--sweeps 3] [--gain 0.5]
"""
import argparse, os, sys
import numpy as np
from pycolmap import SceneManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B  # noqa: E402
import place_local as L  # noqa: E402

ROOT = os.path.expanduser("~/gsplat_eb")
PL = ["p1", "p2", "p3", "p4"]


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--node", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=3.0, help="deformation blending radius (m)")
    ap.add_argument("--voxel", type=float, default=0.04, help="lidar/floor point voxel (m)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "EB_floor2_merged"))
    ap.add_argument("--reuse", action="store_true", help="load the lidar layers from <out>/floor_deform.npz")
    ap.add_argument("--extra", nargs="*", default=[], help="extra deformation layer files (align_places_visual.py)")
    ap.add_argument("--T_file", default="place_T.npz", help="rigid place transforms (in --floor_dir)")
    ap.add_argument("--floor_dir", default="data/floor2")
    ap.add_argument("--merged_fmt", default="data/EB_{p}_merged", help="per-place merged dataset dir")
    ap.add_argument("--strict", action="store_true", help="keep only images whose BA pose moved < 0.3 m and < 3 deg")
    ap.add_argument("--scale_file", default="place_scale.npz", help="along-axis scale file ('' = none)")
    ap.add_argument("--simscale_file", default="", help="per-place xy similarity scale about a point (cx, cy, s)")
    a = ap.parse_args()
    os.chdir(ROOT)
    D = a.floor_dir
    log = B.make_logger(a.out, "merge_log.txt")
    Ts = dict(np.load(f"{D}/{a.T_file}"))
    walls = {p: np.load(f"{D}/walls_{p}.npz") for p in PL}
    layers = {p: [] for p in PL}            # list of (nodes, corr) applied in order
    extra = {p: [] for p in PL}

    def load_layers(path, dst):
        Z = np.load(path)
        for p in PL:
            i = 0
            while f"{p}_L{i}_nodes" in Z:
                dst[p].append((Z[f"{p}_L{i}_nodes"], Z[f"{p}_L{i}_corr"])); i += 1
    for e in a.extra:
        load_layers(e, extra)
        log(f"extra deformation layers from {e}")

    scl = dict(np.load(f"{D}/{a.scale_file}")) if a.scale_file and os.path.exists(f"{D}/{a.scale_file}") else {}
    sim = dict(np.load(f"{D}/{a.simscale_file}")) if a.simscale_file else {}
    for p, v in sim.items():
        log(f"{p}: xy similarity scale x{v[2]:.4f} about ({v[0]:.2f}, {v[1]:.2f}) (lidar junction fit)")
    for p, v in scl.items():
        log(f"{p}: along-axis T265 scale calibration x{v[3]:.4f} ({'xy'[int(v[0])]} axis, reference layout)")

    def deform(p, P):
        P = P @ Ts[p][:3, :3].T + Ts[p][:3, 3]
        if p in scl:                              # along-corridor scale calibration (reference layout)
            k, omin, fmin, b = scl[p]; k = int(k)
            P = P.copy(); P[:, k] = fmin + b * (P[:, k] - omin)
        if p in sim:
            cx, cy, sc = sim[p]
            P = P.copy(); P[:, 0] = cx + sc * (P[:, 0] - cx); P[:, 1] = cy + sc * (P[:, 1] - cy)
        yaw = np.zeros(len(P))
        for nodes, corr in layers[p] + extra[p]:
            P, dy = L.apply_corr(P, nodes, corr, a.sigma)
            yaw += dy
        return P, yaw

    # ---- non-rigid sweeps (all places measured against the current others, then all updated)
    if a.reuse:
        load_layers(os.path.join(a.out, "floor_deform.npz"), layers)
        log("reusing lidar deformation layers")
    for sw in range(0 if a.reuse else a.sweeps):
        Pw = {p: deform(p, walls[p]["walls"])[0] for p in PL}
        Cw = {p: deform(p, walls[p]["cams"])[0] for p in PL}
        new = {}
        for p in PL:
            nodes, corr, ov = local_icp(Pw, Cw, p, a.node)
            new[p] = (nodes, corr * a.gain)
            m = np.linalg.norm(corr[:, :2], axis=1) if len(corr) else np.zeros(1)
            log(f"sweep {sw} {p}: {len(nodes)} nodes, local misalignment median {np.median(m):.2f} m, "
                f"max {m.max():.2f} m, yaw max {np.degrees(np.abs(corr[:, 2])).max() if len(corr) else 0:.2f} deg")
        for p in PL:
            layers[p].append(new[p])

    # ---- assemble the floor dataset
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
    cams, cam_of_img, names, c2ws, pts, rgbs = [], [], [], [], [], []
    for p in PL:
        md = a.merged_fmt.format(p=p)
        m = SceneManager(md + "/sparse/0/"); m.load_cameras(); m.load_images(); m.load_points3D()
        z = np.load(f"{md}/refined.npz")
        nobs, dang, dpos = z["nobs"], z["dang"], z["dpos"]
        good = ((nobs >= 80) & (dang < 3) & (dpos < 0.3)) if a.strict else \
            ((nobs >= 150) | ((nobs >= 80) & (dang < 3) & (dpos < 0.5)))
        cid_map = {}
        for cid in sorted(m.cameras):
            c = m.cameras[cid]
            cams.append((c.width, c.height, np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]])))
            cid_map[cid] = len(cams) - 1
        n_ok = 0
        for n, c2w, g in zip(z["names"], z["c2ws"], good):
            if not g:
                continue
            n = str(n)
            pos, dy = deform(p, c2w[None, :3, 3])
            M = np.eye(4)
            M[:3, :3] = rotz(dy[0]) @ Ts[p][:3, :3] @ c2w[:3, :3]
            M[:3, 3] = pos[0]
            name = f"{p}_{n}"
            os.symlink(os.path.realpath(os.path.join(md, "images", n)), os.path.join(img_dir, name))
            names.append(name); c2ws.append(M); cam_of_img.append(cid_map[m.images[m.name_to_image_id[n]].camera_id])
            n_ok += 1
        lid, _ = deform(p, m.points3D)
        sel = z["sfm_err"] < 2.0
        sfm, _ = deform(p, z["sfm_xyz"][sel])
        pts += [lid, sfm]; rgbs += [m.point3D_colors, z["sfm_rgb"][sel]]
        log(f"{p}: {n_ok}/{len(good)} images, {len(lid)} lidar/floor + {len(sfm)} SfM points")
    P = np.concatenate(pts); C = np.concatenate(rgbs)
    key = np.floor(P / a.voxel).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    P, C = P[keep].astype(np.float32), C[keep].astype(np.uint8)
    B.write_colmap(os.path.join(a.out, "sparse", "0"), cams, cam_of_img, names, np.array(c2ws), P, C,
                   [[] for _ in names])
    log(f"floor2: {len(names)} images, {len(cams)} cameras, {len(P)} points (voxel {a.voxel} m) -> {a.out}")
    np.savez(os.path.join(a.out, "floor_deform.npz"), **{f"{p}_T": Ts[p] for p in PL},
             **{f"{p}_L{i}_{k}": v for p in PL for i, l in enumerate(layers[p]) for k, v in zip(("nodes", "corr"), l)})


def local_icp(Pw, Cw, p, node):
    import open3d as o3d
    from scipy.spatial import cKDTree
    others = np.concatenate([Pw[q] for q in PL if q != p])
    otree = cKDTree(others[:, :2])
    tgt = L.pc(others, 0.05)
    nodes = []
    for c in Cw[p][:, :2]:
        if not nodes or np.min(np.linalg.norm(np.array(nodes) - c, axis=1)) > node:
            nodes.append(c)
    tree = cKDTree(Pw[p][:, :2])
    out_n, out_c, out_o = [], [], []
    for n in nodes:
        idx = tree.query_ball_point(n, 7.0)
        if len(idx) < 2000:
            continue
        src = L.pc(Pw[p][idx], 0.05)
        dd, _ = otree.query(np.asarray(src.points)[:, :2], distance_upper_bound=0.3)
        ov = np.isfinite(dd).mean()
        if ov < 0.3:
            continue
        T = np.eye(4)
        for d in [0.6, 0.25]:
            T = o3d.pipelines.registration.registration_icp(
                src, tgt, d, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=d / 2)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)).transformation
        e = L.xyyaw(T); c, s = np.cos(e[2]), np.sin(e[2])
        tn = (np.array([[c, -s], [s, c]]) @ n + e[:2]) - n
        out_n.append(n); out_c.append([tn[0], tn[1], e[2]]); out_o.append(ov)
    return np.array(out_n).reshape(-1, 2), np.array(out_c).reshape(-1, 3), np.array(out_o)


if __name__ == "__main__":
    main()
