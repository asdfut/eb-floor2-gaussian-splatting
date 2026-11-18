#!/usr/bin/env python3
"""Visual fine alignment between places (colmap env), after the lidar alignment of build_floor2.py.

Triangulate with the current poses (strict thresholds -> tracks stay within one place where places are still
misaligned), then turn every verified cross-place feature match whose two features belong to 3D points of
*different* places into a 3D-3D correspondence. A smooth per-place deformation (control nodes along the path,
x / y / yaw / z per node, Gaussian blended) is fitted to these, all places simultaneously (gain < 1).
Writes <dataset>/visual_deform.npz with layers that build_floor2.py --extra appends.
"""
import argparse, os, sys, time
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_colmap import read_ours, write_text_model  # noqa: E402

MAXID = 2147483647


def apply_corr(P, nodes, corr, sigma):
    """corr per node: dx, dy, dyaw (about the node), dz. Gaussian blend that fades to 0 away from the nodes."""
    if len(nodes) == 0:
        return P.copy(), np.zeros(len(P))
    d2 = ((P[:, None, :2] - nodes[None]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma ** 2)); ws = w.sum(1)
    w = w / np.maximum(ws, 1e-9)[:, None] * np.clip(ws, 0, 1)[:, None]
    rel = P[:, None, :2] - nodes[None]
    disp = np.stack([-rel[..., 1] * corr[None, :, 2], rel[..., 0] * corr[None, :, 2]], -1) + corr[None, :, :2]
    out = P.copy(); out[:, :2] += (w[..., None] * disp).sum(1); out[:, 2] += w @ corr[:, 3]
    return out, w @ corr[:, 2]


def fit_rigid4(A, B, iters=8, delta=0.1):
    """robust x, y, yaw, z with A -> B, rotation about A's centroid c: returns (c, [dx, dy, dyaw, dz], inlier rms)."""
    w = np.ones(len(A)); c = A[:, :2].mean(0)
    for _ in range(iters):
        ca = (w[:, None] * A).sum(0) / w.sum(); cb = (w[:, None] * B).sum(0) / w.sum()
        a, b = A[:, :2] - ca[:2], B[:, :2] - cb[:2]
        yaw = np.arctan2((w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum(), (w * (a * b).sum(1)).sum())
        Rm = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        t2 = cb[:2] - Rm @ ca[:2]; dz = cb[2] - ca[2]
        pred = np.c_[A[:, :2] @ Rm.T + t2, A[:, 2] + dz]
        r = np.linalg.norm(pred - B, axis=1)
        w = np.where(r < delta, 1.0, delta / np.maximum(r, 1e-9))
    c = (w[:, None] * A[:, :2]).sum(0) / w.sum()
    tn = Rm @ c + t2 - c
    return c, np.array([tn[0], tn[1], yaw, dz]), float(np.sqrt(np.mean(np.minimum(r, delta * 3) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--node", type=float, default=4.0)
    ap.add_argument("--radius", type=float, default=6.0, help="correspondences used per node (m)")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--min_corr", type=int, default=40)
    a = ap.parse_args()
    W = os.path.join(a.dataset, "colmap")
    db_path = os.path.join(W, "database.db")
    logf = open(os.path.join(W, "align_log.txt"), "a")

    def log(s):
        s = f"[{time.strftime('%H:%M:%S')}] {s}"; print(s, flush=True); logf.write(s + "\n"); logf.flush()

    cams, imgs = read_ours(os.path.join(a.dataset, "sparse", "0"))
    out = os.path.join(W, "align_tri")
    write_text_model(out, cams, imgs)
    rec = pycolmap.triangulate_points(pycolmap.Reconstruction(out), db_path, os.path.join(a.dataset, "images"),
                                      out, clear_points=True)
    log(f"triangulated {rec.num_points3D()} points, mean reproj {rec.compute_mean_reprojection_error():.3f}px")

    place_of_img = {iid: rec.image(iid).name.split("_")[0] for iid in rec.reg_image_ids()}
    places = sorted(set(place_of_img.values()))
    pid = {p: i for i, p in enumerate(places)}
    # 3D point -> majority place, xyz, quality
    P3 = {}
    for p3id, pt in rec.points3D.items():
        el = [e.image_id for e in pt.track.elements]
        cnt = np.bincount([pid[place_of_img[i]] for i in el], minlength=len(places))
        if pt.error < 2.0 and len(el) >= 3 and cnt.max() >= 0.8 * len(el):
            P3[p3id] = (int(cnt.argmax()), pt.xyz)
    log(f"{len(P3)} good single-place points")

    db = pycolmap.Database.open(db_path)
    pair_ids, geoms = db.read_two_view_geometries()
    db.close()
    seen = set(); A, Bq, pa_, pb_ = [], [], [], []
    for pair, g in zip(pair_ids, geoms):
        i, j = int(pair // MAXID), int(pair % MAXID)
        if i not in place_of_img or j not in place_of_img or place_of_img[i] == place_of_img[j]:
            continue
        m = np.asarray(g.inlier_matches)
        if len(m) == 0:
            continue
        Ii, Ij = rec.image(i), rec.image(j)
        for k1, k2 in m:
            q1, q2 = Ii.point2D(int(k1)), Ij.point2D(int(k2))
            if not (q1.has_point3D() and q2.has_point3D()):
                continue
            x, y = q1.point3D_id, q2.point3D_id
            if x == y or x not in P3 or y not in P3 or P3[x][0] == P3[y][0] or (x, y) in seen:
                continue
            seen.add((x, y)); seen.add((y, x))
            A.append(P3[x][1]); Bq.append(P3[y][1]); pa_.append(P3[x][0]); pb_.append(P3[y][0])
    A, Bq, pa_, pb_ = np.array(A), np.array(Bq), np.array(pa_), np.array(pb_)
    log(f"{len(A)} cross-place 3D-3D correspondences")
    for i, p in enumerate(places):
        for j, q in enumerate(places):
            if i < j:
                m = ((pa_ == i) & (pb_ == j)) | ((pa_ == j) & (pb_ == i))
                if m.sum():
                    d = np.linalg.norm(A[m] - Bq[m], axis=1)
                    log(f"  {p}-{q}: {m.sum()} corr, distance median {np.median(d):.3f} m, 90% {np.percentile(d, 90):.3f} m")

    # camera paths per place for node placement
    C = {p: [] for p in places}
    for iid in rec.reg_image_ids():
        C[place_of_img[iid]].append(rec.image(iid).projection_center())
    nodes = {}
    for p in places:
        nn = []
        for c in np.array(C[p])[:, :2]:
            if not nn or np.min(np.linalg.norm(np.array(nn) - c, axis=1)) > a.node:
                nn.append(c)
        nodes[p] = np.array(nn)

    # correspondences in both directions: (point of place p, its partner of another place)
    src_place = np.r_[pa_, pb_]; X0 = np.r_[A, Bq]; Y0 = np.r_[Bq, A]; dst_place = np.r_[pb_, pa_]
    layers = {p: [] for p in places}

    def deform_pts(Pts, place_idx):
        out = Pts.copy()
        for i, p in enumerate(places):
            m = place_idx == i
            for nd, cr in layers[p]:
                out[m], _ = apply_corr(out[m], nd, cr, a.sigma)
        return out

    for sw in range(a.sweeps):
        X, Y = deform_pts(X0, src_place), deform_pts(Y0, dst_place)
        d = np.linalg.norm(X - Y, axis=1)
        new = {}
        msg = []
        for i, p in enumerate(places):
            m = src_place == i
            msg.append(f"{p} {np.median(d[m]):.3f}")
            tree = cKDTree(X[m][:, :2]); Xi, Yi = X[m], Y[m]
            nd_, cr_ = [], []
            for n in nodes[p]:
                idx = tree.query_ball_point(n, a.radius)
                if len(idx) < a.min_corr:
                    continue
                c, corr, _ = fit_rigid4(Xi[idx], Yi[idx])
                nd_.append(c); cr_.append(corr * a.gain)
            new[p] = (np.array(nd_).reshape(-1, 2), np.array(cr_).reshape(-1, 4))
        log(f"sweep {sw}: median corr distance " + ", ".join(msg))
        for p in places:
            layers[p].append(new[p])
    X, Y = deform_pts(X0, src_place), deform_pts(Y0, dst_place)
    d = np.linalg.norm(X - Y, axis=1)
    log(f"final: median corr distance {np.median(d):.3f} m, 90% {np.percentile(d, 90):.3f} m")
    np.savez(os.path.join(a.dataset, "visual_deform.npz"),
             **{f"{p}_L{i}_{k}": v for p in places for i, l in enumerate(layers[p]) for k, v in zip(("nodes", "corr"), l)})
    log("saved visual_deform.npz")


if __name__ == "__main__":
    main()
