#!/usr/bin/env python3
"""Local (non-rigid) misalignment between places: control nodes every NODE m along each place's camera path;
for each node ICP (x, y, yaw) the place's local cloud (radius RAD) against the other places' cloud.
Saves node corrections; apply with smooth Gaussian interpolation (see apply_corr)."""
import sys, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

D = "data/floor2"; pl = ["p1", "p2", "p3", "p4"]
NODE, RAD = 4.0, 7.0


def pc(P, v):
    c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); c = c.voxel_down_sample(v)
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30)); return c


def xyyaw(T): return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])


def mat(d):
    c, s = np.cos(d[2]), np.sin(d[2]); M = np.eye(4); M[:2, :2] = [[c, -s], [s, c]]; M[:2, 3] = d[:2]; return M


def apply_corr(P, nodes, corr, sigma):
    """P: [N,3] world points; nodes [K,2]; corr [K,3] (dx, dy, dyaw about the node). Gaussian-weighted blend."""
    if len(nodes) == 0: return P.copy(), np.zeros(len(P))
    d2 = ((P[:, None, :2] - nodes[None]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma ** 2)); ws = w.sum(1)
    w = w / np.maximum(ws, 1e-9)[:, None] * np.clip(ws, 0, 1)[:, None]     # fade to 0 away from nodes
    dyaw = w @ corr[:, 2]
    # displacement: rotate about each node then translate, blended
    rel = P[:, None, :2] - nodes[None]
    disp = np.stack([-rel[..., 1] * corr[None, :, 2], rel[..., 0] * corr[None, :, 2]], -1) + corr[None, :, :2]
    out = P.copy(); out[:, :2] += (w[..., None] * disp).sum(1)
    if corr.shape[1] > 3:                         # optional dz (visual alignment layers)
        out[:, 2] += w @ corr[:, 3]
    return out, dyaw


if __name__ == "__main__":
    Ts = dict(np.load(f"{D}/place_T.npz"))
    W = {p: np.load(f"{D}/walls_{p}.npz") for p in pl}
    Pw = {p: W[p]["walls"] @ Ts[p][:3, :3].T + Ts[p][:3, 3] for p in pl}
    Cw = {p: W[p]["cams"] @ Ts[p][:3, :3].T + Ts[p][:3, 3] for p in pl}
    out = {}
    for p in pl:
        others = np.concatenate([Pw[q] for q in pl if q != p]); otree = cKDTree(others[:, :2])
        tgt = pc(others, 0.05)
        # nodes: greedy subsample of camera xy at NODE spacing
        nodes = []
        for c in Cw[p][:, :2]:
            if not nodes or np.min(np.linalg.norm(np.array(nodes) - c, axis=1)) > NODE: nodes.append(c)
        nodes = np.array(nodes); tree = cKDTree(Pw[p][:, :2])
        res = []
        for n in nodes:
            idx = tree.query_ball_point(n, RAD)
            if len(idx) < 2000: res.append((n, None, 0)); continue
            src = pc(Pw[p][idx], 0.05)
            # overlap: fraction of local points with another place within 0.3 m
            dd, _ = otree.query(np.asarray(src.points)[:, :2], distance_upper_bound=0.3)
            ov = np.isfinite(dd).mean()
            if ov < 0.3: res.append((n, None, ov)); continue
            T = np.eye(4)
            for d in [0.6, 0.25]:
                r = o3d.pipelines.registration.registration_icp(src, tgt, d, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(k=d / 2)),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
                T = r.transformation
            # express as rotation about the node + translation
            e = xyyaw(T); c, s = np.cos(e[2]), np.sin(e[2])
            tn = (np.array([[c, -s], [s, c]]) @ n + e[:2]) - n
            res.append((n, np.array([tn[0], tn[1], e[2]]), ov, r.fitness))
        good = [(r[0], r[1]) for r in res if r[1] is not None]
        mags = np.array([np.linalg.norm(g[1][:2]) for g in good]) if good else np.zeros(1)
        yaws = np.degrees(np.abs([g[1][2] for g in good])) if good else np.zeros(1)
        print(f"{p}: {len(nodes)} nodes, {len(good)} with overlap; shift median {np.median(mags):.2f} m, "
              f"90% {np.percentile(mags, 90):.2f} m, max {mags.max():.2f} m; yaw median {np.median(yaws):.2f} deg, max {yaws.max():.2f}", flush=True)
        out[p + "_nodes"] = np.array([g[0] for g in good]).reshape(-1, 2)
        out[p + "_corr"] = np.array([g[1] for g in good]).reshape(-1, 3)
    np.savez(f"{D}/place_local.npz", **out)
