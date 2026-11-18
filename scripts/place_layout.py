#!/usr/bin/env python3
"""Place-to-place alignment seeded by the known floor layout (p1 frame): the building is a ~60 x 20 m loop
with self-similar corridors, so global search alone picks symmetric wrong solutions.
Seeds: each place's camera path principal axis/centre -> target corridor (both directions tried),
then multi-scale point-to-plane ICP (x, y, yaw) against the already placed places; best fitness wins."""
import numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation as R

D = "data/floor2"
# target corridor centre (x, y) and direction (deg) in the p1 frame, read off the reference layout
TARGET = {"p2": ((26.5, -10.0), 0.0), "p4": ((26.5, 1.0), 0.0), "p3": ((45.5, -7.0), 90.0)}
ORDER = ["p2", "p4", "p3"]
W = {p: np.load(f"{D}/walls_{p}.npz") for p in ["p1"] + ORDER}


def pc(P, v):
    c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); c = c.voxel_down_sample(v)
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30)); return c


def planar(T):
    yaw = R.from_matrix(T[:3, :3]).as_euler("zyx")[0]
    out = np.eye(4); out[:3, :3] = R.from_euler("z", yaw).as_matrix(); out[:2, 3] = T[:2, 3]; return out


def tf(P, T): return P @ T[:3, :3].T + T[:3, 3]


Ts = {"p1": np.eye(4)}
for p in ORDER:
    cams = W[p]["cams"][:, :2]
    c = cams.mean(0); u, s, vt = np.linalg.svd(cams - c, full_matrices=False); ax = vt[0]
    a0 = np.arctan2(ax[1], ax[0])
    ref = np.concatenate([tf(W[q]["walls"], Ts[q]) for q in Ts])
    tgt = {v: pc(ref, v) for v in (0.2, 0.1, 0.05)}
    res = []
    for flip in (0, np.pi):
        for dyaw in np.radians([-6, -3, 0, 3, 6]):
            yaw = np.radians(TARGET[p][1]) - a0 + flip + dyaw
            T = np.eye(4); T[:3, :3] = R.from_euler("z", yaw).as_matrix()
            T[:2, 3] = np.array(TARGET[p][0]) - T[:2, :2] @ c
            for v, d in [(0.2, 2.5), (0.2, 1.2), (0.1, 0.6), (0.05, 0.25)]:
                r = o3d.pipelines.registration.registration_icp(
                    pc(W[p]["walls"], v), tgt[v], d, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(
                        o3d.pipelines.registration.TukeyLoss(k=d / 2)),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                T = planar(r.transformation)
            cc = tf(np.r_[c, 0][None], T)[0]
            res.append((r.fitness, r.inlier_rmse, flip, np.degrees(dyaw), T, cc))
    res.sort(key=lambda x: -x[0])
    for f, rm, flip, dy, T, cc in res[:4]:
        print(f"  {p} flip {int(np.degrees(flip))} seed {dy:+.0f}: fitness {f:.3f} rmse {rm:.3f} "
              f"yaw {np.degrees(np.arctan2(T[1,0],T[0,0])):.1f} path centre -> {cc[:2].round(1)}")
    best = res[0]
    other = next(x for x in res if x[2] != best[2])
    print(f"{p}: chose flip {int(np.degrees(best[2]))} (fitness {best[0]:.3f} vs flipped {other[0]:.3f})", flush=True)
    Ts[p] = best[4]
np.savez(f"{D}/place_T.npz", **Ts)
