#!/usr/bin/env python3
"""Refine place-to-place transforms (from place_register2d.py) with 3D point-to-plane ICP against the union of the
other places (leave-one-out, a few sweeps). Keeps gravity: only x, y, yaw are taken from ICP (floors share z=0)."""
import numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation as R

D = "data/floor2"
pl = ["p1", "p2", "p3", "p4"]
T2 = None
Ts = dict(np.load(f"{D}/place_T.npz"))  # seeds (place_layout.py)
W = {p: np.load(f"{D}/walls_{p}.npz")["walls"] for p in pl}


def pc(P, v):
    c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); c = c.voxel_down_sample(v)
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30)); return c


def planar(T):
    yaw = R.from_matrix(T[:3, :3]).as_euler("zyx")[0]
    out = np.eye(4); out[:3, :3] = R.from_euler("z", yaw).as_matrix(); out[:2, 3] = T[:2, 3]; return out


for sweep in range(3):
    for p in pl[1:]:
        others = np.concatenate([W[q] @ Ts[q][:3, :3].T + Ts[q][:3, 3] for q in pl if q != p])
        tgt = pc(others, 0.05)
        T = Ts[p]
        for v, d in [(0.2, 1.5), (0.1, 0.6), (0.05, 0.25)]:
            r = o3d.pipelines.registration.registration_icp(
                pc(W[p], v), tgt if v == 0.05 else pc(others, v), d, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=d / 2)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = planar(r.transformation)
        dT = np.linalg.inv(Ts[p]) @ T
        print(f"sweep {sweep} {p}: fitness {r.fitness:.3f} rmse {r.inlier_rmse:.3f}  moved {np.linalg.norm(dT[:2,3]):.2f} m "
              f"{np.degrees(np.arctan2(dT[1,0], dT[0,0])):.2f} deg", flush=True)
        Ts[p] = T
np.savez(f"{D}/place_T.npz", **Ts)
for p in pl:
    print(p, "yaw", round(np.degrees(np.arctan2(Ts[p][1, 0], Ts[p][0, 0])), 2), "t", Ts[p][:3, 3].round(3))
