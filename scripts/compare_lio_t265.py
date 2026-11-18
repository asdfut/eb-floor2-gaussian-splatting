#!/usr/bin/env python3
"""FAST-LIO (primary) vs T265 (auxiliary) per session: path length ratio, loop closure, and windowed relative-motion
disagreement (5 s windows, translation in the body frame + yaw). Large disagreement = one of them is wrong there;
around T265 relocalisation jumps it is T265. usage (gs env): compare_lio_t265.py [sessions...]"""
import os, re, sys, glob, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R

ROOT = os.path.expanduser("~/gsplat_eb")
sess = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/lio/EB_p*_h*.tum"))


def t265(bag):
    mc = glob.glob(f"{ROOT}/bags/{bag}/*.mcap")[0]; out = []
    with open(mc, "rb") as f:
        for _, ch, msg, dm in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=["/t265/odom"]):
            p, q = dm.pose.pose.position, dm.pose.pose.orientation
            out.append([dm.header.stamp.sec + dm.header.stamp.nanosec * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    return np.array(out)


def interp(A, t):
    i = np.clip(np.searchsorted(A[:, 0], t), 1, len(A) - 1)
    return A[i, 1:4], R.from_quat(A[i, 4:8])


print(f"{'session':10s} {'LIO path':>9s} {'T265 path':>9s} {'ratio':>6s} {'LIO loop':>8s} {'T265 loop':>9s} {'win>0.3m':>8s} {'win>5deg':>8s}  worst windows (s from start: dpos m / dyaw deg)")
for s in sess:
    L = np.loadtxt(f"{ROOT}/lio/{s}.tum"); T = t265(s)
    t0, t1 = max(L[0, 0], T[0, 0]), min(L[-1, 0], T[-1, 0])
    ts = np.arange(t0, t1, 5.0)
    pl, rl = interp(L, ts); pt, rt = interp(T, ts)
    dl = np.einsum("nij,nj->ni", rl[:-1].inv().as_matrix(), np.diff(pl, axis=0))      # body-frame step
    dt = np.einsum("nij,nj->ni", rt[:-1].inv().as_matrix(), np.diff(pt, axis=0))
    dpos = np.linalg.norm(dl[:, :2] - dt[:, :2], axis=1)
    yl = (rl[:-1].inv() * rl[1:]).as_euler("zyx", degrees=True)[:, 0]; yt = (rt[:-1].inv() * rt[1:]).as_euler("zyx", degrees=True)[:, 0]
    dyaw = np.abs((yl - yt + 180) % 360 - 180)
    plen = lambda P: np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()
    worst = np.argsort(-dpos)[:3]
    print(f"{s:10s} {plen(L[:,1:4]):8.1f}m {plen(T[:,1:4]):8.1f}m {plen(T[:,1:4])/plen(L[:,1:4]):6.3f} "
          f"{np.linalg.norm(L[-1,1:3]-L[0,1:3]):7.2f}m {np.linalg.norm(T[-1,1:3]-T[0,1:3]):8.2f}m {(dpos>0.3).sum():5d}/{len(dpos):<3d}"
          f"{(dyaw>5).sum():5d}    " + ", ".join(f"{ts[i]-t0:.0f}s: {dpos[i]:.2f}/{dyaw[i]:.1f}" for i in worst))
