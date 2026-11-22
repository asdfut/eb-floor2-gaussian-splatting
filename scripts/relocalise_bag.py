#!/usr/bin/env python3
"""Re-localise a drifting FAST-LIO run against the existing 2F lidar map (floor frame of data/EB_liod_floor).
gs env. usage: relocalise_bag.py <bag_dir> <lio.tum> <out.tum> [--init x y yaw_deg z] [--plot out.png]
Per 1 s submap (deskewed with the LIO poses): point-to-plane Gauss-Newton against the floor map with a prior on the
LIO relative motion (keeps degenerate corridor directions on the odometry). Writes FAST-LIO-style TUM (IMU body pose)
in the floor frame, so bag_to_colmap.py --odom_dir <dir of out.tum> yields floor-frame cameras directly."""
import argparse, os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/gsplat/examples'))
import bag_to_colmap as B
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument('bag'); ap.add_argument('tum'); ap.add_argument('out')
ap.add_argument('--map', default=os.path.expanduser('~/gsplat_eb/data/EB_liod_floor/sparse/0/'))
ap.add_argument('--init', nargs=4, type=float, default=None)
ap.add_argument('--sub_s', type=float, default=1.0)
ap.add_argument('--sig_pt', type=float, default=0.05)
ap.add_argument('--sig_rot_deg', type=float, default=1.0)
ap.add_argument('--sig_tr', type=float, default=0.03)
ap.add_argument('--plot', default=None)
ap.add_argument('--recover_below', type=float, default=0.7)
ap.add_argument('--search_deg', type=float, default=45.0)
a = ap.parse_args()

def T_of(q, t): T = np.eye(4); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = t; return T
def exp_se3(xi):   # xi = (w, v), left perturbation
    T = np.eye(4); T[:3, :3] = R.from_rotvec(xi[:3]).as_matrix(); T[:3, 3] = xi[3:]; return T
def log_se3(T): return np.concatenate([R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])

# ---- map: floor-frame lidar points (drop floor/synthetic floor), normals
from pycolmap import SceneManager
m = SceneManager(a.map); m.load_points3D(); M = m.points3D.astype(np.float64)
M = M[M[:, 2] > 0.05]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(M)).voxel_down_sample(0.05)
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
MP, MN = np.asarray(pc.points), np.asarray(pc.normals); tree = cKDTree(MP)
print(f"map {len(MP)} pts (5 cm)", flush=True)

# ---- LIO poses + deskewed scans
L = np.loadtxt(a.tum); lt = L[:, 0]
Ts = np.array([T_of(q, t) for q, t in zip(L[:, 4:8], L[:, 1:4])])
sl = Slerp(lt, R.from_quat(L[:, 4:8]))
def lio_at(ts):
    ts = np.clip(ts, lt[0], lt[-1]); T = np.tile(np.eye(4), (len(ts), 1, 1))
    T[:, :3, :3] = sl(ts).as_matrix(); T[:, :3, 3] = np.column_stack([np.interp(ts, lt, L[:, i]) for i in (1, 2, 3)]); return T
t_imu_lidar = np.array([-0.011, -0.02329, 0.04412])
scans = []
with open(os.path.join(a.bag, sorted(f for f in os.listdir(a.bag) if f.endswith('.mcap'))[0]), 'rb') as f:
    for sch, ch, msg in make_reader(f).iter_messages(topics=["/livox/lidar"]):
        tb, xyz, off, tag = B.parse_livox(msg.data)
        r = np.linalg.norm(xyz, axis=1); k = (r > 0.6) & (r < 20); xyz, off = xyz[k][::2], off[k][::2]
        ts = tb / 1e9 + off / 1e9
        if ts.min() < lt[0] or ts.max() > lt[-1]: continue
        sel = np.linspace(0, len(ts) - 1, min(len(ts), 400)).astype(int)    # pose per point subsample, interp rest
        Tp = lio_at(ts[sel]); idx = np.searchsorted(sel, np.arange(len(ts))).clip(0, len(sel) - 1)
        p = xyz.astype(np.float64) + t_imu_lidar
        w = np.einsum('nij,nj->ni', Tp[idx, :3, :3], p) + Tp[idx, :3, 3]
        scans.append((ts.mean(), w))
print(f"{len(scans)} scans", flush=True)
st = np.array([s[0] for s in scans])
centers = np.arange(st[0] + a.sub_s / 2, st[-1], a.sub_s)
subs = []
for tc in centers:
    k = np.where(np.abs(st - tc) <= a.sub_s / 2)[0]
    if len(k) < 3: continue
    W = np.concatenate([scans[i][1] for i in k]); Tc = lio_at(np.array([tc]))[0]
    loc = (W - Tc[:3, 3]) @ Tc[:3, :3]
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(loc)).voxel_down_sample(0.1)
    loc = np.asarray(p.points)
    if len(loc) > 6000: loc = loc[np.random.default_rng(0).choice(len(loc), 6000, replace=False)]
    subs.append((tc, Tc, loc))
print(f"{len(subs)} submaps of {a.sub_s} s", flush=True)

# ---- initial pose: 2D wall-band correlation of the first 10 s, or --init
def gn(X, loc, Xpred=None, iters=12, dmax=(1.0, 0.6, 0.4, 0.3, 0.2, 0.15)):
    Hp = None
    if Xpred is not None:
        Hp = np.diag([1 / np.radians(a.sig_rot_deg) ** 2] * 3 + [1 / a.sig_tr ** 2] * 3)
    for it in range(iters):
        d_ = dmax[min(it, len(dmax) - 1)]
        q = loc @ X[:3, :3].T + X[:3, 3]; dist, j = tree.query(q, distance_upper_bound=d_)
        ok = np.isfinite(dist); qq, n, mp = q[ok], MN[j[ok]], MP[j[ok]]
        r = np.einsum('ij,ij->i', qq - mp, n)
        w = np.minimum(1.0, (2 * a.sig_pt) / np.maximum(np.abs(r), 1e-9)) / a.sig_pt ** 2   # Huber
        J = np.hstack([np.cross(qq, n), n])
        H = (J * w[:, None]).T @ J; b = -(J * w[:, None]).T @ r
        if Hp is not None:
            e = log_se3(X @ np.linalg.inv(Xpred)); H = H + Hp; b = b - Hp @ e
        dx = np.linalg.solve(H + 1e-6 * np.eye(6), b); X = exp_se3(dx) @ X
        if np.linalg.norm(dx) < 1e-5: break
    q = loc @ X[:3, :3].T + X[:3, 3]; dist, _ = tree.query(q)
    ev = np.linalg.eigvalsh((J * w[:, None]).T @ J) if ok.sum() > 10 else np.zeros(6)
    return X, float(np.mean(dist < 0.15)), float(np.median(dist)), ev

if a.init:
    x, y, yaw, z = a.init; X0 = np.eye(4); X0[:3, :3] = R.from_euler('z', yaw, degrees=True).as_matrix(); X0[:3, 3] = [x, y, z]
    X0 = X0 @ np.linalg.inv(subs[0][1]) @ subs[0][1]
else:
    T0 = subs[0][1]; Wl = []
    for tc, Tc, loc in subs[:10]:
        Wl.append((loc @ Tc[:3, :3].T + Tc[:3, 3] - T0[:3, 3]) @ T0[:3, :3])     # in first-submap frame
    Wl = np.concatenate(Wl); band = Wl[(Wl[:, 2] > -1.4) & (Wl[:, 2] < -0.2), :2]
    Mb = MP[(MP[:, 2] > 0.6) & (MP[:, 2] < 1.8), :2]; res = 0.1; org = Mb.min(0) - 2
    shape = tuple(((Mb.max(0) + 2 - org) / res).astype(int))
    def raster(xy, o, s):
        g = np.zeros(s, np.float32); ij = ((xy - o) / res).astype(int)
        k = (ij >= 0).all(1) & (ij[:, 0] < s[0]) & (ij[:, 1] < s[1]); g[ij[k, 0], ij[k, 1]] = 1; return g
    gm = raster(Mb, org, shape); cands = []
    for yaw in np.arange(0, 360, 1.0):
        Rm = R.from_euler('z', yaw, degrees=True).as_matrix()[:2, :2]; qb = band @ Rm.T; qo = qb.min(0)
        qs = tuple(((qb.max(0) - qo) / res).astype(int) + 1); cc = fftconvolve(gm, raster(qb, qo, qs)[::-1, ::-1], mode='full')
        k = np.unravel_index(np.argmax(cc), cc.shape); cands.append((cc[k], yaw, org + (np.array(k) - np.array(qs) + 1) * res - qo))
    cands.sort(key=lambda c: -c[0]); best = None
    for sc, yaw, xy in cands[:8]:      # verify top candidates with full 3D GN over the first 10 s
        for z in (1.6, 1.9, 2.2):
            X = np.eye(4); X[:3, :3] = R.from_euler('z', yaw, degrees=True).as_matrix(); X[:2, 3] = xy; X[2, 3] = z
            Xr, inl, med, _ = gn(X, Wl[::3])
            if best is None or inl > best[1]: best = (Xr, inl, med, yaw)
    print(f"init: yaw {best[3]:.0f}, inliers {best[1]:.2f}, median {best[2]:.3f} m", flush=True)
    X0 = best[0]          # pose of first-submap frame in floor frame
# ---- sequential tracking
out, stats = [], []; nrec = [0]
X = X0 @ np.eye(4)
Xprev, Tprev = None, None
for k, (tc, Tc, loc) in enumerate(subs):
    Xpred = X0 if k == 0 else Xprev @ np.linalg.inv(Tprev) @ Tc
    if k == 0: Xpred = X0          # X0 maps first submap frame (= LIO pose Tc of subs[0]) to floor
    X, inl, med, ev = gn(Xpred, loc, Xpred)
    if inl < a.recover_below:          # lost: search yaw / xy around the prediction (rotation about the sensor), no prior
        best = (X, inl, med, ev); c = Xpred[:3, 3].copy()
        for dyaw in np.arange(-a.search_deg, a.search_deg + 1e-6, 3.0):
            Rz = R.from_euler('z', dyaw, degrees=True).as_matrix()
            for dx in (-1.0, 0.0, 1.0):
                for dy in (-1.0, 0.0, 1.0):
                    X1 = Xpred.copy(); X1[:3, :3] = Rz @ Xpred[:3, :3]; X1[:3, 3] = c + [dx, dy, 0]
                    q = loc[::4] @ X1[:3, :3].T + X1[:3, 3]; d1, _ = tree.query(q, distance_upper_bound=0.5)
                    if np.mean(np.isfinite(d1)) < best[1] * 0.8: continue
                    Xr, i1, m1, e1 = gn(X1, loc[::2], None, iters=10)
                    if i1 > best[1]: best = (Xr, i1, m1, e1)
        if best[1] > inl + 0.05:
            X, inl, med, ev = gn(best[0], loc, None, iters=6); nrec[0] += 1
    out.append((tc, X @ np.linalg.inv(Tc))); stats.append((tc, inl, med, ev[0] / max(ev[-1], 1e-9)))
    Xprev, Tprev = X, Tc
    if k % 10 == 0 or inl < a.recover_below: print(f"  {tc - subs[0][0]:6.1f}s  inliers {inl:.2f}  median {med:.3f} m  cond {stats[-1][3]:.1e}", flush=True)
# ---- corrected TUM: C(t) @ T_lio(t), C interpolated between submap centres
ct = np.array([o[0] for o in out]); Cs = np.array([o[1] for o in out])
cs = Slerp(ct, R.from_matrix(Cs[:, :3, :3])); tt = np.clip(lt, ct[0], ct[-1])
C = np.tile(np.eye(4), (len(lt), 1, 1)); C[:, :3, :3] = cs(tt).as_matrix()
C[:, :3, 3] = np.column_stack([np.interp(tt, ct, Cs[:, i, 3]) for i in range(3)])
Tf = C @ Ts
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'w') as f:
    for t, T in zip(lt, Tf):
        f.write(f"{t:.9f} {' '.join(f'{x:.6f}' for x in T[:3, 3])} {' '.join(f'{x:.8f}' for x in R.from_matrix(T[:3, :3]).as_quat())}\n")
S = np.array(stats)
summ = {"submaps": len(S), "inlier_15cm_median": round(float(np.median(S[:, 1])), 3), "inlier_15cm_p10": round(float(np.percentile(S[:, 1], 10)), 3),
        "resid_median_m": round(float(np.median(S[:, 2])), 3), "z_range": [round(float(Tf[:, 2, 3].min()), 2), round(float(Tf[:, 2, 3].max()), 2)],
        "recoveries": nrec[0], "path_m": round(float(np.linalg.norm(np.diff(Tf[:, :2, 3], axis=0), axis=1).sum()), 1)}
print(json.dumps(summ)); np.save(os.path.splitext(a.out)[0] + '_stats.npy', S)
if a.plot:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [3, 1]})
    Mb = MP[(MP[:, 2] > 0.6) & (MP[:, 2] < 1.8)]; ax[0].scatter(Mb[::15, 0], Mb[::15, 1], s=0.2, c='0.7')
    ax[0].plot(Tf[:, 0, 3], Tf[:, 1, 3], 'r-', lw=1.5); ax[0].plot(*Tf[0, :2, 3], 'go', ms=8); ax[0].plot(*Tf[-1, :2, 3], 'ks', ms=8)
    ax[0].set_aspect('equal'); ax[0].set_title('h4 relocalised (red), start green, end black')
    ax[1].plot(S[:, 0] - S[0, 0], S[:, 1], label='inliers <15cm'); ax[1].plot(S[:, 0] - S[0, 0], S[:, 2], label='median resid (m)')
    ax[1].legend(); ax[1].set_xlabel('s'); fig.tight_layout(); fig.savefig(a.plot, dpi=80)
