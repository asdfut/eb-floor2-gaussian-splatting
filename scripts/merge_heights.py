#!/usr/bin/env python3
"""Merge several sessions of the same place (different sensor heights h1/h2/h3) into one
COLMAP dataset for gsplat.

Within a session, camera poses stay T265 odometry (as prepared by bag_to_colmap.py --prep_only).
Between sessions, each T265 odom frame has its own origin, so every session is registered to the
reference session (the first one, h1 = sensors at floor level) by aligning their static lidar maps:
global FPFH+RANSAC (plus identity as a candidate) -> coarse-to-fine point-to-plane ICP.
The ceiling height (a strong plane every session sees) is used to cross-check the z offsets.

usage: merge_heights.py <out_dir> <session_dir_ref> <session_dir> [<session_dir> ...]
"""
import argparse, os, sys
import numpy as np, cv2, open3d as o3d
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B  # noqa: E402


def reg_cloud(S, voxel):
    """Points above the sensor base only: removes floor-mirror points, keeps walls + ceiling."""
    m = S["lidar_map"]
    m = m[m[:, 2] > float(S["base_z"]) + 0.05]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m.astype(np.float64)))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pc


def ceiling_above_base(S):
    m = S["lidar_map"]
    z = m[:, 2] - float(S["base_z"])
    z = z[(z > 1.0) & (z < 6.0)]
    h, e = np.histogram(z, bins=np.arange(1.0, 6.0, 0.02))
    h = np.convolve(h, [1, 2, 1], "same")
    return float(e[np.argmax(h)] + 0.01)


def register(src_S, ref_S, log):
    v = 0.25
    src, ref = reg_cloud(src_S, v), reg_cloud(ref_S, v)
    feat = lambda pc: o3d.pipelines.registration.compute_fpfh_feature(
        pc, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 5, max_nn=100))
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, ref, feat(src), feat(ref), True, v * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(v * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.9999))
    best = None
    for name, T0 in [("identity", np.eye(4)), ("fpfh-ransac", ransac.transformation)]:
        T = T0.copy()
        for vv, thr in [(0.20, 1.0), (0.10, 0.4), (0.05, 0.15), (0.05, 0.06)]:
            s, r = reg_cloud(src_S, vv), reg_cloud(ref_S, vv)
            res = o3d.pipelines.registration.registration_icp(
                s, r, thr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=thr)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = res.transformation
        log(f"    init={name:12s} fitness={res.fitness:.3f} inlier_rmse={res.inlier_rmse*100:.1f} cm")
        if best is None or res.fitness > best[1].fitness:
            best = (name, res, T)
    return best[2], best[1]


def o3d_cloud(P, voxel):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pc


def tf(P, T):
    return P @ T[:3, :3].T + T[:3, 3]


def split_segments(S):
    return np.split(S["seg_pts"], np.cumsum(S["seg_len"])[:-1])


def yaw_xy_from(T, c):
    """Project a correction to (dx, dy, dyaw) about the segment centroid c (height stays fixed)."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    Rz = R.from_euler("z", yaw).as_matrix()
    t = tf(c[None], T)[0] - Rz @ c
    return np.array([t[0], t[1], yaw])


def delta_mat(d):
    D = np.eye(4)
    D[:3, :3] = R.from_euler("z", d[2]).as_matrix()
    D[:2, 3] = d[:2]
    return D


def drift_corrections(S, T, ref_pc, log, win=1, pre=None):
    """T265 drift correction: ICP every (2*win+1)-segment window of this session (already rigidly
    aligned by T) against the reference map; keep x, y, yaw only; median-smooth over time."""
    segs = split_segments(S)
    bz = float(S["base_z"])
    ts, ds, oks = [], [], []
    for k in range(len(segs)):
        Tk = T if pre is None else pre[k] @ T
        P = np.concatenate(segs[max(0, k - win):k + win + 1]).astype(np.float64)
        P = tf(P[P[:, 2] > bz + 0.05], Tk)
        if len(P) < 2000:
            continue
        src = o3d_cloud(P, 0.08)
        D = np.eye(4)
        for thr in (0.6, 0.25, 0.1):
            res = o3d.pipelines.registration.registration_icp(
                src, ref_pc, thr, D, o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=thr)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
            D = res.transformation
        d = yaw_xy_from(D, P.mean(0))
        ok = res.fitness > 0.3 and np.hypot(*d[:2]) < 3.0 and abs(np.degrees(d[2])) < 10
        ts.append(S["seg_t"][k]); ds.append(d); oks.append(ok)
    ts, ds, oks = np.array(ts), np.array(ds), np.array(oks)
    tv, dv = ts[oks], ds[oks]
    if len(dv) >= 3:  # median filter (3) against single bad matches
        dv = np.array([np.median(dv[max(0, i - 1):i + 2], axis=0) for i in range(len(dv))])
    if len(dv) == 0:
        tv, dv = np.array([ts[0] if len(ts) else 0.0]), np.zeros((1, 3))
    log(f"    drift correction: {oks.sum()}/{len(oks)} segments matched; |dxy| max {np.hypot(dv[:,0],dv[:,1]).max():.2f} m, "
        f"median {np.median(np.hypot(dv[:,0],dv[:,1])):.2f} m; yaw {np.degrees(dv[:,2]).min():+.1f}..{np.degrees(dv[:,2]).max():+.1f} deg")

    def at(t):
        t = np.atleast_1d(t)
        return np.stack([np.interp(t, tv, dv[:, i]) for i in range(3)], 1)
    return at


def floor_points(trajs, floor, radius, grid):
    """Grid points on the floor plane within `radius` (xy) of any trajectory."""
    from scipy.spatial import cKDTree
    xy = np.concatenate([t[0][:, :2] for t in trajs])
    lo, hi = xy.min(0) - radius, xy.max(0) + radius
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], grid), np.arange(lo[1], hi[1], grid))
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    d, _ = cKDTree(xy[::4]).query(g, distance_upper_bound=radius)
    g = g[d < radius]
    return np.column_stack([g, np.full(len(g), floor)])


def topdown(maps, floor, path, res=0.03):
    """R/G/B = session 1/2/3 lidar walls (0.3..2 m above floor); white = all agree."""
    band = [m[(m[:, 2] > floor + 0.3) & (m[:, 2] < floor + 2.0)] for m in maps]
    allp = np.concatenate(band)
    lo, hi = allp[:, :2].min(0) - 1, allp[:, :2].max(0) + 1
    W, H = ((hi - lo) / res).astype(int) + 1
    img = np.zeros((H, W, 3), np.float32)
    for c, b in enumerate(band[:3]):
        u = ((b[:, 0] - lo[0]) / res).astype(int)
        v = H - 1 - ((b[:, 1] - lo[1]) / res).astype(int)
        np.add.at(img[:, :, 2 - c], (v, u), 1)
    cv2.imwrite(path, (255 * np.clip(img / 3, 0, 1)).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("sessions", nargs="+", help="session dirs from bag_to_colmap.py --prep_only; first = reference")
    ap.add_argument("--align_only", action="store_true", help="stop after alignment + debug/topdown_sessions.png")
    ap.add_argument("--rounds", type=int, default=2, help="drift-correction rounds (2+ = leave-one-out refinement)")
    ap.add_argument("--floor_radius", type=float, default=3.0, help="add floor points within this xy distance of the path (0 = off)")
    ap.add_argument("--floor_grid", type=float, default=0.04)
    ap.add_argument("--rigid_only", action="store_true", help="skip per-segment drift correction")
    ap.add_argument("--planar", action="store_true", help="session-to-reference transforms restricted to x, y, z, yaw")
    B.add_args(ap)
    a = ap.parse_args()
    log = B.make_logger(a.out_dir, "merge_log.txt")
    Ss = [np.load(os.path.join(d, "session.npz")) for d in a.sessions]
    tags = [os.path.basename(os.path.normpath(d)).split("_")[-1] for d in a.sessions]  # h1, h2, ...

    ceil = [ceiling_above_base(S) for S in Ss]
    for t, c, S in zip(tags, ceil, Ss):
        log(f"{t}: {len(S['names'])} keyframes, ceiling {c:.2f} m above sensor base")

    # ---- register every session to the reference
    Ts = [np.eye(4)]
    for t, S, c in zip(tags[1:], Ss[1:], ceil[1:]):
        log(f"registering {t} -> {tags[0]}")
        T, res = register(S, Ss[0], log)
        if a.planar:   # all sessions are gravity-levelled and the robot drives on one floor: x, y, yaw, z only
            yaw = R.from_matrix(T[:3, :3]).as_euler("zyx")[0]
            log(f"    planar: dropping roll/pitch {np.round(R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=True)[:2], 2)} deg")
            Tp = np.eye(4); Tp[:3, :3] = R.from_euler("z", yaw).as_matrix(); Tp[:3, 3] = T[:3, 3]; T = Tp
        rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        base_ref = (T @ np.array([0, 0, float(S["base_z"]), 1]))[2] - float(Ss[0]["base_z"])
        log(f"    -> t={T[:3,3].round(3)} rpy={rpy.round(2)} deg; {t} base is {base_ref:+.3f} m vs {tags[0]} "
            f"(ceiling check: {ceil[0]-c:+.3f} m)")
        if res.fitness < 0.3:
            log("    WARNING: low ICP fitness – check debug/topdown_sessions.png")
        Ts.append(T)

    # ---- per-segment drift correction (T265 odometry kept as backbone)
    # round 1: every other session against the reference map;
    # rounds 2..: every session (incl. the reference) against the union of the others (leave-one-out)
    corr = [[] for _ in Ss]  # list of correction functions per session, applied in order

    def full_T(i, times):  # world(ref) <- session i odom, at the given times
        out = np.repeat(Ts[i][None], len(np.atleast_1d(times)), 0)
        for f in corr[i]:
            out = np.einsum("nij,njk->nik", np.array([delta_mat(d) for d in f(times)]), out)
        return out

    def session_map(i, above_base=True):
        S = Ss[i]
        segs = split_segments(S)
        Tk = full_T(i, S["seg_t"])
        bz = float(S["base_z"])
        return np.concatenate([tf(P[P[:, 2] > bz + 0.05] if above_base else P, Tk[k]).astype(np.float32)
                               for k, P in enumerate(segs)])

    if not a.rigid_only:
        ref_pc = o3d_cloud(session_map(0), 0.05)
        for i in range(1, len(Ss)):
            log(f"round 1: refining {tags[i]} against {tags[0]} per {np.median(np.diff(Ss[i]['seg_t'])):.0f}s segment")
            Ti = full_T(i, Ss[i]["seg_t"])
            corr[i].append(drift_corrections(Ss[i], np.eye(4), ref_pc, log, pre=Ti))
        for r in range(2, a.rounds + 1):
            maps_now = [session_map(i) for i in range(len(Ss))]
            new = []
            for i in range(len(Ss)):
                others = o3d_cloud(np.concatenate([m for j, m in enumerate(maps_now) if j != i]), 0.05)
                log(f"round {r}: refining {tags[i]} against the other sessions")
                new.append(drift_corrections(Ss[i], np.eye(4), others, log, pre=full_T(i, Ss[i]["seg_t"])))
            for i in range(len(Ss)):
                corr[i].append(new[i])

    maps, trajs = [], []
    for i, S in enumerate(Ss):
        maps.append(session_map(i, above_base=False).astype(np.float64))
        Tt = full_T(i, S["base_traj_t"])
        btraj = np.einsum("nij,nj->ni", Tt[:, :3, :3], S["base_traj"].astype(np.float64)) + Tt[:, :3, 3]
        trajs.append((btraj, float(np.median(btraj[:, 2]))))
    floor = float(Ss[0]["base_z"])  # reference = h1, sensor base sits on the floor
    os.makedirs(os.path.join(a.out_dir, "debug"), exist_ok=True)
    topdown(maps, floor, os.path.join(a.out_dir, "debug", "topdown_sessions.png"))
    if a.align_only:
        return
    pts = B.voxelize(np.concatenate(maps), a.voxel)
    pts = B.clean_map(pts, floor, trajs, log)
    fl = floor_points(trajs, floor, a.floor_radius, a.floor_grid)
    log(f"  added {len(fl)} floor-plane points (glossy floor is invisible to the lidar; "
        f"occluded ones are dropped by the z-buffer)")
    pts = np.concatenate([pts, fl])

    # ---- images (symlinks), poses, cameras
    img_dir = os.path.join(a.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    names, paths, c2ws, cams, cam_of_img = [], [], [], [], []
    for si, (t, d, S) in enumerate(zip(tags, a.sessions, Ss)):
        K = S["K"]
        ci = next((i for i, (w, h, k) in enumerate(cams)
                   if w == int(S["W"]) and h == int(S["H"]) and np.allclose(k, K, atol=1e-3)), None)
        if ci is None:
            cams.append((int(S["W"]), int(S["H"]), K))
            ci = len(cams) - 1
        Tk = full_T(si, S["kf_times"])
        for n, c2w, Tc in zip(S["names"], S["c2ws"], Tk):
            name = f"{t}_{n}"
            src = os.path.abspath(os.path.join(d, "images", str(n)))
            dst = os.path.join(img_dir, name)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            names.append(name)
            paths.append(src)
            c2ws.append(Tc @ c2w)
            cam_of_img.append(ci)
    log(f"merged: {len(names)} images, {len(cams)} camera model(s), {len(pts)} lidar points before thinning")
    np.savetxt(os.path.join(a.out_dir, "session_transforms.txt"),
               np.concatenate([T.reshape(1, 16) for T in Ts]), header=" ".join(tags) + " (4x4 row-major, -> ref)")
    B.finalize(a.out_dir, paths, names, np.array(c2ws), cams, cam_of_img, pts.astype(np.float32), a, log)


if __name__ == "__main__":
    main()
