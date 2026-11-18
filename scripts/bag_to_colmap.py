#!/usr/bin/env python3
"""ROS2 bag (ZED2 + T265 + Livox Mid-360) -> COLMAP-format dataset for gsplat.

Camera poses come from T265 odometry (not SfM):
  T_odom_cam(t) = T_odom_t265(t) * T_t265_base * T_base_zedlink * (ZED static TF chain) -> left optical
T265 fixes applied:
  * relocalization jumps: the jump offset is spread back over the path since the previous
    anchor (linear in path length), so the pre-jump trajectory agrees with the relocalized map
  * planar: the robot drives on a flat floor, so T265 z drift is removed (z := 0), roll/pitch kept
Init point cloud: Livox scans de-skewed with the same poses, voxel-downsampled, colored and
given visibility tracks (-> gsplat depth loss) via z-buffered projection into keyframes.

Usage: python bag_to_colmap.py <bag_dir> <out_dir> [options]
"""
import argparse, glob, json, os, struct, sys
import numpy as np, cv2, torch, yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R, Slerp

IMG_TOPIC = "/zed/zed_node/rgb/color/rect/image/compressed"
INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
ODOM_TOPIC = "/t265/odom"
LIDAR_TOPIC = "/livox/lidar"


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def tf_mat(t, q):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def parse_livox(buf):
    """Raw CDR of livox_ros_driver2/CustomMsg -> (timebase_ns, Nx3 xyz, N offset_ns, N tag)."""
    o = 4 + 8  # encapsulation + stamp
    flen = struct.unpack_from("<I", buf, o)[0]
    o += 4 + flen
    o = (o - 4 + 7) // 8 * 8 + 4  # align 8 (relative to payload start)
    timebase = struct.unpack_from("<Q", buf, o)[0]
    o += 8 + 4 + 1 + 3  # timebase, point_num, lidar_id, rsvd
    o = (o - 4 + 3) // 4 * 4 + 4
    n = struct.unpack_from("<I", buf, o)[0]
    o += 4
    dt = np.dtype([("t", "<u4"), ("xyz", "<f4", 3), ("refl", "u1"), ("tag", "u1"), ("line", "u1"), ("pad", "u1")])
    raw = bytes(buf[o:o + n * 20]).ljust(n * 20, b"\0")
    p = np.frombuffer(raw, dt, n)
    return timebase, p["xyz"].astype(np.float64), p["t"].astype(np.int64), p["tag"]


def read_bag(path, want_lidar):
    odom, imgs, info, static = [], [], None, {}
    scans = []
    topics = [IMG_TOPIC, INFO_TOPIC, ODOM_TOPIC, "/tf_static"] + ([LIDAR_TOPIC] if want_lidar else [])
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        dec = DecoderFactory()
        decoders = {}
        for sch, ch, msg in reader.iter_messages(topics=topics):
            if ch.topic == LIDAR_TOPIC:
                tb, xyz, off, tag = parse_livox(msg.data)
                r = np.linalg.norm(xyz, axis=1)
                keep = (r > 0.3) & (r < 40.0)  # <0.2 m = robot body
                scans.append((tb * 1e-9, xyz[keep].astype(np.float32), (off[keep] * 1e-9).astype(np.float32)))
                continue
            if ch.id not in decoders:
                decoders[ch.id] = dec.decoder_for(ch.message_encoding, sch)
            ros = decoders[ch.id](msg.data)
            if ch.topic == ODOM_TOPIC:
                p, q = ros.pose.pose.position, ros.pose.pose.orientation
                odom.append([stamp(ros.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            elif ch.topic == IMG_TOPIC:
                imgs.append((stamp(ros.header), bytes(ros.data)))
            elif ch.topic == INFO_TOPIC and info is None:
                info = dict(w=ros.width, h=ros.height, K=list(ros.k), D=list(ros.d), frame=ros.header.frame_id)
            elif ch.topic == "/tf_static":
                for tr in ros.transforms:
                    t, q = tr.transform.translation, tr.transform.rotation
                    static[(tr.header.frame_id, tr.child_frame_id)] = tf_mat([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
    odom = np.array(odom)
    odom = odom[np.argsort(odom[:, 0])]
    imgs.sort(key=lambda x: x[0])
    return odom, imgs, info, static, scans


def lookup(static, parent, child):
    """Compose static transforms parent->child (T_parent_child) by BFS over the undirected tree."""
    edges = {}
    for (a, b), T in static.items():
        edges.setdefault(a, []).append((b, T))
        edges.setdefault(b, []).append((a, np.linalg.inv(T)))
    stack, seen = [(parent, np.eye(4))], {parent}
    while stack:
        fr, T = stack.pop()
        if fr == child:
            return T
        for nb, Tn in edges.get(fr, []):
            if nb not in seen:
                seen.add(nb)
                stack.append((nb, T @ Tn))
    raise KeyError(f"no static TF path {parent} -> {child}")


def fix_trajectory(odom, jump_m, planar, log, cluster_s=3.0):
    """Spread relocalization jumps back over the path, then optionally clamp z.

    Jumps closer than `cluster_s` seconds are one relocalization event: their summed offset is
    ramped (linear in path length) over the path since the previous event; frames inside the
    event get the offsets of the later jumps in it (they are dropped by --jump_guard anyway).
    """
    t, p, q = odom[:, 0], odom[:, 1:4].copy(), odom[:, 4:8]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.maximum(np.diff(t), 1e-4)
    jumps = np.where((step > jump_m) & (step / dt > 3.0))[0]  # jump between i and i+1
    seglen = np.concatenate([[0], np.cumsum(np.where(np.isin(np.arange(len(step)), jumps), 0, step))])
    events = []
    for j in jumps:
        if events and t[j] - t[events[-1][-1]] < cluster_s:
            events[-1].append(j)
        else:
            events.append([j])
    anchor = 0
    for ev in events:
        deltas = [p[j + 1] - p[j] for j in ev]
        total = np.sum(deltas, axis=0)
        j0 = ev[0]
        L = seglen[j0] - seglen[anchor]
        w = (seglen[anchor:j0 + 1] - seglen[anchor]) / L if L > 1e-6 else np.ones(j0 + 1 - anchor)
        p[anchor:j0 + 1] += w[:, None] * total[None]
        for k in range(1, len(ev)):  # frames between jumps inside the event
            p[ev[k - 1] + 1:ev[k] + 1] += np.sum(deltas[k:], axis=0)
        log(f"  reloc event at t={t[j0+1]-t[0]:.1f}s ({len(ev)} jumps): dxy={np.linalg.norm(total[:2]):.2f} m "
            f"dz={total[2]:+.2f} m -> spread over {L:.1f} m of path")
        anchor = ev[-1] + 1
    if planar:
        log(f"  planar: removing z drift (range was {p[:,2].min():+.2f}..{p[:,2].max():+.2f} m)")
        p[:, 2] = 0.0
    jump_times = t[jumps + 1] if len(jumps) else np.array([])
    return np.column_stack([t, p, q]), jump_times


class PoseInterp:
    def __init__(self, traj):
        self.t = traj[:, 0]
        self.p = traj[:, 1:4]
        self.slerp = Slerp(self.t, R.from_quat(traj[:, 4:8]))

    def __call__(self, ts):
        ts = np.clip(np.atleast_1d(ts), self.t[0], self.t[-1])
        P = np.stack([np.interp(ts, self.t, self.p[:, k]) for k in range(3)], 1)
        T = np.tile(np.eye(4), (len(ts), 1, 1))
        T[:, :3, :3] = self.slerp(ts).as_matrix()
        T[:, :3, 3] = P
        return T


def sharpness(jpg):
    g = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return cv2.Laplacian(g, cv2.CV_32F).var()


def select_keyframes(times, poses, jpgs, valid, dist, deg, window):
    """Motion-based keyframes; among the `window` frames after a trigger take the sharpest."""
    kf, last, i, n = [], None, 0, len(times)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        if last is not None:
            dT = np.linalg.inv(poses[last]) @ poses[i]
            ang = np.degrees(np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()))
            if np.linalg.norm(dT[:3, 3]) < dist and ang < deg:
                i += 1
                continue
        cand = [k for k in range(i, min(n, i + window)) if valid[k]]
        best = max(cand, key=lambda k: sharpness(jpgs[k]))
        kf.append(best)
        last = best
        i = best + 1
    return kf


def voxelize(pts, voxel, min_count=1):
    key = np.floor(pts / voxel).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    h = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cnt), 3))
    for k in range(3):
        sums[:, k] = np.bincount(inv, weights=pts[:, k], minlength=len(cnt))
    return (sums / cnt[:, None])[cnt >= min_count]


def static_voxels(pts, pt, voxel, min_span):
    """Voxel centroids of points whose hits span >= min_span seconds (drops people walking by)."""
    key = np.floor(pts / voxel).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    h = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    order = np.argsort(h, kind="stable")
    h, pts, pt = h[order], pts[order], pt[order]
    starts = np.concatenate([[0], np.nonzero(np.diff(h))[0] + 1])
    cnt = np.diff(np.append(starts, len(h)))
    cen = np.add.reduceat(pts, starts, axis=0) / cnt[:, None]
    tspan = np.maximum.reduceat(pt, starts) - np.minimum.reduceat(pt, starts)
    ok = (cnt >= 2) & (tspan >= min_span)
    return cen[ok].astype(np.float32)


def build_segment_maps(scans, base_interp, T_base_lidar, voxel, log, seg_s=10.0, min_span=1.5):
    """De-skewed scans -> one static voxel submap per seg_s seconds (used for drift correction
    when merging sessions; their union is the session map)."""
    t0 = scans[0][0]
    segs, seg_t, cur, curt, n_raw = [], [], [], [], 0
    k_end = t0 + seg_s

    def flush():
        if cur:
            P, Tm = np.concatenate(cur), np.concatenate(curt)
            segs.append(static_voxels(P, Tm, voxel, min_span))
            seg_t.append(float(np.median(Tm)) + t0)

    for tb, xyz, off in scans:
        if tb >= k_end:
            flush()
            cur, curt = [], []
            while tb >= k_end:
                k_end += seg_s
        if len(xyz) == 0:
            continue
        nb = 10  # de-skew: 10 sub-times across the scan
        tmax = float(off.max())
        edges = np.linspace(0, tmax + 1e-6, nb + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        Ts = base_interp(tb + mids) @ T_base_lidar
        idx = np.clip(np.searchsorted(edges, off, side="right") - 1, 0, nb - 1)
        cur.append(np.einsum("nij,nj->ni", Ts[idx, :3, :3], xyz.astype(np.float64)) + Ts[idx, :3, 3])
        curt.append((tb - t0 + off).astype(np.float64))
        n_raw += len(xyz)
    flush()
    log(f"  lidar: {n_raw/1e6:.1f}M raw points -> {len(segs)} submaps of {seg_s:.0f}s, "
        f"{sum(len(x) for x in segs)/1e6:.2f}M static voxels ({voxel*100:.0f} cm, hit over >= {min_span}s)")
    return segs, np.array(seg_t)


def estimate_floor(cen, base_z, log, below=1.5):
    """Floor = densest horizontal layer below the sensor base (shiny-floor mirror points are
    spread out below it, walls are spread in z, so the floor plane is a sharp peak)."""
    z = cen[(cen[:, 2] > base_z - below) & (cen[:, 2] < base_z + 0.05), 2]
    hist, e = np.histogram(z, bins=np.arange(base_z - below, base_z + 0.05 + 1e-9, 0.01))
    hist = np.convolve(hist, [1, 2, 1], "same")
    floor = float(e[np.argmax(hist)] + 0.005)
    log(f"  floor at z={floor:+.3f} (sensor base_link z={base_z:+.3f} -> base is {base_z-floor:.2f} m above floor)")
    return floor


def clean_map(cen, floor, trajs, log, carve_r=0.25):
    """Drop below-floor mirror points and anything inside the robot's swept volume.
    trajs: list of (Nx3 base_link positions, base_z) – the robot occupies floor..base_z+0.25."""
    from scipy.spatial import cKDTree
    keep = cen[:, 2] > floor - 0.05
    log(f"  dropping {np.sum(~keep)} below-floor (mirror) points")
    cen = cen[keep]
    inside = np.zeros(len(cen), bool)
    for traj, base_z in trajs:
        d, _ = cKDTree(traj[:, :2]).query(cen[:, :2], distance_upper_bound=carve_r)
        inside |= (d < carve_r) & (cen[:, 2] > floor + 0.03) & (cen[:, 2] < base_z + 0.25)
    log(f"  carved {inside.sum()} points inside the robot's swept volume")
    return cen[~inside]


@torch.no_grad()
def color_and_tracks(points, c2w_list, K, W, H, jpg_list, tracks_per_img, voxel, log, ds=4):
    # K may be a single 3x3 or a list (one per image); W, H shared
    """Z-buffered projection of map points into each keyframe: mean color + visibility tracks."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.from_numpy(points).to(dev).double()
    N = len(points)
    csum = torch.zeros(N, 3, device=dev, dtype=torch.float64)
    cw = torch.zeros(N, device=dev, dtype=torch.float64)
    w, h = W // ds, H // ds
    Klist = K if isinstance(K, (list, tuple)) else [K] * len(c2w_list)
    tracks = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i, (c2w, jpg) in enumerate(zip(c2w_list, jpg_list)):
        if isinstance(jpg, str):
            jpg = open(jpg, "rb").read()
        Kd = torch.tensor(np.asarray(Klist[i]), device=dev, dtype=torch.float64)
        w2c = torch.from_numpy(np.linalg.inv(c2w)).to(dev)
        Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = Pc[:, 2]
        uv = (Pc @ Kd.T)[:, :2] / z[:, None].clamp(min=1e-6)
        m = (z > 0.2) & (z < 25) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        ids = torch.nonzero(m).squeeze(1)
        if len(ids) == 0:
            tracks.append(np.zeros(0, np.int64))
            continue
        u = (uv[ids, 0] / ds).long().clamp(0, w - 1)
        v = (uv[ids, 1] / ds).long().clamp(0, h - 1)
        pix = v * w + u
        zi = z[ids]
        # splat each point as a square ~ its voxel footprint so sparse near surfaces occlude
        rad = (0.75 * voxel * Kd[0, 0] / (zi * ds)).round().long().clamp(0, 6)
        zb = torch.full((h, w), 1e9, device=dev, dtype=torch.float64)
        for r in range(0, 7):
            sel = rad >= r
            if not sel.any():
                break
            us, vs, zs = u[sel], v[sel], zi[sel]
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    uu, vv = us + dx, vs + dy
                    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                    zb.view(-1).scatter_reduce_(0, (vv[ok] * w + uu[ok]), zs[ok], reduce="amin")
        zb = zb.view(-1)
        vis = zi <= zb[pix] * 1.03 + 0.03
        ids, pix = ids[vis], pix[vis]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        col = torch.from_numpy(img.reshape(-1, 3).astype(np.float64)).to(dev)[pix]
        wt = 1.0 / z[ids]  # closer views weigh more
        csum.index_add_(0, ids, col * wt[:, None])
        cw.index_add_(0, ids, wt)
        sel = ids.cpu()
        if len(sel) > tracks_per_img:
            sel = sel[torch.randperm(len(sel), generator=gen)[:tracks_per_img]]
        tracks.append(sel.numpy())
    seen = cw > 0
    rgb = torch.full((N, 3), 128.0, device=dev, dtype=torch.float64)
    rgb[seen] = csum[seen] / cw[seen, None]
    log(f"  colored {int(seen.sum())}/{N} points seen by at least one keyframe")
    return rgb.clamp(0, 255).byte().cpu().numpy(), seen.cpu().numpy(), tracks


def write_colmap(sparse, cams, cam_of_img, names, c2ws, points, rgb, tracks):
    """cams: list of (W, H, K); cam_of_img: camera index per image."""
    os.makedirs(sparse, exist_ok=True)
    with open(os.path.join(sparse, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for ci, (W, H, K) in enumerate(cams):
            f.write(struct.pack("<iiQQ", ci + 1, 1, W, H))  # id, PINHOLE, w, h
            f.write(struct.pack("<4d", K[0][0], K[1][1], K[0][2], K[1][2]))
    with open(os.path.join(sparse, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, (n, c2w) in enumerate(zip(names, c2ws)):
            w2c = np.linalg.inv(c2w)
            qx, qy, qz, qw = R.from_matrix(w2c[:3, :3]).as_quat()
            f.write(struct.pack("<i4d3di", i + 1, qw, qx, qy, qz, *w2c[:3, 3], cam_of_img[i] + 1))
            f.write(n.encode() + b"\0")
            f.write(struct.pack("<Q", 0))
    img_of_pt = [[] for _ in range(len(points))]
    for i, tr in enumerate(tracks):
        for pid in tr:
            img_of_pt[pid].append(i + 1)
    with open(os.path.join(sparse, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for j in range(len(points)):
            imgs = img_of_pt[j]
            f.write(struct.pack("<Q3d3BdQ", j + 1, *points[j].astype(np.float64), *rgb[j], 0.0, len(imgs)))
            if imgs:
                f.write(np.array([[im, 0] for im in imgs], dtype="<u4").tobytes())


def add_args(ap):
    ap.add_argument("--kf_dist", type=float, default=0.12, help="new keyframe after this translation (m)")
    ap.add_argument("--kf_deg", type=float, default=5.0, help="... or this rotation (deg)")
    ap.add_argument("--kf_window", type=int, default=5, help="pick sharpest of N frames after trigger")
    ap.add_argument("--jump_m", type=float, default=0.04, help="T265 step larger than this = relocalization")
    ap.add_argument("--jump_guard", type=float, default=0.5, help="drop images within +-s of a jump")
    ap.add_argument("--no_planar", action="store_true", help="keep T265 z (default: constant height)")
    ap.add_argument("--odom_dir", default=None,
                    help="use lidar(-inertial) odometry <odom_dir>/<bag>.tum (FAST-LIO, Mid-360 IMU body frame) as the "
                         "primary odometry instead of T265 (T265 kept as auxiliary reference)")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--max_points", type=int, default=800_000)
    ap.add_argument("--tracks_per_img", type=int, default=4000)


def make_logger(out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, fname), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    return log


def prepare_session(bag_dir, out_dir, a, log):
    """Bag -> keyframe images + T265 camera poses + static lidar map, saved to out_dir/session.npz."""
    bag = sorted(glob.glob(os.path.join(bag_dir, "*.mcap")))[0]
    log(f"reading {bag}")
    odom, imgs, info, static, scans = read_bag(bag, True)
    log(f"  {len(odom)} T265 poses, {len(imgs)} images, {len(scans)} lidar scans")

    # extrinsics from /tf_static in the bag (rec_all.sh publishes extrinsics.yaml there)
    T_base_t265 = lookup(static, "base_link", "t265_mount")
    T_base_cam = lookup(static, "base_link", info["frame"])
    T_base_lidar = lookup(static, "base_link", "livox_frame")
    T_t265_cam = np.linalg.inv(T_base_t265) @ T_base_cam
    T_t265_base = np.linalg.inv(T_base_t265)

    if getattr(a, "odom_dir", None):
        # FAST-LIO body = Mid-360 IMU; lidar in IMU frame (Mid-360 datasheet / FAST-LIO mid360.yaml)
        T_imu_lidar = np.eye(4); T_imu_lidar[:3, 3] = [-0.011, -0.02329, 0.04412]
        T_imu_t265 = T_imu_lidar @ np.linalg.inv(T_base_lidar) @ T_base_t265
        L = np.loadtxt(os.path.join(a.odom_dir, os.path.basename(os.path.normpath(bag_dir)) + ".tum"))
        T = np.tile(np.eye(4), (len(L), 1, 1)); T[:, :3, :3] = R.from_quat(L[:, 4:8]).as_matrix(); T[:, :3, 3] = L[:, 1:4]
        T = T @ T_imu_t265                       # odom <- (virtual) t265 frame, so everything downstream is unchanged
        lio = np.column_stack([L[:, 0], T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat()])
        log(f"  primary odometry: lidar-inertial (FAST-LIO) {len(lio)} poses; T265 ({len(odom)} poses) kept as reference")
        aux_t265 = odom
        traj, jump_times = fix_trajectory(lio, 1e9, not a.no_planar, log)
    else:
        aux_t265 = None
        traj, jump_times = fix_trajectory(odom, a.jump_m, not a.no_planar, log)
    t265_interp = PoseInterp(traj)

    def base_interp(ts):
        return t265_interp(ts) @ T_t265_base

    times = np.array([t for t, _ in imgs])
    jpgs = [d for _, d in imgs]
    valid = (times >= traj[0, 0]) & (times <= traj[-1, 0])
    for jt in jump_times:
        valid &= np.abs(times - jt) > a.jump_guard
    c2w_all = t265_interp(times) @ T_t265_cam
    kf = select_keyframes(times, c2w_all, jpgs, valid, a.kf_dist, a.kf_deg, a.kf_window)
    log(f"  keyframes: {len(kf)} / {len(imgs)} images")

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in glob.glob(os.path.join(img_dir, "*.jpg")):
        os.remove(f)
    names = [f"{n:05d}.jpg" for n in range(len(kf))]
    for name, k in zip(names, kf):
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(jpgs[k])
    c2ws = c2w_all[kf]
    with open(os.path.join(out_dir, "poses_tum.txt"), "w") as f:  # camera poses, TUM format
        for k, c2w in zip(kf, c2ws):
            q = R.from_matrix(c2w[:3, :3]).as_quat()
            f.write(f"{times[k]:.6f} {' '.join(f'{x:.6f}' for x in c2w[:3,3])} {' '.join(f'{x:.8f}' for x in q)}\n")
    del imgs, jpgs

    assert not any(abs(d) > 1e-9 for d in info["D"]), "expected rectified images (D=0)"
    segs, seg_t = build_segment_maps(scans, base_interp, T_base_lidar, a.voxel, log)
    del scans
    lidar_map = voxelize(np.concatenate(segs).astype(np.float64), a.voxel).astype(np.float32)
    traj_t = np.arange(traj[0, 0], traj[-1, 0], 0.05)
    base_traj = base_interp(traj_t)[:, :3, 3]
    base_z = float(np.median(base_traj[:, 2]))
    np.savez(os.path.join(out_dir, "session.npz"), names=np.array(names), c2ws=c2ws, kf_times=times[kf],
             K=np.array(info["K"]).reshape(3, 3), W=info["w"], H=info["h"], lidar_map=lidar_map,
             seg_pts=np.concatenate(segs), seg_len=np.array([len(x) for x in segs]), seg_t=seg_t,
             base_traj=base_traj.astype(np.float32), base_traj_t=traj_t, base_z=base_z)
    return np.load(os.path.join(out_dir, "session.npz"))


def finalize(out_dir, image_paths, names, c2ws, cams, cam_of_img, points, a, log):
    """Seen-filter + thin the cloud, color it, write COLMAP sparse/0 and debug overlays."""
    W, H = cams[0][0], cams[0][1]
    Ks = [cams[c][2] for c in cam_of_img]
    _, seen, _ = color_and_tracks(points, c2ws, Ks, W, H, image_paths, 0, a.voxel, log, ds=8)
    points, voxel = points[seen], a.voxel
    while len(points) > a.max_points:
        voxel *= 1.2
        points = voxelize(points, voxel).astype(np.float32)
    log(f"  init cloud: {len(points)} points (effective voxel {voxel*100:.1f} cm)")
    rgb, seen, tracks = color_and_tracks(points, c2ws, Ks, W, H, image_paths, a.tracks_per_img, voxel, log)
    write_colmap(os.path.join(out_dir, "sparse", "0"), cams, cam_of_img, names, c2ws, points, rgb, tracks)

    dbg = os.path.join(out_dir, "debug")
    os.makedirs(dbg, exist_ok=True)
    for n in np.linspace(0, len(names) - 1, 9).astype(int):
        img = cv2.imread(image_paths[n])
        Pc = points[tracks[n]] @ np.linalg.inv(c2ws[n])[:3, :3].T + np.linalg.inv(c2ws[n])[:3, 3]
        uv = (Pc @ np.asarray(Ks[n]).T)[:, :2] / Pc[:, 2:3]
        col = cv2.applyColorMap(np.clip(Pc[:, 2] / 10 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        for (u, v), c in zip(uv.astype(int), col):
            cv2.circle(img, (u, v), 2, tuple(int(x) for x in c), -1)
        cv2.imwrite(os.path.join(dbg, f"overlay_{os.path.splitext(names[n])[0]}.jpg"), img)
    log(f"done -> {out_dir}  ({len(names)} images, {len(points)} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prep_only", action="store_true", help="only write images + session.npz (for merging)")
    ap.add_argument("--base_height", type=float, default=0.0, help="base_link height above the floor (m)")
    add_args(ap)
    a = ap.parse_args()
    log = make_logger(a.out_dir, "prep_log.txt")
    S = prepare_session(a.bag_dir, a.out_dir, a, log)
    if a.prep_only:
        return
    # single session: floor unknown from lidar (glossy floor); --base_height = sensor base above floor
    floor = float(S["base_z"]) - a.base_height
    pts = clean_map(S["lidar_map"], floor, [(S["base_traj"], float(S["base_z"]))], log)
    names = list(S["names"])
    finalize(a.out_dir, [os.path.join(a.out_dir, "images", n) for n in names], names, S["c2ws"],
             [(int(S["W"]), int(S["H"]), S["K"])], [0] * len(names), pts.astype(np.float32), a, log)


if __name__ == "__main__":
    main()
