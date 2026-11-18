#!/usr/bin/env python3
"""Smooth fly-through of the whole 2F loop with the merged (world-frame) model: forward view and/or 360.

Route (floor frame, along the recorded lanes / corridor centre lines): west arm (p1) eastwards on its south lane -> north along p1's N-S corridor -> east along
the north corridor (p4) -> south along the east corridor (p3) -> west along the south corridor (p2) -> north
back up p1's N-S corridor -> west back along the arm. Constant speed, rounded corners, level camera, heading
along the path, fixed height per video (median training-camera height of h1 / h2 / h3).
usage (gs env): render_floor_video.py [--heights h1 h2 h3] [--modes front 360] [--max_frames N]
"""
import argparse, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from pycolmap import SceneManager
from gsplat.rendering import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_pano import resample, FACES, equirect_maps, inject_spherical  # noqa: E402

ROOT = os.path.expanduser("~/gsplat_eb")
# waypoints (x, y) on the corridor centre lines, floor frame (m)
ROUTE = [(-14.5, -2.8), (4.0, -2.8), (6.9, -1.6), (6.9, 0.6), (8.5, 1.15), (44.0, 0.8), (45.85, 0.0),
         (45.85, -9.0), (44.5, -10.0), (8.5, -10.1), (6.9, -9.0), (6.9, -1.2), (5.0, 0.0), (-14.5, 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results", "EB_floor2_gs", "ckpts", "ckpt_merged.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "EB_floor2_final"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "EB_floor2_gs", "videos"))
    ap.add_argument("--heights", nargs="+", default=["h1", "h2", "h3"])
    ap.add_argument("--modes", nargs="+", default=["front", "360"])
    ap.add_argument("--step", type=float, default=0.02, help="m per frame at 30 fps (0.02 = 0.6 m/s)")
    ap.add_argument("--corner", type=float, default=0.8, help="corner rounding sigma (m)")
    ap.add_argument("--yaw_smooth", type=float, default=0.6)
    ap.add_argument("--max_turn", type=float, default=2.5)
    ap.add_argument("--near", type=float, default=0.15)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--face", type=int, default=1280)
    ap.add_argument("--fov", type=float, default=95.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--plan_only", action="store_true", help="only save route_<h>.npz")
    ap.add_argument("--labels", action="store_true", help="write the height name on each row of the stacked video")
    ap.add_argument("--route", default=None, help=".npy with (x, y) waypoints (default: built-in ROUTE)")
    ap.add_argument("--clear", type=float, default=0.0, help="minimum distance of the camera path from walls (m, 0 = off; wall cells still noisy)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    # training cameras: heights + free-space check of the route
    m = SceneManager(os.path.join(a.data, "sparse", "0") + "/"); m.load_cameras(); m.load_images()
    cams, names = [], []
    for i in m.images:
        T = np.eye(4); T[:3, :3] = m.images[i].R(); T[:3, 3] = m.images[i].tvec
        cams.append(np.linalg.inv(T)); names.append(m.images[i].name)
    cams = np.array(cams); names = np.array(names)
    c0 = m.cameras[sorted(m.cameras)[0]]
    K_front = torch.tensor([[c0.fx, 0, c0.cx], [0, c0.fy, c0.cy], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    FW, FH = c0.width, c0.height

    def plan(xy):
        """constant-speed, smoothed, heading-limited path from a polyline"""
        xy = resample(gaussian_filter1d(xy, a.corner / 0.05, axis=0, mode="nearest") if len(xy) > 50 else xy, a.step)
        t = gaussian_filter1d(np.gradient(xy, axis=0), a.yaw_smooth / a.step, axis=0, mode="nearest")
        yaw = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        for i in range(1, len(yaw)):
            yaw[i] = yaw[i - 1] + np.clip(yaw[i] - yaw[i - 1], -np.radians(a.max_turn), np.radians(a.max_turn))
        return xy, yaw

    route = np.load(a.route) if a.route else np.array(ROUTE, float)
    base_xy, base_yaw = plan(resample(route, 0.05))
    fwd = cams[:, :3, 2]; cam_yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    m.load_points3D()
    Pw = m.points3D[(m.points3D[:, 2] > 0.1) & (m.points3D[:, 2] < 1.8)]
    cells, cnt = np.unique(np.round(Pw[:, :2] / 0.05).astype(np.int64), axis=0, return_counts=True)
    W2 = cells[cnt >= 5] * 0.05                    # 5 cm 2D wall/object cells with real support (no stray points)
    wtree = cKDTree(W2)

    def snap_route(h):
        """pull the route onto the lane actually driven at this height in the same direction (training views)"""
        sel_h = np.array([n.split("_")[1] == h for n in names])
        C, Y, Z = cams[sel_h, :2, 3], cam_yaw[sel_h], cams[sel_h, 2, 3]
        tree = cKDTree(C)
        xy, yaw = base_xy.copy(), base_yaw.copy()
        for it in range(2):
            tgt, zz, ok = xy.copy(), np.full(len(xy), np.nan), np.zeros(len(xy), bool)
            for i in range(len(xy)):
                idx = np.array(tree.query_ball_point(xy[i], 2.0), int)
                if len(idx) == 0:
                    continue
                idx = idx[np.cos(Y[idx] - yaw[i]) > np.cos(np.radians(35))]
                if len(idx) < 3:
                    continue
                idx = idx[np.argsort(np.linalg.norm(C[idx] - xy[i], axis=1))[:10]]
                tgt[i] = np.median(C[idx], 0); zz[i] = np.median(Z[idx]); ok[i] = True
            xy, yaw = plan(gaussian_filter1d(tgt, 0.6 / a.step, axis=0, mode="nearest"))
        # wall clearance: push the path away from walls/objects closer than a.clear (corner cutting otherwise brings
        # the camera within a few cm of a wall corner -> Gaussians popping through the near plane)
        for it in range(4 if a.clear > 0 else 0):
            d, wi = wtree.query(xy)
            m_ = d < a.clear
            if not m_.any():
                break
            push = (xy[m_] - W2[wi[m_]]) / np.maximum(d[m_], 1e-3)[:, None] * (a.clear - d[m_])[:, None]
            tgt = xy.copy(); tgt[m_] += push
            xy, yaw = plan(gaussian_filter1d(tgt, 0.3 / a.step, axis=0, mode="nearest"))
        dw, _ = wtree.query(xy)
        zfill = np.where(np.isnan(zz), np.nanmedian(zz), zz)
        z = gaussian_filter1d(np.interp(np.linspace(0, 1, len(xy)), np.linspace(0, 1, len(zfill)), zfill), 2.0 / a.step)
        d, _ = tree.query(xy)
        print(f"  {h}: route {len(xy) * a.step:.0f} m, {len(xy)} frames; on-lane {ok.mean() * 100:.0f}%, distance to nearest "
              f"same-height camera median {np.median(d):.2f} m, 95% {np.percentile(d, 95):.2f} m; height {np.median(z):.2f} m; "
              f"min wall clearance {dw.min():.2f} m",
              flush=True)
        if a.max_frames:
            xy, yaw, z = xy[:a.max_frames], yaw[:a.max_frames], z[:a.max_frames]
        return xy, yaw, z

    if a.plan_only:                                # routes only, no GPU work
        for h in a.heights:
            xy, yaw, zs = snap_route(h)
            np.savez(os.path.join(a.out, f"route_{h}.npz"), xy=xy, yaw=yaw, z=zs)
        return
    sp = torch.load(a.ckpt, map_location=dev, weights_only=False)["splats"]
    means, quats = sp["means"], sp["quats"]
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    S, W, H = a.face, a.width, a.width // 2
    f = (S / 2) / np.tan(np.radians(a.fov) / 2)
    K_cube = torch.tensor([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None].repeat(6, 1, 1)
    grids, masks = equirect_maps(W, H, a.fov, dev) if "360" in a.modes else (None, None)

    if "stack" in a.modes:
        # one video, three heights stacked (top h3, middle h2, bottom h1) on a common route (h2 lanes)
        import cv2
        routes = {h: snap_route(h) for h in ["h1", "h2", "h3"]}
        xy, yaw, _ = routes["h2"]
        frac = np.linspace(0, 1, len(xy))
        zs = {h: np.interp(frac, np.linspace(0, 1, len(r[2])), r[2]) for h, r in routes.items()}
        path = os.path.join(a.out, "floor2_3heights_front.mp4")
        wr = imageio.get_writer(path, fps=30, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=16, ffmpeg_log_level="error",
                                output_params=["-g", "15", "-movflags", "+faststart"])
        wr3 = imageio.get_writer(os.path.join(a.out, "floor2_h3_front.mp4"), fps=30, codec="libx264", quality=8,
                                 pixelformat="yuv420p", macro_block_size=16, ffmpeg_log_level="error",
                                output_params=["-g", "15", "-movflags", "+faststart"])
        t0 = time.time()
        K3 = K_front.repeat(3, 1, 1)
        for i in range(len(xy)):
            c, s_ = np.cos(yaw[i]), np.sin(yaw[i])
            c2w = np.tile(np.eye(4), (3, 1, 1))
            c2w[:, :3, :3] = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ FACES[0]
            for k, h in enumerate(["h3", "h2", "h1"]):
                c2w[k, :3, 3] = [xy[i, 0], xy[i, 1], zs[h][i]]
            vm = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))
            with torch.no_grad():   # one height at a time (a 3-camera batch of ~9M Gaussians does not fit in 16 GB)
                img = torch.cat([rasterization(means, quats, scales, opac, colors, vm[k:k + 1], K3[:1], FW, FH,
                                               sh_degree=3, near_plane=a.near, far_plane=1e10)[0] for k in range(3)])
            rows = (img.clamp(0, 1) * 255).round().byte().cpu().numpy()
            rows = [np.ascontiguousarray(r) for r in rows]
            if a.labels:
                for r, h in zip(rows, ["h3", "h2", "h1"]):
                    cv2.putText(r, h, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(r, h, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
            frame = np.concatenate(rows, 0)
            wr.append_data(frame)
            wr3.append_data(rows[0])                     # h3 alone
            if i in (len(xy) // 4, len(xy) // 2, 3 * len(xy) // 4):
                imageio.imwrite(os.path.join(a.out, f"floor2_3heights_{i:05d}.jpg"), frame, quality=90)
        wr.close(); wr3.close()
        print(f"  3 heights stacked -> {path} ({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
        a.modes = [m_ for m_ in a.modes if m_ != "stack"]

    for h in (a.heights if a.modes else []):
        xy, yaw, zs = snap_route(h)
        np.savez(os.path.join(a.out, f"route_{h}.npz"), xy=xy, yaw=yaw, z=zs)
        if a.plan_only:
            continue
        for mode in a.modes:
            path = os.path.join(a.out, f"floor2_{h}_{mode}.mp4")
            wr = imageio.get_writer(path, fps=30, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=16, ffmpeg_log_level="error",
                                output_params=["-g", "15", "-movflags", "+faststart"])
            R_faces = np.stack(FACES) if mode == "360" else np.stack(FACES)[:1]
            t0 = time.time()
            for i in range(len(xy)):
                c, s_ = np.cos(yaw[i]), np.sin(yaw[i])
                c2w = np.tile(np.eye(4), (len(R_faces), 1, 1))
                c2w[:, :3, :3] = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ R_faces
                c2w[:, :3, 3] = [xy[i, 0], xy[i, 1], zs[i]]
                vm = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))
                with torch.no_grad():
                    if mode == "front":
                        img, _, _ = rasterization(means, quats, scales, opac, colors, vm, K_front, FW, FH,
                                                  sh_degree=3, near_plane=a.near, far_plane=1e10)
                        frame = (img[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
                    else:
                        # one face at a time: a 6-camera batch needs ~9 GB for the SH evaluation of 8M Gaussians
                        img = torch.cat([rasterization(means, quats, scales, opac, colors, vm[k:k + 1], K_cube[:1], S, S,
                                                       sh_degree=3, near_plane=a.near, far_plane=1e10, packed=False,
                                                       with_ut=True, with_eval3d=True)[0] for k in range(6)])
                        img = img.clamp(0, 1).permute(0, 3, 1, 2)
                        eq = (F.grid_sample(img, grids, mode="bilinear", padding_mode="border", align_corners=False)
                              * masks).sum(0)
                        frame = (eq.permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                wr.append_data(frame)
                if i in (len(xy) // 4, len(xy) // 2, 3 * len(xy) // 4):
                    imageio.imwrite(os.path.join(a.out, f"floor2_{h}_{mode}_{i:05d}.jpg"), frame, quality=90)
            wr.close()
            if mode == "360":
                inject_spherical(path)
            print(f"  {h} {mode} -> {path} ({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
