#!/usr/bin/env python3
"""Smooth 360-degree (equirectangular) fly-through videos of a trained MCMC model, one per sensor height.

For each height hX the camera follows that session's (BA-refined) trajectory, resampled at constant speed,
smoothed, kept at a fixed height (median of that session's camera height), level (no roll / pitch), heading
along the direction of travel. Each frame renders 6 cube faces in one batch and resamples them to equirect.
The mp4 gets Google Spherical Video V1 metadata so 360 players / YouTube recognise it.

usage (gs env): render_pano.py EB_p1_final [--heights h1 h2 h3] [--max_frames N]
"""
import argparse, os, struct, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.expanduser("~/gsplat/examples"))
from datasets.colmap import Parser  # noqa: E402
from datasets.normalize import transform_cameras  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

ROOT = os.path.expanduser("~/gsplat_eb")


# ---------------------------------------------------------------- path
def resample(xy, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, xy[:, k]) for k in range(xy.shape[1])], 1)


def make_path(c2w_world, step, smooth_m, yaw_smooth_m, max_yaw_step):
    xy = resample(c2w_world[:, :2, 3], step)
    xy = gaussian_filter1d(xy, smooth_m / step, axis=0, mode="nearest")
    xy = resample(xy, step)                       # constant speed again after smoothing
    t = np.gradient(xy, axis=0)
    t = gaussian_filter1d(t, yaw_smooth_m / step, axis=0, mode="nearest")
    yaw = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    for i in range(1, len(yaw)):                  # cap turn rate (turnarounds at corridor ends)
        yaw[i] = yaw[i - 1] + np.clip(yaw[i] - yaw[i - 1], -max_yaw_step, max_yaw_step)
    return xy, yaw


# ---------------------------------------------------------------- cube -> equirect
def rot_to(f, up):
    """OpenCV camera (x right, y down, z forward) looking along f, with 'up' as image up; columns = cam axes in rig."""
    z = f / np.linalg.norm(f); y = -up / np.linalg.norm(up); x = np.cross(y, z)
    return np.stack([x, y, z], 1)


# rig frame: x forward, y left, z up
EX, EY, EZ = np.eye(3)
FACES = [rot_to(EX, EZ), rot_to(-EY, EZ), rot_to(-EX, EZ), rot_to(EY, EZ), rot_to(EZ, -EX), rot_to(-EZ, EX)]


def equirect_maps(W, H, fov_deg, device):
    lon = (torch.arange(W, device=device) + 0.5) / W * 2 * np.pi - np.pi       # -pi (behind, left) .. pi
    lat = np.pi / 2 - (torch.arange(H, device=device) + 0.5) / H * np.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    d = torch.stack([torch.cos(lat) * torch.cos(lon), -torch.cos(lat) * torch.sin(lon), torch.sin(lat)], -1)
    R = torch.tensor(np.stack(FACES), dtype=torch.float32, device=device)       # [6,3,3] rig<-cam
    c = torch.einsum("fji,hwj->fhwi", R, d)                                     # cam coords per face
    best = c[..., 2].argmax(0)                                                  # face most aligned
    th = np.tan(np.radians(fov_deg) / 2)
    grids = torch.stack([c[..., 0] / c[..., 2], c[..., 1] / c[..., 2]], -1) / th   # [-1,1] inside face
    masks = torch.stack([(best == f) for f in range(6)]).float()[:, None]          # [6,1,H,W]
    return grids, masks


def inject_spherical(path):
    xml = (b'<?xml version="1.0"?><rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
           b'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/"><GSpherical:Spherical>true</GSpherical:Spherical>'
           b'<GSpherical:Stitched>true</GSpherical:Stitched><GSpherical:StitchingSoftware>gsplat</GSpherical:StitchingSoftware>'
           b'<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType></rdf:SphericalVideo>')
    box = struct.pack(">I4s", 8 + 16 + len(xml), b"uuid") + bytes.fromhex("ffcc8263f8554a938814587a02521fdd") + xml
    data = bytearray(open(path, "rb").read())

    def children(start, end):
        i = start
        while i < end:
            size, typ = struct.unpack(">I4s", data[i:i + 8])
            yield i, size, typ
            i += size

    top = list(children(0, len(data)))
    moov = next(b for b in top if b[2] == b"moov")
    mdat = next(b for b in top if b[2] == b"mdat")
    for t0, tsz, ttyp in children(moov[0] + 8, moov[0] + moov[1]):
        if ttyp == b"trak" and b"vide" in data[t0:t0 + tsz] and b"hdlr" in data[t0:t0 + tsz]:
            break
    if moov[0] < mdat[0]:                         # faststart layout: chunk offsets move by len(box)
        for k in (b"stco", b"co64"):
            j = data.find(k, moov[0], moov[0] + moov[1])
            while j > 0:
                n = struct.unpack(">I", data[j + 8:j + 12])[0]
                w = 4 if k == b"stco" else 8
                for e in range(n):
                    o = j + 12 + e * w
                    v = int.from_bytes(data[o:o + w], "big") + len(box)
                    data[o:o + w] = v.to_bytes(w, "big")
                j = data.find(k, j + 4, moov[0] + moov[1])
    data[t0 + tsz:t0 + tsz] = box
    struct.pack_into(">I", data, t0, tsz + len(box))
    struct.pack_into(">I", data, moov[0], moov[1] + len(box))
    open(path, "wb").write(data)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--heights", nargs="+", default=["h1", "h2", "h3"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--step", type=float, default=0.04, help="m per frame (x30 fps = speed)")
    ap.add_argument("--smooth", type=float, default=0.6, help="path smoothing sigma (m)")
    ap.add_argument("--yaw_smooth", type=float, default=1.0, help="heading smoothing sigma (m)")
    ap.add_argument("--max_turn", type=float, default=3.0, help="max heading change per frame (deg)")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--face", type=int, default=1280)
    ap.add_argument("--fov", type=float, default=95.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--test_frame", type=int, default=-1, help="only render this frame to a jpg")
    ap.add_argument("--no_ut", action="store_true", help="plain EWA rasterization (shows seams between cube faces)")
    ap.add_argument("--front", action="store_true", help="normal forward view (ZED intrinsics, 1280x720) instead of 360")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    res = os.path.join(ROOT, "results", a.scene)
    out_dir = a.out or os.path.join(res, "pano")
    os.makedirs(out_dir, exist_ok=True)

    parser = Parser(os.path.join(ROOT, "data", a.scene), factor=1, normalize=True, test_every=8)
    T = parser.transform
    world = transform_cameras(np.linalg.inv(T), parser.camtoworlds.copy())
    names = np.array(parser.image_names)

    ckpt = a.ckpt or sorted(os.listdir(os.path.join(res, "ckpts")), key=lambda s: int(s.split("_")[1]))[-1]
    sp = torch.load(os.path.join(res, "ckpts", ckpt), map_location=dev, weights_only=False)["splats"]
    means, quats = sp["means"], sp["quats"]
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    print(f"{a.scene}: {ckpt}, {len(means)} Gaussians", flush=True)

    S, W, H = a.face, a.width, a.width // 2
    f = (S / 2) / np.tan(np.radians(a.fov) / 2)
    K = torch.tensor([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None].repeat(6, 1, 1)
    grids, masks = equirect_maps(W, H, a.fov, dev)
    R_faces = np.stack(FACES)
    if a.front:                                   # single level forward camera with the training intrinsics
        cid = parser.camera_ids[0]
        K = torch.tensor(parser.Ks_dict[cid], dtype=torch.float32, device=dev)[None]
        FW, FH = parser.imsize_dict[cid]
        R_faces = R_faces[:1]

    for h in a.heights:
        sel = np.array([n.startswith(h + "_") for n in names])
        order = np.argsort(names[sel])
        cw = world[sel][order]
        z = float(np.median(cw[:, 2, 3]))
        xy, yaw = make_path(cw, a.step, a.smooth, a.yaw_smooth, np.radians(a.max_turn))
        if a.test_frame >= 0:
            xy, yaw = xy[a.test_frame:a.test_frame + 1], yaw[a.test_frame:a.test_frame + 1]
        if a.max_frames:
            xy, yaw = xy[:a.max_frames], yaw[:a.max_frames]
        n = len(xy)
        print(f"  {h}: {sel.sum()} cams, height {z:.2f} m, path {n * a.step:.1f} m -> {n} frames "
              f"({n / 30:.0f} s)", flush=True)
        path = os.path.join(out_dir, f"{a.scene}_{h}_{'front' if a.front else '360'}.mp4")
        wr = imageio.get_writer(path, fps=30, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=16, ffmpeg_log_level="error")
        t0 = time.time()
        for i in range(n):
            c, s_ = np.cos(yaw[i]), np.sin(yaw[i])
            R_rig = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            c2w = np.tile(np.eye(4), (len(R_faces), 1, 1))
            c2w[:, :3, :3] = R_rig @ R_faces
            c2w[:, :3, 3] = [xy[i, 0], xy[i, 1], z]
            c2w = transform_cameras(T, c2w)
            vm = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=dev))
            with torch.no_grad():
              if a.front:
                img, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, FW, FH, sh_degree=3,
                                          near_plane=0.01, far_plane=1e10)
                frame = (img[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
              else:
                img, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, S, S, sh_degree=3,
                                          near_plane=0.01, far_plane=1e10, packed=a.no_ut,
                                          with_ut=not a.no_ut, with_eval3d=not a.no_ut)
                img = img.clamp(0, 1).permute(0, 3, 1, 2)                       # [6,3,S,S]
                eq = (F.grid_sample(img, grids, mode="bilinear", padding_mode="border", align_corners=False)
                      * masks).sum(0)
                frame = (eq.permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            wr.append_data(frame)
            if i == n // 2:
                tag = ("front_" if a.front else "") + (f"f{a.test_frame}" + ("_ewa" if a.no_ut else "_ut") if a.test_frame >= 0 else "mid")
                imageio.imwrite(os.path.join(out_dir, f"{a.scene}_{h}_{tag}.jpg"), frame, quality=90)
            if i % 300 == 0:
                print(f"    {i}/{n}  {time.time() - t0:.0f}s", flush=True)
        wr.close()
        if not a.front:
            inject_spherical(path)
        print(f"  -> {path} ({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
