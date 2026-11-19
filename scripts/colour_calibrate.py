#!/usr/bin/env python3
"""Undo the free global colour gauge of --exposure-comp models (gs env).
With per-image exposure (gain+bias) learned jointly, a partition's Gaussian colours are only defined up to a global
gain/bias, so partitions drift apart (and away from the photos). Fit photo ~= a*render + b per channel over a sample of the
partition's images and bake it into the SH: c' = a*c + b. usage: colour_calibrate.py <part_result_dir> <part_data> [n=200]"""
import sys, glob, os, json, numpy as np, torch
from PIL import Image
from pycolmap import SceneManager
from gsplat.rendering import rasterization
res, data = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
ck = sorted(glob.glob(f"{res}/ckpts/ckpt_*_rank0.pt"), key=lambda f: int(os.path.basename(f).split("_")[1]))[-1]
d = torch.load(ck, map_location="cuda", weights_only=False); sp = d["splats"]
G = (sp["means"], sp["quats"], torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]), torch.cat([sp["sh0"], sp["shN"]], 1))
m = SceneManager(data + "/sparse/0/"); m.load_cameras(); m.load_images()
ids = sorted(m.images, key=lambda i: m.images[i].name); ids = [ids[k] for k in np.linspace(0, len(ids) - 1, n).astype(int)]
S = torch.zeros(3, 5, device="cuda", dtype=torch.float64)   # sums: r, g, r*r, r*g, count
per = []
for i in ids:
    im = m.images[i]; c = m.cameras[im.camera_id]
    T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
    K = torch.tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    with torch.no_grad():
        r, _, _ = rasterization(*G, torch.tensor(T, dtype=torch.float32, device="cuda")[None], K, c.width, c.height, sh_degree=3, near_plane=0.15)
    r = r[0].clamp(0, 1).reshape(-1, 3).double()
    g = torch.from_numpy(np.asarray(Image.open(f"{data}/images/{im.name}").convert("RGB"), np.float32) / 255).cuda().reshape(-1, 3).double()
    ok = (g < 0.98).all(1) & (r < 0.98).all(1)                     # ignore clipped lamps
    r, g = r[ok], g[ok]
    S += torch.stack([r.sum(0), g.sum(0), (r * r).sum(0), (r * g).sum(0), torch.full((3,), float(len(r)), device="cuda", dtype=torch.float64)], 1)
    per.append(float(g.mean() / max(float(r.mean()), 1e-6)))
sr, sg, srr, srg, N = S.unbind(1)
a = (N * srg - sr * sg) / (N * srr - sr * sr); b = (sg - a * sr) / N
a, b = a.float(), b.float()
C0 = 0.28209479177387814
with torch.no_grad():
    c_dc = 0.5 + C0 * sp["sh0"]                                    # (N,1,3)
    sp["sh0"] = ((a * c_dc + b) - 0.5) / C0
    sp["shN"] = sp["shN"] * a
torch.save({"step": d["step"], "splats": sp}, f"{res}/ckpts/ckpt_colourfix.pt")
info = {"gain": a.tolist(), "bias": b.tolist(), "photo/render brightness per image p5/p50/p95": np.percentile(per, [5, 50, 95]).round(3).tolist()}
json.dump(info, open(f"{res}/colourfix.json", "w")); print(os.path.basename(res), json.dumps(info))
