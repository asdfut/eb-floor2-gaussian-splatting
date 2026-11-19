#!/usr/bin/env python3
"""Re-evaluate merged models on held-out images: raw PSNR and exposure-fair PSNR (per-image per-channel
gain+bias fit of the render to the photo before PSNR), per group. usage: eval_colourfit.py <out_dir> <data> [names.json]"""
import sys, json, os, numpy as np, torch
from PIL import Image
from pycolmap import SceneManager
from gsplat.rendering import rasterization
out, data = sys.argv[1], sys.argv[2]
names = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else list(json.load(open(f"{out}/eval.json"))["per_image"])
sp = torch.load(f"{out}/ckpts/ckpt_merged.pt", map_location="cuda", weights_only=False)["splats"]
G = (sp["means"], sp["quats"], torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]), torch.cat([sp["sh0"], sp["shN"]], 1))
m = SceneManager(data + "/sparse/0/"); m.load_cameras(); m.load_images(); by = {m.images[i].name: m.images[i] for i in m.images}
res = {}
for n in names:
    if n not in by: continue
    im = by[n]; c = m.cameras[im.camera_id]
    T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
    K = torch.tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    with torch.no_grad():
        r, _, _ = rasterization(*G, torch.tensor(T, dtype=torch.float32, device="cuda")[None], K, c.width, c.height, sh_degree=3, near_plane=0.15)
    r = r[0].clamp(0, 1).reshape(-1, 3)
    g = torch.from_numpy(np.asarray(Image.open(f"{data}/images/{n}").convert("RGB"), np.float32) / 255).cuda().reshape(-1, 3)
    raw = float(-10 * torch.log10(((r - g) ** 2).mean()))
    A = torch.stack([r, torch.ones_like(r)], -1)                       # (N,3,2)
    rf = torch.empty_like(r)
    for ch in range(3):
        k = torch.linalg.lstsq(A[:, ch], g[:, ch:ch + 1]).solution; rf[:, ch] = (A[:, ch] @ k)[:, 0]
    res[n] = (raw, float(-10 * torch.log10(((rf - g) ** 2).mean())))
a = np.array(list(res.values()))
groups = sorted(set(n[:5] for n in res))
summ = {"n": len(res), "psnr_raw": float(a[:, 0].mean()), "psnr_fair": float(a[:, 1].mean()),
        "per_group": {gname: [round(float(np.mean([v[0] for k, v in res.items() if k.startswith(gname)])), 2),
                              round(float(np.mean([v[1] for k, v in res.items() if k.startswith(gname)])), 2)] for gname in groups}}
json.dump(summ, open(f"{out}/eval_fair.json", "w"), indent=1); print(json.dumps(summ))
