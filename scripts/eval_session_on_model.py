#!/usr/bin/env python3
"""Render a merged model at a session's floor-frame cameras (sessions_lio_dense/<bag>/session.npz) and compare with the
photos: raw + exposure-fair PSNR, LPIPS, side-by-side jpgs. Loads only Gaussians within --radius of each camera, so it
fits next to a running training job. gs env.
usage: eval_session_on_model.py <model_dir> <session_dir> <out_dir> [--every 16] [--radius 15] [--save 12]"""
import argparse, os, json, numpy as np, torch
from PIL import Image
from gsplat.rendering import rasterization
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('session'); ap.add_argument('out')
ap.add_argument('--every', type=int, default=16); ap.add_argument('--radius', type=float, default=40.0); ap.add_argument('--save', type=int, default=12)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sp = torch.load(f"{a.model}/ckpts/ckpt_merged.pt", map_location="cpu", weights_only=False)["splats"]
means = sp["means"]; S = np.load(f"{a.session}/session.npz")
K = torch.tensor(S["K"], dtype=torch.float32, device="cuda")[None]; W, H = int(S["W"]), int(S["H"])
lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
idx = list(range(0, len(S["names"]), a.every)); save_at = set(idx[::max(1, len(idx) // a.save)])
res = {}
for i in idx:
    c2w = S["c2ws"][i]; c = torch.tensor(c2w[:3, 3], dtype=torch.float32)
    Rw = torch.tensor(c2w[:3, :3], dtype=torch.float32); pc = (means - c) @ Rw          # camera frame (x right, y down, z fwd)
    sel = (pc[:, 2] > -0.5) & ((pc ** 2).sum(1) < a.radius ** 2) & (pc[:, 0].abs() < pc[:, 2].clamp(min=0) * 1.4 + 1.5) \
          & (pc[:, 1].abs() < pc[:, 2].clamp(min=0) * 0.9 + 1.5)                               # view frustum + margin
    G = [t.cuda() for t in (means[sel], sp["quats"][sel], torch.exp(sp["scales"][sel]), torch.sigmoid(sp["opacities"][sel]),
                            torch.cat([sp["sh0"][sel], sp["shN"][sel]], 1))]
    with torch.no_grad():
        r, _, _ = rasterization(*G, torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda")[None], K, W, H, sh_degree=3, near_plane=0.15)
    r = r[0].clamp(0, 1); g = torch.from_numpy(np.asarray(Image.open(f"{a.session}/images/{S['names'][i]}").convert("RGB"), np.float32) / 255).cuda()
    rr, gg = r.reshape(-1, 3), g.reshape(-1, 3); raw = float(-10 * torch.log10(((rr - gg) ** 2).mean()))
    A = torch.stack([rr, torch.ones_like(rr)], -1); rf = torch.empty_like(rr)
    for ch in range(3):
        k = torch.linalg.lstsq(A[:, ch], gg[:, ch:ch + 1]).solution; rf[:, ch] = (A[:, ch] @ k)[:, 0]
    fair = float(-10 * torch.log10(((rf - gg) ** 2).mean()))
    with torch.no_grad(): l = float(lp(r.permute(2, 0, 1)[None], g.permute(2, 0, 1)[None]))
    res[str(S["names"][i])] = [raw, fair, l, [round(float(x), 2) for x in c2w[:3, 3]]]
    if i in save_at:
        Image.fromarray((torch.cat([g, rf.reshape(H, W, 3).clamp(0, 1)], 1).cpu().numpy() * 255).astype(np.uint8)).resize((W, H // 2)).save(f"{a.out}/cmp_{S['names'][i]}")
    del G; torch.cuda.empty_cache()
v = np.array([x[:3] for x in res.values()])
summ = {"n": len(res), "psnr_raw": round(float(v[:, 0].mean()), 2), "psnr_fair": round(float(v[:, 1].mean()), 2), "lpips": round(float(v[:, 2].mean()), 3),
        "psnr_fair_p10": round(float(np.percentile(v[:, 1], 10)), 2)}
json.dump({"summary": summ, "per_image": res}, open(f"{a.out}/eval.json", "w"), indent=1); print(json.dumps(summ))
