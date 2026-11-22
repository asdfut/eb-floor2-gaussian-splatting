#!/usr/bin/env python3
"""Release PLYs of a merged model (gs env).
  default:  SuperSplat-compressed PLY + standard 3DGS PLY split at x=SPLIT into west/east halves.
  --single: ONE standard 3DGS PLY (full SH degree 3) under GitHub's 2 GB asset limit: if the model is too big, drop the
            most transparent Gaussians (lowest opacity) until it fits. Measured on EB_1121 (12.79M -> 8.45M GS, opacity
            < ~0.02 removed): exposure-fair PSNR 32.33 -> 32.3. Truncating SH instead costs far more (deg 2: -1.5 dB).
usage: export_release_ply.py <ckpt> <out_dir> <name> [--split_x 23.9] [--single] [--max_bytes 2000000000]"""
import argparse, os, torch
from gsplat.exporter import export_splats
ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('name')
ap.add_argument('--split_x', type=float, default=23.9); ap.add_argument('--single', action='store_true')
ap.add_argument('--max_bytes', type=int, default=2_000_000_000)
a = ap.parse_args()
sp = torch.load(a.ckpt, map_location="cpu", weights_only=False)["splats"]; os.makedirs(a.out, exist_ok=True)
def ex(k, fmt, f):
    export_splats(means=sp["means"][k], scales=sp["scales"][k], quats=sp["quats"][k], opacities=sp["opacities"][k],
                  sh0=sp["sh0"][k], shN=sp["shN"][k], format=fmt, save_to=os.path.join(a.out, f))
    print(f, int(k.sum()), f"{os.path.getsize(os.path.join(a.out, f))/1e9:.3f} GB", flush=True)
N = len(sp["means"])
if a.single:
    per = 4 * (3 + 3 + sp["shN"].shape[1] * 3 + 1 + 3 + 4)          # bytes per Gaussian in a float32 3DGS PLY
    nmax = (a.max_bytes - 8192) // per
    k = torch.ones(N, dtype=torch.bool)
    if N > nmax:
        order = torch.argsort(sp["opacities"], descending=True); k = torch.zeros(N, dtype=torch.bool); k[order[:nmax]] = True
        print(f"{N} Gaussians > {nmax} that fit in {a.max_bytes/1e9:.2f} GB: dropping the {N - nmax} most transparent "
              f"(opacity < {torch.sigmoid(sp['opacities'][order[nmax]]).item():.4f})")
    ex(k, "ply", f"{a.name}.ply")
else:
    allk = torch.ones(N, dtype=torch.bool); w = sp["means"][:, 0] < a.split_x
    ex(allk, "ply_compressed", f"{a.name}_compressed.ply"); ex(w, "ply", f"{a.name}_west.ply"); ex(~w, "ply", f"{a.name}_east.ply")
