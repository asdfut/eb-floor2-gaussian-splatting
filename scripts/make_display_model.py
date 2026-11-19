#!/usr/bin/env python3
"""Display version of a merged floor model for viewing from outside / above (gs env).

Keeps only Gaussians whose centre lies in the dilated lidar surface occupancy (walls, floor, ceiling, furniture)
and that are not nearly transparent. Removes the clutter the photometric training leaves behind walls, above the
ceiling and under the floor (mirror images of the glossy floor). The training model is not modified.
usage: make_display_model.py <ckpt> <surface_grid.npz> <out_dir> [min_opacity=0.05]"""
import os, sys, numpy as np, torch
from gsplat.exporter import export_splats

ck, grid, out = sys.argv[1:4]
min_op = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
sp = torch.load(ck, map_location="cpu", weights_only=False)["splats"]
z = np.load(grid); org, vox, G = z["origin"], float(z["voxel"]), z["grid"]
M = sp["means"].numpy()
idx = np.floor((M - org) / vox).astype(int)
inb = ((idx >= 0) & (idx < np.array(G.shape))).all(1)
inside = np.zeros(len(M), bool); inside[inb] = G[idx[inb, 0], idx[inb, 1], idx[inb, 2]]
op = torch.sigmoid(sp["opacities"]).numpy()
sc = torch.exp(sp["scales"]).numpy(); smax, smin = sc.max(1), sc.min(1)
needle = (smax > 0.3) & (smax / np.maximum(smin, 1e-6) > 15)          # long thin spikes sticking out of surfaces
keep = inside & (op >= min_op) & ~needle
print(f"needles (max scale > 0.3 m, aspect > 15): {int((needle & inside & (op >= min_op)).sum())}")
print(f"{len(M)} Gaussians: off-surface {int((~inside).sum())}, transparent {int(((op < min_op) & inside).sum())} -> keep {int(keep.sum())}")
k = torch.from_numpy(keep)
sp = {n: v[k] for n, v in sp.items()}
os.makedirs(os.path.join(out, "ckpts"), exist_ok=True)
torch.save({"step": 0, "splats": sp}, os.path.join(out, "ckpts", "ckpt_display.pt"))
export_splats(means=sp["means"], scales=sp["scales"], quats=sp["quats"], opacities=sp["opacities"],
              sh0=sp["sh0"], shN=sp["shN"], format="ply", save_to=os.path.join(out, "floor2_display.ply"))
print("written", out)
