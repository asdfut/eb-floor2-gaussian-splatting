#!/usr/bin/env python3
"""Release PLYs of a merged model (gs env): one SuperSplat-compressed PLY + standard 3DGS PLY split at x=SPLIT into
west/east halves (GitHub release assets must be < 2 GB each). usage: export_release_ply.py <ckpt> <out_dir> <name> [split_x=23.9]"""
import os, sys, torch
from gsplat.exporter import export_splats
ck, out, name = sys.argv[1:4]; sx = float(sys.argv[4]) if len(sys.argv) > 4 else 23.9
sp = torch.load(ck, map_location="cpu", weights_only=False)["splats"]; os.makedirs(out, exist_ok=True)
def ex(k, fmt, f):
    export_splats(means=sp["means"][k], scales=sp["scales"][k], quats=sp["quats"][k], opacities=sp["opacities"][k],
                  sh0=sp["sh0"][k], shN=sp["shN"][k], format=fmt, save_to=os.path.join(out, f))
    print(f, int(k.sum()), f"{os.path.getsize(os.path.join(out, f))/1e9:.2f} GB", flush=True)
allk = torch.ones(len(sp["means"]), dtype=torch.bool); w = sp["means"][:, 0] < sx
ex(allk, "ply_compressed", f"{name}_compressed.ply"); ex(w, "ply", f"{name}_west.ply"); ex(~w, "ply", f"{name}_east.ply")
