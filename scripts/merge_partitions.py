#!/usr/bin/env python3
"""Merge per-partition MCMC models (trained in the shared floor frame, --no-normalize-world-space) into one model.

Ownership: every Gaussian is kept only by the partition whose own cameras are closest to it (2D, floor frame),
so overlapping junction regions are not doubled. Writes ckpt + ply, then evaluates on the floor validation split
(every 8th image of EB_floor2_final, same split as the trainer) and saves side-by-side renders.
usage (gs env): merge_partitions.py --parts p1 p2 p3 p4 --out results/EB_floor2_gs
"""
import argparse, glob, json, os
import numpy as np, torch
from PIL import Image
from scipy.spatial import cKDTree
from pycolmap import SceneManager
from gsplat.rendering import rasterization
from gsplat.exporter import export_splats

ROOT = os.path.expanduser("~/gsplat_eb")
KEYS = ["means", "quats", "scales", "opacities", "sh0", "shN"]


def floor_cams(data):
    m = SceneManager(os.path.join(data, "sparse", "0") + "/"); m.load_cameras(); m.load_images()
    out = []
    for i in m.images:
        im = m.images[i]; c = m.cameras[im.camera_id]
        T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
        K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], np.float32)
        out.append((im.name, T, K, c.width, c.height))
    return sorted(out, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["p1", "p2", "p3", "p4"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "EB_floor2_gs"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "EB_floor2_final"))
    ap.add_argument("--near", type=float, default=0.15)
    ap.add_argument("--n_save", type=int, default=40)
    ap.add_argument("--blend_m", type=float, default=1.5, help="cross-fade width at partition boundaries (0 = hard)")
    ap.add_argument("--prefix", default="EB_part_", help="partition dataset/result name prefix")
    ap.add_argument("--ckpt_name", default=None, help="checkpoint file name in each partition (default: latest step)")
    ap.add_argument("--part_map", default=None, help="JSON image name -> partition label (make_part_map.py)")
    ap.add_argument("--carve_r", type=float, default=0.25,
                    help="free-space carving: drop Gaussians within this xy distance of a camera centre, between "
                         "just above the floor and just above the highest camera (the robot's swept volume; 0 = off)")
    ap.add_argument("--floor_z", type=float, default=-0.21)
    ap.add_argument("--carve_zmin", type=float, default=0.15, help="carve only above floor_z + this (keeps the floor)")
    ap.add_argument("--prune_thr", type=float, default=1e-6, help="contribution pruning threshold")
    ap.add_argument("--min_opacity", type=float, default=0.0)
    ap.add_argument("--prune_views", type=int, default=3, help="contribution pruning: use every Nth training view (0 = off)")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(os.path.join(a.out, "ckpts"), exist_ok=True); os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
    cams = floor_cams(a.data)
    C = np.array([np.linalg.inv(T)[:2, 3] for _, T, _, _, _ in cams])
    if a.part_map:
        pm = json.load(open(a.part_map)); owner = np.array([pm.get(n, "") for n, *_ in cams])
    else:
        owner = np.array([n.split("_")[0] for n, *_ in cams])
    tree = cKDTree(C)
    merged = {k: [] for k in KEYS}
    trees_own = {p: cKDTree(C[owner == p]) for p in a.parts}
    trees_oth = {p: cKDTree(C[owner != p]) for p in a.parts}
    log = []
    for p in a.parts:
        ck = os.path.join(ROOT, "results", f"{a.prefix}{p}", "ckpts", a.ckpt_name) if a.ckpt_name else \
            sorted(glob.glob(os.path.join(ROOT, "results", f"{a.prefix}{p}", "ckpts", "ckpt_*_rank0.pt")),
                   key=lambda s: int(os.path.basename(s).split("_")[1]))[-1]
        sp = torch.load(ck, map_location="cpu", weights_only=False)["splats"]
        xy = sp["means"][:, :2].numpy()
        if a.blend_m > 0:
            # soft ownership: cross-fade the two partitions over +-blend_m around the boundary (opacity weight),
            # so a camera crossing a partition boundary sees no sudden switch between the two models
            d_own, _ = trees_own[p].query(xy); d_oth, _ = trees_oth[p].query(xy)
            w = np.clip(0.5 + (d_oth - d_own) / (2 * a.blend_m), 0, 1)
            keep = torch.from_numpy(w > 0.02)
            op = torch.sigmoid(sp["opacities"]) * torch.from_numpy(w).float()
            sp["opacities"] = torch.logit(op.clamp(1e-6, 1 - 1e-6))
        else:
            _, idx = tree.query(xy)
            keep = torch.from_numpy(owner[idx] == p)
        for k in KEYS:
            merged[k].append(sp[k][keep])
        msg = f"{p}: {os.path.basename(ck)} {len(keep)} Gaussians, kept {int(keep.sum())} in own region"
        print(msg); log.append(msg)
    sp = {k: torch.cat(v).to(dev) for k, v in merged.items()}
    print(f"merged: {len(sp['means'])} Gaussians")
    Cz = np.array([np.linalg.inv(T)[:3, 3] for _, T, _, _, _ in cams])
    keep = torch.sigmoid(sp["opacities"]).cpu() >= a.min_opacity
    msg = f"opacity < {a.min_opacity}: removed {int((~keep).sum())}"
    if a.carve_r > 0:
        mxyz = sp["means"].cpu().numpy()
        dxy, _ = cKDTree(Cz[:, :2]).query(mxyz[:, :2])
        inside = (dxy < a.carve_r) & (mxyz[:, 2] > a.floor_z + a.carve_zmin) & (mxyz[:, 2] < Cz[:, 2].max() + 0.2)
        keep &= torch.from_numpy(~inside)
        msg += f"; free-space carving (robot swept volume, r={a.carve_r} m): removed {int(inside.sum())}"
    print(msg); log.append(msg)
    sp = {k: v[keep.to(dev)] for k, v in sp.items()}
    if a.prune_views:
        # contribution pruning: d(sum of rendered alpha)/d(opacity) over the training views ~ how much each
        # Gaussian is ever seen; drop those that never contribute (unseen clutter, fully occluded floaters)
        op = sp["opacities"].clone().requires_grad_(True)
        acc = torch.zeros_like(op)
        colors0 = torch.cat([sp["sh0"], sp["shN"]], 1)
        sc0 = torch.exp(sp["scales"])
        for name, T, K, W, H in cams[::a.prune_views]:
            _, alpha, _ = rasterization(sp["means"], sp["quats"], sc0, torch.sigmoid(op), colors0,
                                        torch.tensor(T, dtype=torch.float32, device=dev)[None],
                                        torch.tensor(np.diag([0.5, 0.5, 1.0]) @ K, dtype=torch.float32, device=dev)[None],
                                        W // 2, H // 2, sh_degree=0,
                                        near_plane=a.near, far_plane=1e10)
            g, = torch.autograd.grad(alpha.sum(), op)
            acc = torch.maximum(acc, g.abs())
        keep = (acc > a.prune_thr).cpu()
        msg = f"contribution pruning: kept {int(keep.sum())} / {len(keep)} Gaussians"
        print(msg); log.append(msg)
        sp = {k: v[keep.to(dev)] for k, v in sp.items()}
    torch.save({"step": 0, "splats": {k: v.cpu() for k, v in sp.items()}}, os.path.join(a.out, "ckpts", "ckpt_merged.pt"))
    export_splats(means=sp["means"], scales=sp["scales"], quats=sp["quats"], opacities=sp["opacities"],
                  sh0=sp["sh0"], shN=sp["shN"], format="ply", save_to=os.path.join(a.out, "floor2_merged.ply"))

    # ---- evaluation on the floor validation split
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    # held-out images: in the validation split (every 8th, sorted) of every partition that contains them,
    # i.e. never used for training by any partition
    train_any, val_any = set(), set()
    for p in a.parts:
        ns = sorted(os.listdir(os.path.join(ROOT, "data", f"{a.prefix}{p}", "images")))
        for k, n in enumerate(ns):
            (val_any if k % 8 == 0 else train_any).add(n)
    held = val_any - train_any
    val = [c for c in cams if c[0] in held]
    print(f"evaluating on {len(val)} held-out images")
    res = {}
    save_every = max(1, len(val) // a.n_save)
    for k, (name, T, K, W, H) in enumerate(val):
        gt = torch.from_numpy(np.asarray(Image.open(os.path.join(a.data, "images", name)).convert("RGB"),
                                         np.float32) / 255).to(dev)
        with torch.no_grad():
            img, _, _ = rasterization(sp["means"], sp["quats"], scales, opac, colors,
                                      torch.tensor(T, dtype=torch.float32, device=dev)[None],
                                      torch.tensor(K, device=dev)[None], W, H, sh_degree=3, near_plane=a.near,
                                      far_plane=1e10)
            img = img[0].clamp(0, 1)
            x, y = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
            ps = float(-10 * torch.log10(((img - gt) ** 2).mean()))
            res[name] = (ps, float(ssim(x, y)), float(lpips(x, y)))
        if k % save_every == 0:
            Image.fromarray((torch.cat([gt, img], 1).cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(a.out, "renders", f"{name[:-4]}.jpg"), quality=90)
    arr = np.array(list(res.values()))
    summ = {"psnr": float(arr[:, 0].mean()), "ssim": float(arr[:, 1].mean()), "lpips": float(arr[:, 2].mean()),
            "num_GS": int(len(sp["means"])), "n_val": len(res)}
    groups = sorted(set("_".join(n.split("_")[:2]) for n in res))
    summ["per_group_psnr"] = {g: float(np.mean([v[0] for n, v in res.items() if n.startswith(g)])) for g in groups}
    json.dump({"summary": summ, "per_image": res, "log": log}, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
