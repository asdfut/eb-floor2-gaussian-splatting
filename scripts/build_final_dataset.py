#!/usr/bin/env python3
"""merged dataset + COLMAP-refined poses (refined.npz) -> final gsplat dataset.

* poses: bundle-adjusted (T265/lidar priors + SIFT tracks + camera-mount correction)
* images: only those whose refined pose is trustworthy (`good` from refine_colmap.py)
* init points: SfM points (reproj err < max_err) + the lidar/floor cloud of the merged dataset
usage (gs env): build_final_dataset.py <merged_dir> <out_dir>
"""
import argparse, os, sys
import numpy as np
from pycolmap import SceneManager  # gsplat's pycolmap fork (gs env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("out")
    ap.add_argument("--max_err", type=float, default=2.0, help="keep SfM points with reproj error below (px)")
    ap.add_argument("--min_obs_strong", type=int, default=150)
    ap.add_argument("--min_obs", type=int, default=80)
    ap.add_argument("--max_rot", type=float, default=3.0)
    ap.add_argument("--max_pos", type=float, default=0.5)
    a = ap.parse_args()
    m = SceneManager(os.path.join(a.merged, "sparse", "0") + "/")
    m.load_cameras(); m.load_images(); m.load_points3D()
    z = np.load(os.path.join(a.merged, "refined.npz"))
    # trust the BA pose when it is well constrained (many 3D points), even if it moved far from the
    # T265 prior (those come in contiguous segments where the odometry drifted); otherwise require
    # it to stay close to the prior
    nobs, dang, dpos = z["nobs"], z["dang"], z["dpos"]
    good = (nobs >= a.min_obs_strong) | ((nobs >= a.min_obs) & (dang < a.max_rot) & (dpos < a.max_pos))
    allnames = np.array([str(x) for x in z["names"]]); C = z["c2ws"][:, :3, 3]
    # spikes: camera far from the median of its +-3 neighbours in the same recording
    sess = np.array(["_".join(x.split("_")[:-1]) for x in allnames])
    spike = np.zeros(len(allnames), bool)
    for s_ in np.unique(sess):
        idx = np.where(sess == s_)[0]; idx = idx[np.argsort(allnames[idx])]
        for k, i in enumerate(idx):
            nb = idx[max(0, k - 3):k + 4]
            if len(nb) >= 4 and np.linalg.norm(C[i, :2] - np.median(C[nb, :2], 0)) > 0.4:
                spike[i] = True
    # corridors (optional <merged>/corridors.json): cameras must lie between the corridor walls
    outside = np.zeros(len(allnames), bool)
    cj = os.path.join(a.merged, "corridors.json")
    if os.path.exists(cj):
        import json
        for cor in json.load(open(cj)):
            k = 0 if cor["axis"] == "x" else 1           # along-corridor coordinate
            u, v = C[:, k], C[:, 1 - k]
            inr = (u > cor["range"][0]) & (u < cor["range"][1])
            t = (u - cor["range"][0]) / (cor["range"][1] - cor["range"][0])
            lo = cor["wall_lo"][0] + t * (cor["wall_lo"][1] - cor["wall_lo"][0]) + cor.get("margin", 0.15)
            hi = cor["wall_hi"][0] + t * (cor["wall_hi"][1] - cor["wall_hi"][0]) - cor.get("margin", 0.15)
            near = inr & (v > lo - 3.0) & (v < hi + 3.0)          # only cameras that belong to this corridor
            bad = near & ((v < lo) | (v > hi))
            outside |= bad
            print(f"  corridor {cor['name']}: {int(bad.sum())} cameras outside the walls")
    print(f"  pose filter: {int(spike.sum())} spikes, {int(outside.sum())} outside corridors")
    good &= ~spike & ~outside
    names = [str(n) for n in z["names"][good]]
    c2ws = z["c2ws"][good]

    cam_ids = sorted(m.cameras)
    cams = [(m.cameras[c].width, m.cameras[c].height,
             np.array([[m.cameras[c].fx, 0, m.cameras[c].cx], [0, m.cameras[c].fy, m.cameras[c].cy], [0, 0, 1]]))
            for c in cam_ids]
    cam_of_img = [cam_ids.index(m.images[m.name_to_image_id[n]].camera_id) for n in names]

    sel = z["sfm_err"] < a.max_err
    sfm, sfm_rgb = z["sfm_xyz"][sel], z["sfm_rgb"][sel]
    lid, lid_rgb = m.points3D, m.point3D_colors
    pts = np.concatenate([sfm, lid]).astype(np.float32)
    rgb = np.concatenate([sfm_rgb, lid_rgb]).astype(np.uint8)

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
    for n in names:
        os.symlink(os.path.realpath(os.path.join(a.merged, "images", n)), os.path.join(img_dir, n))
    B.write_colmap(os.path.join(a.out, "sparse", "0"), cams, cam_of_img, names, c2ws, pts, rgb, [[] for _ in names])
    msg = (f"{a.out}: {len(names)}/{len(good)} images (dropped {int((~good).sum())} with unreliable poses), "
           f"{len(sfm)} SfM + {len(lid)} lidar/floor points; camera mount fix {z['mount_fix_deg'].round(2)} deg")
    open(os.path.join(a.out, "build_log.txt"), "w").write(msg + "\n")
    print(msg)


if __name__ == "__main__":
    main()
