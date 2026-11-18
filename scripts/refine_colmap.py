#!/usr/bin/env python3
"""Refine T265(+lidar-aligned) camera poses with COLMAP: SIFT features, pose-guided pair
selection, triangulation from the known poses, then bundle adjustment anchored by position
priors (the T265/lidar camera centres) so the solution stays in the odometry frame.

Runs in the `colmap` conda env (official pycolmap 4.x).
usage: refine_colmap.py <dataset_dir (from merge_heights.py)> [--rounds 3]
writes <dataset_dir>/colmap/{database.db, pairs.txt, init/, refined/} and refined.npz
"""
import argparse, os, struct, time
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def read_ours(sparse):
    """Read the cameras/images written by bag_to_colmap.write_colmap (no 2D points)."""
    cams = {}
    with open(os.path.join(sparse, "cameras.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            cams[cid] = (w, h, struct.unpack("<4d", f.read(32)))
    imgs = []
    with open(os.path.join(sparse, "images.bin"), "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i4d3di", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            f.read(8)
            imgs.append((iid, name.decode(), cid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz])))
    return cams, imgs


def apply_cam_rot(imgs, rotvec_deg):
    """c2w <- c2w @ Rfix (camera-frame rotation, e.g. a camera mounting-angle error)."""
    Rf = R.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    out = []
    for iid, name, cid, q, t in imgs:
        Rw = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        Rn = Rf.T @ Rw
        x, y, z, w = R.from_matrix(Rn).as_quat()
        out.append((iid, name, cid, np.array([w, x, y, z]), Rf.T @ t))
    return out


def pose_changes(rec, imgs):
    """per image: camera-frame rotation change (rotvec deg), position change (m), #3D points."""
    by_name = {x[1]: x for x in imgs}
    names, c2ws, rv, dp, nobs = [], [], [], [], []
    for iid in sorted(rec.reg_image_ids()):
        im = rec.image(iid)
        cw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        T = np.eye(4)
        T[:3, :3] = cw.rotation.matrix()
        T[:3, 3] = cw.translation
        c2w = np.linalg.inv(T)
        _, _, _, q, t = by_name[im.name]
        R0 = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # world->cam prior
        rv.append(np.degrees(R.from_matrix(R0 @ c2w[:3, :3]).as_rotvec()))
        dp.append(np.linalg.norm(c2w[:3, 3] + R0.T @ t))
        names.append(im.name)
        c2ws.append(c2w)
        nobs.append(im.num_points3D)
    return names, np.array(c2ws), np.array(rv), np.array(dp), np.array(nobs)


def write_text_model(path, cams, imgs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "cameras.txt"), "w") as f:
        for cid, (w, h, p) in cams.items():
            f.write(f"{cid} PINHOLE {w} {h} {' '.join(map(str, p))}\n")
    with open(os.path.join(path, "images.txt"), "w") as f:
        for iid, name, cid, q, t in imgs:
            f.write(f"{iid} {' '.join(map(str, q))} {' '.join(map(str, t))} {cid} {name}\n\n")
    open(os.path.join(path, "points3D.txt"), "w").close()


def centers_dirs(imgs):
    C, D = [], []
    for _, _, _, q, t in imgs:
        Rw = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # world->cam
        C.append(-Rw.T @ t)
        D.append(Rw.T @ np.array([0, 0, 1.0]))
    return np.array(C), np.array(D)


def make_pairs(imgs, radius, max_angle, k, seq, cross_k=0):
    C, D = centers_dirs(imgs)
    names = [x[1] for x in imgs]
    sess = ["_".join(n.split("_")[:-1]) for n in names]       # h1 / p1_h1 (recording)
    place = [n.split("_")[0] for n in names]                  # h1 / p1 (with --cross_k: other places)
    tree = cKDTree(C)
    cos_max = np.cos(np.radians(max_angle))
    pairs = set()
    for i in range(len(imgs)):
        idx = tree.query_ball_point(C[i], radius)
        cand = [j for j in idx if j != i and D[i] @ D[j] > cos_max]
        cand.sort(key=lambda j: np.linalg.norm(C[i] - C[j]))
        for j in cand[:k]:
            pairs.add((min(i, j), max(i, j)))
        if cross_k:
            for j in [j for j in cand if place[j] != place[i]][:cross_k]:
                pairs.add((min(i, j), max(i, j)))
        for j in range(i + 1, min(len(imgs), i + seq + 1)):  # sequential neighbours, same session
            if sess[j] == sess[i]:
                pairs.add((i, j))
    return [(names[i], names[j]) for i, j in sorted(pairs)], C


def ba_with_priors(rec, prior_C, prior_sigma, log):
    """Bundle adjustment of poses + points, intrinsics fixed, anchored by position priors."""
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = False
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    opts.print_summary = False
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = 1.0
    opts.ceres.solver_options.max_num_iterations = 100
    cfg = pycolmap.BundleAdjustmentConfig()
    for iid in rec.reg_image_ids():
        cfg.add_image(iid)
    priors = []
    for iid in rec.reg_image_ids():
        img = rec.image(iid)
        pp = pycolmap.PosePrior()
        pp.position = prior_C[img.name]
        pp.position_covariance = np.eye(3) * prior_sigma ** 2
        pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        pp.corr_data_id = img.data_id
        priors.append(pp)
    popts = pycolmap.PosePriorBundleAdjustmentOptions()
    popts.alignment_ransac.max_error = 0.0  # already in the prior frame: no re-alignment
    adj = pycolmap.create_pose_prior_bundle_adjuster(opts, popts, cfg, priors, rec)
    summ = adj.solve()
    log(f"    BA: {summ.brief_report() if hasattr(summ,'brief_report') else summ}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--max_angle", type=float, default=60.0)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--prior_sigma", type=float, default=0.15, help="T265/lidar camera-centre prior std (m)")
    ap.add_argument("--max_features", type=int, default=8192)
    ap.add_argument("--max_rot", type=float, default=3.0, help="drop images whose rotation moved more (deg)")
    ap.add_argument("--max_pos", type=float, default=0.5, help="... or position moved more (m)")
    ap.add_argument("--min_obs", type=int, default=80, help="... or that see fewer 3D points")
    ap.add_argument("--cross_k", type=int, default=0, help="extra pairs per image with images of other places")
    ap.add_argument("--skip_mount", action="store_true", help="poses are already mount-corrected: no stage A")
    ap.add_argument("--match_only", action="store_true", help="stop after feature extraction + matching")
    ap.add_argument("--relax_first", type=float, default=0, help="round 0 triangulation reproj threshold (px, 0 = default)"
                    " so tracks can join images whose priors are still a little inconsistent (e.g. different places)")
    a = ap.parse_args()
    W = os.path.join(a.dataset, "colmap")
    os.makedirs(W, exist_ok=True)
    logf = open(os.path.join(W, "refine_log.txt"), "a")

    def log(s):
        s = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cams, imgs = read_ours(os.path.join(a.dataset, "sparse", "0"))
    init = os.path.join(W, "init")
    write_text_model(init, cams, imgs)
    rec0 = pycolmap.Reconstruction(init)
    log(f"init model: {rec0.num_cameras()} cameras, {rec0.num_reg_images()} images")

    db_path = os.path.join(W, "database.db")
    img_dir = os.path.join(a.dataset, "images")
    if not os.path.exists(db_path):
        db = pycolmap.Database.open(db_path)
        for cam in rec0.cameras.values():
            db.write_camera(cam, use_camera_id=True)
        for rig in rec0.rigs.values():
            db.write_rig(rig, use_rig_id=True)
        for fr in rec0.frames.values():
            db.write_frame(fr, use_frame_id=True)
        for im in rec0.images.values():
            db.write_image(im, use_image_id=True)
        db.close()
        eo = pycolmap.FeatureExtractionOptions()
        eo.sift.max_num_features = a.max_features
        eo.num_threads = os.cpu_count()
        log("extracting SIFT ...")
        pycolmap.extract_features(db_path, img_dir, image_names=[x[1] for x in imgs],
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE, extraction_options=eo,
                                  device=pycolmap.Device.cpu)
    db = pycolmap.Database.open(db_path)
    log(f"database: {db.num_images()} images, {db.num_keypoints()} keypoints, {db.num_verified_image_pairs()} verified pairs")
    nverified = db.num_verified_image_pairs()
    db.close()

    pairs, C = make_pairs(imgs, a.radius, a.max_angle, a.k, a.seq, a.cross_k)
    prior_C = {x[1]: c for x, c in zip(imgs, C)}
    pairs_path = os.path.join(W, "pairs.txt")
    with open(pairs_path, "w") as f:
        f.writelines(f"{p} {q}\n" for p, q in pairs)
    if nverified == 0:
        log(f"matching {len(pairs)} pose-guided pairs ...")
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = os.cpu_count()
        po = pycolmap.ImportedPairingOptions()
        po.match_list_path = pairs_path
        pycolmap.match_image_pairs(db_path, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
        db = pycolmap.Database.open(db_path)
        log(f"verified pairs: {db.num_verified_image_pairs()} / {len(pairs)}")
        db.close()

    if a.match_only:
        log("match_only: done")
        return

    def run_rounds(rec, rounds, tag):
        for r in range(rounds):
            out = os.path.join(W, f"{tag}{r}")
            os.makedirs(out, exist_ok=True)
            topt = pycolmap.IncrementalPipelineOptions()
            if a.relax_first and r == 0:
                px = a.relax_first
                topt.triangulation.create_max_angle_error = px / 3
                topt.triangulation.continue_max_angle_error = px / 3
                topt.triangulation.merge_max_reproj_error = px
                topt.triangulation.complete_max_reproj_error = px
                topt.mapper.filter_max_reproj_error = px
            rec = pycolmap.triangulate_points(rec, db_path, img_dir, out, clear_points=True, options=topt)
            log(f"{tag} {r}: triangulated {rec.num_points3D()} points, mean reproj {rec.compute_mean_reprojection_error():.3f}px, "
                f"mean track {rec.compute_mean_track_length():.2f}")
            ba_with_priors(rec, prior_C, a.prior_sigma, log)
            log(f"{tag} {r}: after BA mean reproj {rec.compute_mean_reprojection_error():.3f}px")
            rec.write(out)
        return rec

    # stage A: one round from the raw T265 poses -> estimate the constant camera mounting error
    if a.skip_mount:
        fix = np.zeros(3)
    else:
        recA = run_rounds(rec0, 1, "stageA_")
        _, _, rv, _, nobs = pose_changes(recA, imgs)
        fix = np.median(rv[nobs >= a.min_obs], axis=0)
    log(f"camera mounting correction (camera-frame rotvec, median over images): {fix.round(2)} deg")

    # stage B: apply it to all priors and refine again
    imgs = apply_cam_rot(imgs, fix)
    initB = os.path.join(W, "initB")
    write_text_model(initB, cams, imgs)
    rec = run_rounds(pycolmap.Reconstruction(initB), a.rounds, "stageB_")

    names, c2ws, rv, dp, nobs = pose_changes(rec, imgs)
    ang = np.linalg.norm(rv, axis=1)
    good = (ang < a.max_rot) & (dp < a.max_pos) & (nobs >= a.min_obs)
    log(f"pose change vs (mount-corrected) T265/lidar: position median {np.median(dp)*100:.1f} cm, 95% {np.percentile(dp,95)*100:.1f} cm; "
        f"rotation median {np.median(ang):.2f} deg, 95% {np.percentile(ang,95):.2f} deg")
    for sname in sorted(set(n.split("_")[0] for n in names)):
        m = np.array([n.startswith(sname + "_") for n in names])
        log(f"  {sname}: {m.sum()} images, kept {int((good & m).sum())}  (rot>{a.max_rot}: {int(((ang>=a.max_rot)&m).sum())}, "
            f"pos>{a.max_pos}: {int(((dp>=a.max_pos)&m).sum())}, obs<{a.min_obs}: {int(((nobs<a.min_obs)&m).sum())})")
    dpos, dang = dp, ang
    pts = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()])
    err = np.array([p.error for p in rec.points3D.values()])
    np.savez(os.path.join(a.dataset, "refined.npz"), names=np.array(names), c2ws=np.array(c2ws),
             sfm_xyz=pts, sfm_rgb=rgb, sfm_err=err, dpos=dpos, dang=dang, nobs=nobs, good=good, mount_fix_deg=fix)
    log("done")


if __name__ == "__main__":
    main()
