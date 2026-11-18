#!/usr/bin/env python3
"""Structure check of a floor reconstruction (colmap env): top view of SfM points (wall height band) per place,
camera paths, and wall straightness/thickness along the main corridors.
usage: structure_check.py <model_dir> <out_png>"""
import sys, os, numpy as np

if not sys.argv[1].endswith(".npz"):        # colmap env: export, then re-run this script in the gs env
    import pycolmap
    rec = pycolmap.Reconstruction(sys.argv[1])
    pl_img = {i: rec.image(i).name[:2] for i in rec.reg_image_ids()}
    P, PL = [], []
    for p in rec.points3D.values():
        if p.error > 1.5 or p.track.length() < 4:
            continue
        v = np.unique([pl_img[e.image_id] for e in p.track.elements])
        P.append(p.xyz); PL.append(v[0] if len(v) == 1 else "multi")
    C = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    CP = np.array([pl_img[i] for i in rec.reg_image_ids()])
    np.savez(sys.argv[2] + ".npz", P=np.array(P), PL=np.array(PL), C=C, CP=CP)
    sys.exit(0)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = np.load(sys.argv[1]); P, PL, C, CP = Z["P"], Z["PL"], Z["C"], Z["CP"]
band = (P[:, 2] > 0.3) & (P[:, 2] < 2.0)
col = {"p1": "tab:blue", "p2": "tab:green", "p3": "tab:red", "p4": "orange", "multi": "white"}
fig, ax = plt.subplots(figsize=(26, 12)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
for p, c in col.items():
    m = band & (PL == p)
    ax.scatter(P[m, 0], P[m, 1], s=0.3 if p != "multi" else 1.5, c=c, alpha=0.5, linewidths=0, label=f"{p}: {m.sum()} pts")
for p, c in col.items():
    if p != "multi":
        ax.plot(C[CP == p, 0], C[CP == p, 1], ".", ms=1, c=c)
ax.set_aspect("equal"); ax.set_xlim(-20, 52); ax.set_ylim(-21, 9); ax.grid(alpha=0.2)
ax.legend(loc="lower left", markerscale=10, facecolor="0.15", labelcolor="w", fontsize=13)
ax.set_title(f"{os.path.basename(sys.argv[1])}: SfM points 0.3-2.0 m (white = seen from 2+ places)", color="w", fontsize=16)
plt.savefig(sys.argv[2], dpi=50, facecolor="black")

# wall straightness: for each long horizontal corridor wall, robust line fit and residual spread in 2 m bins
def wall_stats(name, x0, x1, ymin, ymax):
    m = band & (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > ymin) & (P[:, 1] < ymax)
    Q = P[m]
    if len(Q) < 200:
        print(f"  {name}: too few points"); return
    A = np.c_[Q[:, 0], np.ones(len(Q))]
    k, b = np.linalg.lstsq(A, Q[:, 1], rcond=None)[0]
    for _ in range(3):
        r = Q[:, 1] - (k * Q[:, 0] + b); w = np.abs(r) < 0.15
        k, b = np.linalg.lstsq(A[w], Q[w, 1], rcond=None)[0]
    r = Q[:, 1] - (k * Q[:, 0] + b)
    bins = np.arange(x0, x1, 2.0)
    offs = [np.median(r[(Q[:, 0] >= a) & (Q[:, 0] < a + 2) & (np.abs(r) < 0.4)]) for a in bins
            if ((Q[:, 0] >= a) & (Q[:, 0] < a + 2) & (np.abs(r) < 0.4)).sum() > 30]
    print(f"  {name}: slope {np.degrees(np.arctan(k)):+.2f} deg, wall thickness (IQR) {np.subtract(*np.percentile(r[np.abs(r)<0.4],[75,25]))*100:.1f} cm, "
          f"waviness (2 m-bin offsets) range {np.ptp(offs)*100:.1f} cm")

print("wall straightness (top view):")
wall_stats("p1 west arm north wall", -16, 4, 0.5, 2.5)
wall_stats("p1 west arm south wall", -16, 4, -5.5, -3.5)
wall_stats("p4 corridor north wall", 9, 43, 1.2, 3.2)
wall_stats("p4 corridor south wall", 9, 43, -0.8, 1.0)
wall_stats("p2 corridor north wall", 9, 43, -10.0, -8.4)
wall_stats("p2 corridor south wall", 9, 43, -11.8, -10.2)
print(f"multi-place points: {(PL=='multi').sum()} of {len(PL)}")
