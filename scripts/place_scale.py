#!/usr/bin/env python3
"""Along-corridor scale calibration of the T265 trajectories from the reference layout plot (user constraint).
For p2/p4 (x axis) and p3 (y axis) in the floor frame: x' = fig_min + b * (x - our_min), b = fig_len / our_len,
from the camera-path extents. Saves data/floor2/place_scale.npz (used by build_floor2.py)."""
import numpy as np
D = "data/floor2"
Ts = dict(np.load(f"{D}/place_T.npz"))
FIG = {"p2": (0, 6.8, 47.3), "p4": (0, 7.5, 45.5), "p3": (1, -17.3, 3.5)}   # axis, min, max of the camera paths
out = {}
for p, (k, fmin, fmax) in FIG.items():
    C = np.load(f"{D}/walls_{p}.npz")["cams"]; C = C @ Ts[p][:3, :3].T + Ts[p][:3, 3]
    omin, omax = C[:, k].min(), C[:, k].max()
    b = (fmax - fmin) / (omax - omin)
    out[p] = np.array([k, omin, fmin, b])
    print(f"{p}: axis {'xy'[k]}, ours {omin:.2f}..{omax:.2f} ({omax-omin:.2f} m) -> {fmin}..{fmax} ({fmax-fmin:.2f} m), scale {b:.4f}")
np.savez(f"{D}/place_scale.npz", **out)
