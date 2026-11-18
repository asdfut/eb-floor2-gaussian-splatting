#!/usr/bin/env python3
"""Impose the reference floor layout (user-supplied plot, frame ~ p1 rotated) on the place transforms:
long corridors p2 / p4 horizontal at fixed y, p3 vertical at fixed x; p1 defines the frame (its west arm is
horizontal in the plot). Only the along-corridor offset is left free and found by a 1D lidar overlap search."""
import numpy as np
from scipy.spatial import cKDTree

D = "data/floor2"
Ts = dict(np.load(f"{D}/place_T_layout_icp.npz"))
W = {p: np.load(f"{D}/walls_{p}.npz") for p in Ts}
# reference layout (read off the plot): axis angle (deg) and lateral coordinate of the camera-path centre line
TARGET = {"p2": (0.0, "y", -10.0), "p4": (0.0, "y", 1.3), "p3": (90.0, "x", 45.7)}
P1_ARM_Y, P1_VERT_X = -1.5, 6.3


def tf(P, T): return P @ T[:3, :3].T + T[:3, 3]


def rotz(a):
    T = np.eye(4); T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]; return T


def axis(P):
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); return c, np.arctan2(vt[0][1], vt[0][0])


# --- frame F: rotate so p1's west arm is horizontal, translate to the plot coordinates
C1 = tf(W["p1"]["cams"], Ts["p1"])[:, :2]
arm = C1[C1[:, 0] < 2]
_, a = axis(arm)
a = (a + np.pi / 2) % np.pi - np.pi / 2
F = rotz(-a)
C1f = tf(np.c_[C1, np.zeros(len(C1))], F)[:, :2]
arm_y = np.median(C1f[C1f[:, 0] < 2][:, 1])
vert_x = np.median(C1f[(C1f[:, 0] > 3) & (C1f[:, 1] > 2)][:, 0])
F[0, 3] += P1_VERT_X - vert_x; F[1, 3] += P1_ARM_Y - arm_y
print(f"frame: rotate p1 by {np.degrees(-a):+.2f} deg, shift {F[:2,3].round(2)}")
out = {"p1": F @ Ts["p1"]}

for p, (ang, lat, val) in TARGET.items():
    T = F @ Ts[p]
    c, phi = axis(tf(W[p]["cams"], T)[:, :2])
    d = np.radians(ang) - phi
    d = (d + np.pi / 2) % np.pi - np.pi / 2            # smallest rotation to the target direction
    Tc = np.eye(4); Tc[:2, 3] = c
    T = Tc @ rotz(d) @ np.linalg.inv(Tc) @ T
    c2 = tf(W[p]["cams"], T)[:, :2]
    k = 1 if lat == "y" else 0
    T[k, 3] += val - np.median(c2[:, k])
    out[p] = T
    print(f"{p}: rotated {np.degrees(d):+.2f} deg about its path centre, lateral -> {lat} = {val}")

# --- 1D along-corridor search against all others (fraction of wall points within 0.15 m of another place)
for sweep in range(2):
    for p, (ang, lat, val) in TARGET.items():
        k = 0 if lat == "y" else 1
        others = np.concatenate([tf(W[q]["walls"], out[q]) for q in out if q != p])[:, :2]
        tree = cKDTree(others[::2])
        P = tf(W[p]["walls"], out[p])[::4, :2]
        best = None
        for s in np.arange(-3, 3.001, 0.05):
            Q = P.copy(); Q[:, k] += s
            d, _ = tree.query(Q, distance_upper_bound=0.15)
            sc = np.isfinite(d).mean()
            if best is None or sc > best[0]:
                best = (sc, s)
        out[p][k, 3] += best[1]
        print(f"sweep {sweep} {p}: along-axis shift {best[1]:+.2f} m, overlap {best[0]:.3f}")
np.savez(f"{D}/place_T.npz", **out)
