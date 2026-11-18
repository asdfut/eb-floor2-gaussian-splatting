#!/usr/bin/env python3
"""Exhaustive 2D (x, y, yaw) registration between places from their wall-height lidar points.
Gravity is shared (T265), so only x/y/yaw differ. FFT cross-correlation of occupancy grids over all yaws."""
import sys, numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

RES = 0.1


def grid(P, lo, shape):
    ij = np.floor((P[:, :2] - lo) / RES).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    G = np.zeros(shape, np.float32)
    np.add.at(G, (ij[ok, 0], ij[ok, 1]), 1)
    return (G >= 3).astype(np.float32)          # cells hit by >=3 points = wall


def rot(P, a):
    c, s = np.cos(a), np.sin(a)
    return P[:, :2] @ np.array([[c, -s], [s, c]]).T


def match(src, ref, step_deg=1.0):
    lo = ref.min(0)[:2] - 1; hi = ref.max(0)[:2] + 1
    R = grid(ref, lo, tuple(np.ceil((hi - lo) / RES).astype(int)))
    Rb = gaussian_filter(R, 1.5); Rb /= Rb.max()
    res = []
    for yaw in np.arange(0, 360, step_deg):
        q = rot(src, np.radians(yaw)); c = q.mean(0); q = q - c
        slo = q.min(0) - 0.5; sshape = tuple(np.ceil((q.max(0) + 0.5 - slo) / RES).astype(int))
        S = grid(np.c_[q, np.zeros(len(q))], slo, sshape)
        C = fftconvolve(Rb, S[::-1, ::-1], mode="full")
        k = np.unravel_index(np.argmax(C), C.shape)
        score = C[k] / S.sum()
        # translation: src grid origin goes to ref cell (k - S.shape + 1)
        off = (np.array(k) - np.array(S.shape) + 1) * RES + lo - slo
        res.append((score, yaw, off - 0 + 0, c))
    res.sort(key=lambda r: -r[0])
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    pl = ["p1", "p2", "p3", "p4"]
    W = {p: np.load(f"{d}/walls_{p}.npz")["walls"] for p in pl}
    for p in pl:
        W[p] = W[p][np.random.default_rng(0).permutation(len(W[p]))[:150000]]
    out = {}
    for i, a in enumerate(pl):
        for b in pl[i + 1:]:
            r = match(W[b], W[a])
            best = r[0]
            # second-best peak at a clearly different yaw
            alt = next(x for x in r if min(abs(x[1] - best[1]), 360 - abs(x[1] - best[1])) > 10)
            yaw = np.radians(best[1]); c, s = np.cos(yaw), np.sin(yaw)
            # world = R*(p) - R*... : q = R p - cen ; ref = q + off  =>  t = off - cen
            t = best[2] - best[3]
            T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = t
            out[f"{b}_to_{a}"] = T
            print(f"{b} -> {a}: score {best[0]:.3f} yaw {best[1]:.0f} t {t.round(2)} | next-best {alt[0]:.3f} @ yaw {alt[1]:.0f}", flush=True)
    np.savez(f"{d}/place_T2d.npz", **out)
