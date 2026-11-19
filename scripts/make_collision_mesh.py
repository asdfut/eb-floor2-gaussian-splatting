#!/usr/bin/env python3
"""Collision mesh of the 2F floor for Isaac Sim (gs env).

floor  : height field from the v2 Gaussian model itself (opaque, flat Gaussians in the floor band, per cell median),
         so the collision floor coincides with the rendered floor (the lidar maps had their floor carved out and
         replaced by a synthetic one during preprocessing, so lidar cannot give the floor).
walls / furniture / ceiling : lidar points of the four places (data/EB_<p>_liod_merged, rigid place_T -> floor frame,
         synthetic floor points removed), 5 cm occupancy, isolated voxels dropped, 1 voxel dilation, marching cubes,
         quadric decimation.
usage: make_collision_mesh.py <ckpt_merged.pt> <dataset_dir> <out.ply> [target_tris=600000]"""
import os, sys, numpy as np, torch, open3d as o3d
from pycolmap import SceneManager
from scipy import ndimage
from skimage.measure import marching_cubes

ck, ds, out = sys.argv[1:4]
target = int(sys.argv[4]) if len(sys.argv) > 4 else 600000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOX, CELL = 0.05, 0.10

# ---- camera footprint (where the recording went)
m = SceneManager(ds + "/sparse/0/"); m.load_images()
C = np.array([-m.images[i].R().T @ m.images[i].tvec for i in m.images])
CP = np.array([int(m.images[i].name[1]) for i in m.images])          # place of each camera (p1..p4)
lo, hi = C[:, :2].min(0) - 6.0, C[:, :2].max(0) + 6.0
nx, ny = np.ceil((hi - lo) / CELL).astype(int)
cam_cells = np.zeros((nx, ny), bool)
ci = np.floor((C[:, :2] - lo) / CELL).astype(int); cam_cells[ci[:, 0], ci[:, 1]] = True

# ---- floor height field from the Gaussians
sp = torch.load(ck, map_location="cpu", weights_only=False)["splats"]
M = sp["means"].numpy(); op = torch.sigmoid(sp["opacities"]).numpy(); sc = torch.exp(sp["scales"]).numpy()
q = torch.nn.functional.normalize(sp["quats"], dim=1).numpy()
w, x, y, z = q.T
# third column of the rotation matrix of each axis; normal = axis of the smallest scale
R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
              np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
              np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)  # (N,3,3)
amin = sc.argmin(1)
nz = np.abs(R[np.arange(len(M)), 2, amin])
sel = (op > 0.3) & (M[:, 2] > -0.32) & (M[:, 2] < -0.06) & (nz > 0.8)
# the floor of each place is planar (levelled lidar) -> one height per place = median of the flat opaque Gaussians
# within 1 m of that place's cameras (avoids furniture tops and the mirror Gaussians of the glossy floor);
# heights blend smoothly over ~1 m at the junctions
from scipy.spatial import cKDTree
tree = cKDTree(C[:, :2])
d, j = tree.query(M[sel, :2])
Fz, Fp = M[sel, 2][d < 1.0], CP[j[d < 1.0]]
zp = {k: float(np.median(Fz[Fp == k])) for k in range(1, 5)}
print("floor height per place (m):", {f"p{k}": round(v, 3) for k, v in zp.items()},
      "n =", [int((Fp == k).sum()) for k in range(1, 5)])
gx_, gy_ = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * CELL, lo[1] + (np.arange(ny) + 0.5) * CELL, indexing="ij")
_, jj = tree.query(np.c_[gx_.ravel(), gy_.ravel()])
Hs = np.vectorize(zp.get)(CP[jj]).reshape(nx, ny).astype(np.float64)
Hs = ndimage.gaussian_filter(Hs, 1.0 / CELL / 2)

# ---- obstacles from lidar
T = dict(np.load(f"{ROOT}/data/floor2_lio/place_T.npz"))
P = []
for p in ["p1", "p2", "p3", "p4"]:
    mm = SceneManager(f"{ROOT}/data/EB_{p}_liod_merged/sparse/0/"); mm.load_points3D()
    X = mm.points3D @ T[p][:3, :3].T + T[p][:3, 3]
    P.append(X[np.abs(X[:, 2] + 0.0252) > 0.004])          # synthetic floor points of the preprocessing
P = np.concatenate(P)
pi = np.floor((P[:, :2] - lo) / CELL).astype(int)
ok = (pi >= 0).all(1) & (pi[:, 0] < nx) & (pi[:, 1] < ny); P, pi = P[ok], pi[ok]
# walkable footprint = free 2D cells (no lidar between 0.15 and 1.5 m above the floor) within 2 m of the camera
# path and connected to it
fl0 = Hs[pi[:, 0], pi[:, 1]]; band = (P[:, 2] - fl0 > 0.15) & (P[:, 2] - fl0 < 1.5)
W = np.zeros((nx, ny), np.int32); np.add.at(W, (pi[band, 0], pi[band, 1]), 1)
wall = ndimage.binary_dilation(W >= 2, iterations=1)
dpath = ndimage.distance_transform_edt(~cam_cells) * CELL
free = ~wall & (dpath < 2.0)
lab, _ = ndimage.label(free)
foot = np.isin(lab, np.unique(lab[cam_cells & free]))
foot = ndimage.binary_closing(foot, iterations=2) | (cam_cells & ~wall)
near = ndimage.binary_dilation(foot, iterations=int(1.0 / CELL))   # keep walls up to 1 m outside the floor
fl = Hs[pi[:, 0], pi[:, 1]]
dz = P[:, 2] - fl
P = P[near[pi[:, 0], pi[:, 1]] & (dz > 0.05) & (dz < 3.3)]
print(f"lidar obstacle points: {len(P)}")

zlo = float(np.nanmin(Hs[foot])) - 0.1
gi = np.floor((P - np.array([lo[0], lo[1], zlo])) / VOX).astype(int)
shape = gi.max(0) + 3
G = np.zeros(shape, np.uint8); np.add.at(G, (gi[:, 0], gi[:, 1], gi[:, 2]), 1)
occ = G > 0
nb = ndimage.convolve(occ.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
occ &= nb >= 4                                                        # isolated / sparse noise
occ = ndimage.binary_dilation(occ, iterations=1)
print(f"occupied voxels: {occ.sum()}")
v, f, _, _ = marching_cubes(np.pad(occ, 1).astype(np.float32), 0.5)
v = (v - 1) * VOX + np.array([lo[0], lo[1], zlo])
obst = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f[:, ::-1].copy()))
obst = obst.simplify_quadric_decimation(target)
obst.remove_degenerate_triangles(); obst.remove_unreferenced_vertices()
print(f"obstacle mesh: {len(obst.triangles)} triangles")

# ---- floor mesh: one quad per footprint cell (2 triangles), shared vertices on the cell grid
F2 = ndimage.binary_dilation(foot, iterations=2)   # floor slightly under the walls
vid = -np.ones((nx + 1, ny + 1), np.int64)
corner = np.zeros((nx + 1, ny + 1), bool)
ii, jj = np.nonzero(F2)
for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    corner[ii + a, jj + b] = True
ci_, cj_ = np.nonzero(corner)
vid[ci_, cj_] = np.arange(len(ci_))
Hp = np.pad(Hs, ((0, 1), (0, 1)), mode="edge")
Hc = ndimage.uniform_filter(Hp, 2, origin=-1)                         # corner = mean of neighbouring cells
fv = np.stack([lo[0] + ci_ * CELL, lo[1] + cj_ * CELL, Hc[ci_, cj_]], 1)
a, b, c, d = vid[ii, jj], vid[ii + 1, jj], vid[ii, jj + 1], vid[ii + 1, jj + 1]
ff = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
floor = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(fv), o3d.utility.Vector3iVector(ff))
floor = floor.simplify_quadric_decimation(max(20000, len(ff) // 8))
print(f"floor mesh: {F2.sum() * CELL * CELL:.0f} m2, {len(floor.triangles)} triangles, "
      f"z range {np.asarray(floor.vertices)[:, 2].min():.3f}..{np.asarray(floor.vertices)[:, 2].max():.3f}")

os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
o3d.io.write_triangle_mesh(out.replace(".ply", "_floor.ply"), floor)
o3d.io.write_triangle_mesh(out.replace(".ply", "_obstacles.ply"), obst)
np.savez(out.replace(".ply", "_floorgrid.npz"), H=Hs, foot=foot, lo=lo, cell=CELL)
print("written", out.replace(".ply", "_{floor,obstacles}.ply"))
