#!/usr/bin/env python3
"""Lidar surface model for the trainer's --wall-normals regulariser (gs env).

From the lidar points of the given datasets: per 5 cm voxel, the local plane (PCA normal over a ~12 cm
neighbourhood) and the centroid. Only planar voxels are kept (walls, ceiling, floor, furniture faces).
Stored sparsely as sorted linear voxel keys so the trainer can look them up with torch.searchsorted.
usage: make_surface_normals.py <ds1,ds2,...> <out.npz> [voxel=0.05]"""
import sys, numpy as np, open3d as o3d
from pycolmap import SceneManager

dss, out = sys.argv[1].split(","), sys.argv[2]
vox = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
P = np.concatenate([(lambda m: (m.load_points3D(), m.points3D)[1])(SceneManager(d + "/sparse/0/")) for d in dss])
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.02)
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=40))
X = np.asarray(pc.points); N = np.asarray(pc.normals)
# planarity: eigenvalue ratio of the local covariance (keep clearly planar neighbourhoods)
tree = o3d.geometry.KDTreeFlann(pc)
org = X.min(0) - 1.0
key3 = np.floor((X - org) / vox).astype(np.int64)
shape = key3.max(0) + 2
lin = (key3[:, 0] * shape[1] + key3[:, 1]) * shape[2] + key3[:, 2]
order = np.argsort(lin); lin, X, N = lin[order], X[order], N[order]
uk, start, cnt = np.unique(lin, return_index=True, return_counts=True)
# per voxel: centroid + average normal (sign-aligned to the first one)
cen = np.add.reduceat(X, start, axis=0) / cnt[:, None]
ref = N[start]
Ns = N * np.sign(np.einsum("ij,ij->i", N, np.repeat(ref, cnt, axis=0)))[:, None]
nrm = np.add.reduceat(Ns, start, axis=0); L = np.linalg.norm(nrm, axis=1)
coh = L / cnt                                                     # 1 = all normals agree (planar)
nrm = nrm / np.maximum(L, 1e-9)[:, None]
keep = (cnt >= 2) & (coh > 0.9)
pk, pn, pc_ = uk[keep], nrm[keep], cen[keep]
# dilate: every voxel within DIL voxels of a planar voxel gets the plane of the nearest planar voxel, so Gaussians
# floating up to ~DIL*vox in front of a wall are also pulled onto it
DIL = int(sys.argv[4]) if len(sys.argv) > 4 else 2
k3 = np.stack([pk // (shape[1] * shape[2]), (pk // shape[2]) % shape[1], pk % shape[2]], 1)
offs = np.array([(i, j, k) for i in range(-DIL, DIL + 1) for j in range(-DIL, DIL + 1) for k in range(-DIL, DIL + 1)
                 if i * i + j * j + k * k <= DIL * DIL])
cand = np.unique(((k3[:, None, :] + offs[None]).reshape(-1, 3)) @ np.array([shape[1] * shape[2], shape[2], 1]))
c3 = np.stack([cand // (shape[1] * shape[2]), (cand // shape[2]) % shape[1], cand % shape[2]], 1)
from scipy.spatial import cKDTree
_, nn = cKDTree(k3).query(c3)
np.savez_compressed(out, keys=cand, normals=pn[nn].astype(np.float32), centroids=pc_[nn].astype(np.float32),
                    origin=org.astype(np.float32), voxel=vox, shape=shape)
print(f"{out}: {keep.sum()} planar voxels of {len(uk)} ({vox} m), dilated by {DIL} voxels -> {len(cand)} lookup voxels")
