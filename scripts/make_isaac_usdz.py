#!/usr/bin/env python3
"""Pack the 2F Gaussian model + collision meshes into one .usdz for Isaac Sim (isaaclab6 env, pxr only).

/World                      Z up, metres; translated so the floor is at z ~ 0
  /World/EB2F/Gaussians     ParticleField3DGaussianSplat (OpenUSD 26.03 schema / omni.usd.schema.usd_particle_field),
                            written with numpy -> Vt (binary crate), SH degree 3
  /World/EB2F/Collision     Floor + Obstacles meshes: invisible, PhysicsCollisionAPI, triangle-mesh approximation
  /World/PhysicsScene       gravity -Z
usage: make_isaac_usdz.py <ckpt.pt> <collision_prefix> <out.usdz> [--half] [--crop xmin,xmax,ymin,ymax] [--z_off 0.19]"""
import argparse, os, shutil, numpy as np, torch
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Vt, Gf, UsdUtils

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("coll"); ap.add_argument("out")
ap.add_argument("--half", action="store_true", help="half precision scales/orientations/opacities/SH (positions stay float)")
ap.add_argument("--sh", type=int, default=3, help="SH degree to keep (0..3)")
ap.add_argument("--crop", default="", help="xmin,xmax,ymin,ymax (m, model frame)")
ap.add_argument("--chunks", type=int, default=4, help="split into this many prims along x (renderer fails on one >2 GB array)")
ap.add_argument("--z_off", type=float, default=0.19, help="lift so the floor (model z ~ -0.19) is at 0")
a = ap.parse_args()

work = a.out[:-5] + "_parts"
os.makedirs(work, exist_ok=True)
sp = torch.load(a.ckpt, map_location="cpu", weights_only=False)["splats"]
M = sp["means"].numpy().astype(np.float32)
keep = np.ones(len(M), bool)
if a.crop:
    x0, x1, y0, y1 = map(float, a.crop.split(","))
    keep = (M[:, 0] >= x0) & (M[:, 0] <= x1) & (M[:, 1] >= y0) & (M[:, 1] <= y1)
M = M[keep]
S = torch.exp(sp["scales"][keep]).numpy().astype(np.float32)
Q = torch.nn.functional.normalize(sp["quats"][keep], dim=1).numpy().astype(np.float32)   # w x y z
O = torch.sigmoid(sp["opacities"][keep]).numpy().astype(np.float32)
nk = (a.sh + 1) ** 2
SH = torch.cat([sp["sh0"][keep], sp["shN"][keep][:, :nk - 1]], 1).numpy().astype(np.float32)   # (N, nk, 3)
N = len(M)
print(f"{N:,} Gaussians, SH degree {a.sh} ({nk} coeffs), {'half' if a.half else 'float'}")

# ---- Gaussians layer
gpath = os.path.join(work, "gaussians.usdc")
st = Usd.Stage.CreateNew(gpath)
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
grp = UsdGeom.Xform.Define(st, "/Gaussians"); st.SetDefaultPrim(grp.GetPrim())
T = Sdf.ValueTypeNames
order = np.argsort(M[:, 0], kind="stable")
for k, idx in enumerate(np.array_split(order, a.chunks)):
    pr = st.DefinePrim(f"/Gaussians/Part{k}", "ParticleField3DGaussianSplat")
    Mk = np.ascontiguousarray(M[idx])
    pr.CreateAttribute("positions", T.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(Mk))
    q_xyzw = np.ascontiguousarray(Q[idx][:, [1, 2, 3, 0]])
    SHk = np.ascontiguousarray(SH[idx].reshape(-1, 3))
    if a.half:
        pr.CreateAttribute("scalesh", T.Half3Array).Set(Vt.Vec3hArray.FromNumpy(S[idx].astype(np.float16)))
        pr.CreateAttribute("orientationsh", T.QuathArray).Set(Vt.QuathArray.FromNumpy(q_xyzw.astype(np.float16)))
        pr.CreateAttribute("opacitiesh", T.HalfArray).Set(Vt.HalfArray.FromNumpy(O[idx].astype(np.float16)))
        sh = pr.CreateAttribute("radiance:sphericalHarmonicsCoefficientsh", T.Half3Array)
        sh.Set(Vt.Vec3hArray.FromNumpy(SHk.astype(np.float16)))
    else:
        pr.CreateAttribute("scales", T.Float3Array).Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(S[idx])))
        pr.CreateAttribute("orientations", T.QuatfArray).Set(Vt.QuatfArray.FromNumpy(q_xyzw))
        pr.CreateAttribute("opacities", T.FloatArray).Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(O[idx])))
        sh = pr.CreateAttribute("radiance:sphericalHarmonicsCoefficients", T.Float3Array)
        sh.Set(Vt.Vec3fArray.FromNumpy(SHk))
    sh.SetMetadata("elementSize", nk); sh.SetMetadata("interpolation", "vertex")
    pr.CreateAttribute("radiance:sphericalHarmonicsDegree", T.Int, custom=False).Set(a.sh)
    pr.CreateAttribute("extent", T.Float3Array).Set(Vt.Vec3fArray.FromNumpy(np.stack([Mk.min(0), Mk.max(0)])))
    print(f"  Part{k}: {len(idx):,} Gaussians, x {Mk[:, 0].min():.1f}..{Mk[:, 0].max():.1f}, SH array {SHk.nbytes / 1e9:.2f} GB")
st.GetRootLayer().Save()
print(f"gaussians layer: {os.path.getsize(gpath) / 1e9:.2f} GB")

# ---- collision layer
import open3d as o3d
cpath = os.path.join(work, "collision.usdc")
cs = Usd.Stage.CreateNew(cpath)
UsdGeom.SetStageUpAxis(cs, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(cs, 1.0)
root = UsdGeom.Scope.Define(cs, "/Collision"); cs.SetDefaultPrim(root.GetPrim())
crop = [float(v) for v in a.crop.split(",")] if a.crop else None
for name in ["floor", "obstacles"]:
    m = o3d.io.read_triangle_mesh(f"{a.coll}_{name}.ply")
    V = np.asarray(m.vertices).astype(np.float32); F = np.asarray(m.triangles).astype(np.int32)
    if crop:
        vin = (V[:, 0] >= crop[0]) & (V[:, 0] <= crop[1]) & (V[:, 1] >= crop[2]) & (V[:, 1] <= crop[3])
        F = F[vin[F].all(1)]
        used = np.unique(F); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
        V, F = V[used], remap[F].astype(np.int32)
    mesh = UsdGeom.Mesh.Define(cs, f"/Collision/{name.capitalize()}")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([V.min(0), V.max(0)])))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)          # Gaussians are the visuals; physics only
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    print(f"collision {name}: {len(F):,} triangles")
cs.GetRootLayer().Save()

# ---- root layer
rpath = os.path.join(work, "eb2f.usda")
rs = Usd.Stage.CreateNew(rpath)
UsdGeom.SetStageUpAxis(rs, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(rs, 1.0)
world = UsdGeom.Xform.Define(rs, "/World"); rs.SetDefaultPrim(world.GetPrim())
eb = UsdGeom.Xform.Define(rs, "/World/EB2F")
eb.AddTranslateOp().Set(Gf.Vec3d(0, 0, a.z_off))
rs.DefinePrim("/World/EB2F/Gaussians").GetReferences().AddReference("./gaussians.usdc")
rs.DefinePrim("/World/EB2F/Collision").GetReferences().AddReference("./collision.usdc")
ps = UsdPhysics.Scene.Define(rs, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); ps.CreateGravityMagnitudeAttr(9.81)
rs.GetRootLayer().Save()

if os.path.exists(a.out):
    os.remove(a.out)
ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(rpath), a.out)
print(f"usdz {'OK' if ok else 'FAILED'}: {a.out} {os.path.getsize(a.out) / 1e9:.2f} GB")
