#!/usr/bin/env python3
"""Headless Isaac Sim check of an EB2F .usdz (isaaclab6 env):
1. render from a real recorded camera pose (results/isaac_usd/testcam.npz) -> <out>_rgb.png
2. physics: a sphere dropped on the floor (rest height) and a sphere thrown at a wall (must stop in front of it)
usage: python isaac_test_usdz.py <scene.usdz> <out_prefix> [--z_off 0.19] [--wall_test x,y,vx,vy]"""
import argparse, sys
ap = argparse.ArgumentParser()
ap.add_argument("usdz"); ap.add_argument("out")
ap.add_argument("--z_off", type=float, default=0.19)
ap.add_argument("--drop", default="-10,-1.5")
ap.add_argument("--wall_test", default="-10,-2.0,0,-4.0")
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--camfile", default="results/isaac_usd/testcam.npz")
a = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
import numpy as np, omni.usd, omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdLux, PhysxSchema
import omni.replicator.core as rep
from isaacsim.core.api import World

omni.usd.get_context().new_stage()
st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
env = st.DefinePrim("/World/Env", "Xform"); env.GetReferences().AddReference(a.usdz)
for _ in range(20):
    app.update()
g = st.GetPrimAtPath("/World/Env/EB2F/Gaussians")
print("Gaussians prim:", g.GetTypeName(), "valid", g.IsValid(), "positions",
      len(g.GetAttribute("positions").Get() or []))

# camera at the recorded pose (COLMAP: +Z forward, +Y down -> USD camera: -Z forward, +Y up)
z = np.load(a.camfile)
R = z["c2w_R"] @ np.diag([1.0, -1.0, -1.0]); c = z["c"] + np.array([0, 0, a.z_off])
M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c
cam = UsdGeom.Camera.Define(st, "/World/TestCam")
ap_w = 20.955
cam.CreateHorizontalApertureAttr(ap_w); cam.CreateVerticalApertureAttr(ap_w * float(z["h"]) / float(z["w"]))
cam.CreateFocalLengthAttr(float(z["fx"]) * ap_w / float(z["w"]))
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200))
cam.AddTransformOp().Set(Gf.Matrix4d(*M.T.ravel().tolist()))
rp = rep.create.render_product("/World/TestCam", (int(z["w"]), int(z["h"])))
rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
import time
t0 = time.time()
while omni.usd.get_context().get_stage_loading_status()[2] > 0 and time.time() - t0 < 300:
    app.update()
for k in range(a.frames):
    rep.orchestrator.step(rt_subframes=1)
    img = rgb.get_data()
    m_ = float(np.asarray(img)[..., :3].mean()) if img is not None and np.asarray(img).size else 0.0
    if k % 10 == 9 or m_ > 0:
        print(f"frame {k + 1} t={time.time() - t0:.0f}s mean {m_:.1f}", flush=True)
    if m_ > 0 and k >= 10:
        break
import PIL.Image
PIL.Image.fromarray(np.asarray(img)[..., :3]).save(a.out + "_rgb.png")
print("render saved", a.out + "_rgb.png", "mean", float(np.asarray(img)[..., :3].mean()))

# physics
world = World(stage_units_in_meters=1.0)
from isaacsim.core.api.objects import DynamicSphere
dx, dy = map(float, a.drop.split(","))
s1 = world.scene.add(DynamicSphere("/World/Drop", name="drop", position=np.array([dx, dy, 1.0]), radius=0.15))
wx, wy, vx, vy = map(float, a.wall_test.split(","))
s2 = world.scene.add(DynamicSphere("/World/Throw", name="throw", position=np.array([wx, wy, 0.6]), radius=0.15))
world.reset()
s2.set_linear_velocity(np.array([vx, vy, 0.0]))
traj = []
for i in range(240):
    world.step(render=False)
    if i % 30 == 29:
        traj.append((i + 1, s1.get_world_pose()[0].round(3).tolist(), s2.get_world_pose()[0].round(3).tolist()))
for t in traj:
    print("step", t[0], "drop", t[1], "throw", t[2])
app.close()
