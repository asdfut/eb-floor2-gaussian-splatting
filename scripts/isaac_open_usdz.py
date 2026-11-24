#!/usr/bin/env python3
"""Open an EB2F .usdz in the Isaac Sim GUI with the viewport camera inside the p1 hall (isaaclab6 env).
usage: python isaac_open_usdz.py [scene.usdz] [--camfile results/isaac_usd/testcam.npz] [--z_off 0.19]"""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("usdz", nargs="?", default=os.path.expanduser("~/gsplat_eb/results/isaac_usd/EB2F_v2.usdz"))
ap.add_argument("--camfile", default=os.path.expanduser("~/gsplat_eb/results/isaac_usd/testcam.npz"))
ap.add_argument("--z_off", type=float, default=0.19)
ap.add_argument("--shots", default="", help="viewport captures, e.g. 30,90 (seconds after load) -> <usdz dir>/gui_shot_<t>s.png")
a = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "width": 1600, "height": 900})
import numpy as np, omni.usd
from isaacsim.core.utils.viewports import set_camera_view

import time
# open_stage(<usdz>) directly -> the ParticleField Gaussians are not drawn in the GUI viewport (grid only, 160 fps).
# Same as isaac_test_usdz.py instead: new Z-up stage + reference the usdz under /World/Env.
from pxr import UsdGeom
omni.usd.get_context().new_stage()
st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
st.DefinePrim("/World/Env", "Xform").GetReferences().AddReference(os.path.abspath(a.usdz))
for _ in range(20):
    app.update()
g = st.GetPrimAtPath("/World/Env/EB2F/Gaussians")
print("Gaussians prim:", g.GetTypeName(), "valid", g.IsValid(), flush=True)
t0 = time.time(); last = -1
# GUI: the content browser (omniverse:// Nucleus) can keep the loading counter > 0 indefinitely -> give up after 120 s
while omni.usd.get_context().get_stage_loading_status()[2] > 0 and time.time() - t0 < 120:
    app.update()
    if int(time.time() - t0) // 10 != last:
        last = int(time.time() - t0) // 10; print(f"[{time.time() - t0:.0f}s] loading status {omni.usd.get_context().get_stage_loading_status()}", flush=True)
print(f"stage loading wait ended after {time.time() - t0:.0f}s: {omni.usd.get_context().get_stage_loading_status()}", flush=True)
z = np.load(a.camfile)
eye = z["c"] + np.array([0, 0, a.z_off])
fwd = z["c2w_R"][:, 2]
for _ in range(10):
    app.update()
set_camera_view(eye=eye, target=eye + 5.0 * fwd, camera_prim_path="/OmniverseKit_Persp")
print("EB2F loaded; viewport camera at", eye.round(2), flush=True)
shots = sorted(float(x) for x in a.shots.split(",") if x); t1 = time.time(); frames = 0
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
while app.is_running():
    app.update(); frames += 1
    if shots and time.time() - t1 >= shots[0]:
        f = os.path.join(os.path.dirname(os.path.abspath(a.usdz)), f"gui_shot_{shots.pop(0):.0f}s.png")
        capture_viewport_to_file(get_active_viewport(), f)
        print(f"[{time.time() - t1:.0f}s] {frames} frames, viewport capture -> {f}", flush=True)
app.close()
