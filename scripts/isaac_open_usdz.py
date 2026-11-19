#!/usr/bin/env python3
"""Open an EB2F .usdz in the Isaac Sim GUI with the viewport camera inside the p1 hall (isaaclab6 env).
usage: python isaac_open_usdz.py [scene.usdz] [--camfile results/isaac_usd/testcam.npz] [--z_off 0.19]"""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("usdz", nargs="?", default=os.path.expanduser("~/gsplat_eb/results/isaac_usd/EB2F_v2.usdz"))
ap.add_argument("--camfile", default=os.path.expanduser("~/gsplat_eb/results/isaac_usd/testcam.npz"))
ap.add_argument("--z_off", type=float, default=0.19)
a = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "width": 1600, "height": 900})
import numpy as np, omni.usd
from isaacsim.core.utils.viewports import set_camera_view

omni.usd.get_context().open_stage(a.usdz)
while omni.usd.get_context().get_stage_loading_status()[2] > 0:
    app.update()
z = np.load(a.camfile)
eye = z["c"] + np.array([0, 0, a.z_off])
fwd = z["c2w_R"][:, 2]
for _ in range(10):
    app.update()
set_camera_view(eye=eye, target=eye + 5.0 * fwd, camera_prim_path="/OmniverseKit_Persp")
print("EB2F loaded; viewport camera at", eye.round(2))
while app.is_running():
    app.update()
app.close()
