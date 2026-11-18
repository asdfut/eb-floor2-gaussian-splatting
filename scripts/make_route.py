#!/usr/bin/env python3
"""Video route (loop over the whole floor) from a floor dataset's camera positions (gs env).
Corridor centre lines are estimated from the cameras of each place; layout: p1 west (arm + N-S corridor),
p4 north, p3 east, p2 south. usage: make_route.py <floor_dataset> <out.npy>"""
import sys, numpy as np
from pycolmap import SceneManager
m = SceneManager(sys.argv[1] + "/sparse/0/"); m.load_images()
C = np.array([-m.images[i].R().T @ m.images[i].tvec for i in m.images]); pl = np.array([m.images[i].name[:2] for i in m.images])


def lane(p, axis, lo, hi):          # median cross-axis coordinate along the corridor
    sel = (pl == p) & (C[:, axis] > lo) & (C[:, axis] < hi)
    return float(np.median(C[sel, 1 - axis]))


x4 = C[pl == "p4", 0]; x2 = C[pl == "p2", 0]; y3 = C[pl == "p3", 1]
p4w, p4e = np.percentile(x4, 3) + 1.5, np.percentile(x4, 97) - 1.5
p2w, p2e = np.percentile(x2, 3) + 1.5, np.percentile(x2, 97) - 1.5
y4w, y4e = lane("p4", 0, p4w, p4w + 6), lane("p4", 0, p4e - 6, p4e)
y2w, y2e = lane("p2", 0, p2w, p2w + 6), lane("p2", 0, p2e - 6, p2e)
x3 = lane("p3", 1, np.percentile(y3, 30), np.percentile(y3, 70))
p1 = C[pl == "p1"]
ns_x = float(np.median(p1[(p1[:, 0] > p4w - 4) & (p1[:, 0] < p4w) & (p1[:, 1] < y4w - 2) & (p1[:, 1] > y2w + 2), 0]))
arm = p1[(p1[:, 0] < ns_x - 4) & (p1[:, 1] > y2w + 3) & (p1[:, 1] < y4w + 2)]
arm_s, arm_n = np.percentile(arm[:, 1], 10), np.percentile(arm[:, 1], 90)
arm_w = np.percentile(arm[:, 0], 2) + 1.0
route = [(arm_w, arm_s), (ns_x - 2.5, arm_s), (ns_x, arm_s + 1.0), (ns_x, y4w - 0.6), (p4w, y4w), (p4e, y4e),
         (x3, y4e - 0.8), (x3, y2e + 0.8), (p2e, y2e), (p2w, y2w), (ns_x, y2w + 0.8), (ns_x, arm_n - 1.0),
         (ns_x - 2.5, arm_n), (arm_w, arm_n)]
np.save(sys.argv[2], np.array(route))
print("route waypoints:", np.round(route, 2).tolist())
