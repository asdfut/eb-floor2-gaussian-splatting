#!/usr/bin/env python3
"""T265 scale check with lidar ranges: in a straight corridor, the metric distance to the two end walls (from single
lidar scans) changes by exactly the true displacement along the corridor. Compare with the T265 displacement.
usage (gs env): scale_check.py BAG_DIR [n_samples]"""
import os, sys, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_to_colmap as B

bag = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
mc = [f for f in os.listdir(bag) if f.endswith(".mcap")][0]
odom_t, odom_p, odom_q, scans = [], [], [], []
with open(os.path.join(bag, mc), "rb") as f:
    rd = make_reader(f, decoder_factories=[DecoderFactory()])
    for sch, ch, msg, dm in rd.iter_decoded_messages(topics=["/t265/odom"]):
        p, q = dm.pose.pose.position, dm.pose.pose.orientation
        odom_t.append(msg.log_time * 1e-9); odom_p.append([p.x, p.y, p.z]); odom_q.append([q.x, q.y, q.z, q.w])
odom_t, odom_p, odom_q = map(np.array, (odom_t, odom_p, odom_q))
# corridor axis from the trajectory, keep the middle 70 % along it
c = odom_p[:, :2].mean(0); _, _, vt = np.linalg.svd(odom_p[:, :2] - c, full_matrices=False); a = vt[0]
x = (odom_p[:, :2] - c) @ a
lo, hi = np.percentile(x, [15, 85])
cand_t = odom_t[(x > lo) & (x < hi)]
want = np.sort(np.random.default_rng(0).choice(cand_t, N, replace=False))
with open(os.path.join(bag, mc), "rb") as f:
    rd = make_reader(f)
    wi = 0
    for sch, ch, msg in rd.iter_messages(topics=["/livox/lidar"]):
        t = msg.log_time * 1e-9
        if wi < len(want) and t >= want[wi]:
            tb, xyz, off, tag = B.parse_livox(msg.data)
            scans.append((t, xyz)); wi += 1
        if wi >= len(want):
            break
rows = []
for t, xyz in scans:
    k = np.searchsorted(odom_t, t)
    Rw = R.from_quat(odom_q[k]).as_matrix()
    P = xyz @ Rw.T                                   # rotation only: metric offsets from the sensor in odom axes
    al = P[:, :2] @ a; cr = P[:, :2] @ np.array([-a[1], a[0]])
    band = (np.abs(cr - np.median(cr[np.abs(al) < 3])) < 1.0) & (P[:, 2] > -0.1) & (P[:, 2] < 1.8)
    alb = al[band]
    if (alb > 2).sum() < 30 or (alb < -2).sum() < 30:
        continue
    dp = np.percentile(alb[alb > 0], 99.5); dm = -np.percentile(alb[alb < 0], 0.5)
    rows.append((t, (odom_p[k, :2] - c) @ a, dp, dm))
rows = np.array(rows)
print(f"{os.path.basename(bag)}: {len(rows)} scans; corridor length from single scans (d+ + d-): "
      f"median {np.median(rows[:,2]+rows[:,3]):.2f} m (spread {np.std(rows[:,2]+rows[:,3]):.2f})")
ratios = []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        dodo = rows[j, 1] - rows[i, 1]
        if abs(dodo) < 6:
            continue
        dtrue = 0.5 * ((rows[i, 2] - rows[j, 2]) - (rows[i, 3] - rows[j, 3]))
        ratios.append(dodo / dtrue)
ratios = np.array(ratios)
print(f"   T265 / lidar displacement ratio (scale error): median {np.median(ratios):.3f}, IQR "
      f"{np.percentile(ratios,25):.3f}..{np.percentile(ratios,75):.3f}  ({len(ratios)} pairs)")
