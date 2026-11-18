"""Top-down render of a dataset's init point cloud + camera path (sanity check for drift)."""
import sys, numpy as np, cv2
from pycolmap import SceneManager
d = sys.argv[1]
m = SceneManager(d + "/sparse/0/"); m.load_images(); m.load_points3D()
P = m.points3D; C = np.array([-m.images[i].R().T @ m.images[i].tvec for i in sorted(m.images)])
band = P[(P[:, 2] > 0.3) & (P[:, 2] < 2.0)]
lo = np.minimum(band.min(0), C.min(0))[:2] - 1; hi = np.maximum(band.max(0), C.max(0))[:2] + 1
res = 0.03; W, H = ((hi - lo) / res).astype(int) + 1
img = np.zeros((H, W), np.float32)
u = ((band[:, 0] - lo[0]) / res).astype(int); v = H - 1 - ((band[:, 1] - lo[1]) / res).astype(int)
np.add.at(img, (v, u), 1)
img = cv2.cvtColor((255 * np.clip(img / 4, 0, 1)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
cu = ((C[:, 0] - lo[0]) / res).astype(int); cv_ = H - 1 - ((C[:, 1] - lo[1]) / res).astype(int)
for i, (a, b) in enumerate(zip(cu, cv_)):
    cv2.circle(img, (a, b), 2, cv2.applyColorMap(np.uint8([[255 * i // len(C)]]), cv2.COLORMAP_JET)[0, 0].tolist(), -1)
cv2.imwrite(sys.argv[2], img); print("saved", sys.argv[2], img.shape)
