#!/usr/bin/env python3
"""Side-by-side: photo | model A | model B at chosen cameras of a dataset (gs env).
usage: compare_models_views.py <data> <names.json> <ckptA> <ckptB> <out.jpg> [--n 6] [--labels A B]"""
import argparse, json, numpy as np, torch, cv2
from PIL import Image
from pycolmap import SceneManager
from gsplat.rendering import rasterization
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('names'); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('out')
ap.add_argument('--n', type=int, default=6); ap.add_argument('--labels', nargs=2, default=['A', 'B'])
a = ap.parse_args()
names = sorted(json.load(open(a.names))); pick = [names[i] for i in np.linspace(0, len(names) - 1, a.n).astype(int)]
m = SceneManager(a.data + '/sparse/0/'); m.load_cameras(); m.load_images(); by = {m.images[i].name: m.images[i] for i in m.images}
def load(p):
    sp = torch.load(p, map_location='cuda', weights_only=False)['splats']
    return (sp['means'], sp['quats'], torch.exp(sp['scales']), torch.sigmoid(sp['opacities']), torch.cat([sp['sh0'], sp['shN']], 1))
def fair(r, g):   # per-channel gain/bias fit, then PSNR
    A = torch.stack([r, torch.ones_like(r)], -1); rf = torch.empty_like(r)
    for ch in range(3):
        k = torch.linalg.lstsq(A[:, ch], g[:, ch:ch + 1]).solution; rf[:, ch] = (A[:, ch] @ k)[:, 0]
    return float(-10 * torch.log10(((rf - g) ** 2).mean()))
rows = {n: [] for n in pick}; scores = {n: [] for n in pick}
for path in (a.a, a.b):
    G = load(path)
    for n in pick:
        im = by[n]; c = m.cameras[im.camera_id]; T = np.eye(4); T[:3, :3] = im.R(); T[:3, 3] = im.tvec
        K = torch.tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], dtype=torch.float32, device='cuda')[None]
        with torch.no_grad():
            r = rasterization(*G, torch.tensor(T, dtype=torch.float32, device='cuda')[None], K, c.width, c.height, sh_degree=3, near_plane=0.15)[0][0].clamp(0, 1)
        g = torch.from_numpy(np.asarray(Image.open(f'{a.data}/images/{n}').convert('RGB'), np.float32) / 255).cuda()
        scores[n].append(fair(r.reshape(-1, 3), g.reshape(-1, 3)))
        if not rows[n]: rows[n].append((g.cpu().numpy() * 255).astype(np.uint8))
        rows[n].append((r.cpu().numpy() * 255).astype(np.uint8))
    del G; torch.cuda.empty_cache()
out = []
for n in pick:
    tiles = [cv2.resize(cv2.cvtColor(t, cv2.COLOR_RGB2BGR), (640, 360)) for t in rows[n]]
    for t, txt in zip(tiles, [f'photo {n}', f'{a.labels[0]}  {scores[n][0]:.1f} dB', f'{a.labels[1]}  {scores[n][1]:.1f} dB']):
        cv2.putText(t, txt, (8, 26), 0, 0.75, (0, 0, 0), 4); cv2.putText(t, txt, (8, 26), 0, 0.75, (0, 255, 255), 2)
    out.append(np.hstack(tiles))
cv2.imwrite(a.out, np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 90])
print({n: [round(s, 2) for s in scores[n]] for n in pick})
