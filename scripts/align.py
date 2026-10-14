"""Gravity-align the VGGT-Omega reconstruction on the table plane and render top-down views.

Writes work/align.json: R (3x3, recon->table frame, z up), origin (recon coords of table-frame origin),
plus a top-down orthographic RGB render (work/topdown.png) in table-frame units (still non-metric).
"""
import json, sys
import numpy as np, cv2

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "recon/predictions.npz")
imgs, P, C = d["images"], d["world_points"], d["depth_conf"]
S, H, W, _ = imgs.shape

def bamboo_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 10) & (h <= 22) & (s >= 70) & (s <= 170) & (v >= 140)

# table pixels from the wide views (first four frames), confident only
pts = []
for i in range(4):
    m = bamboo_mask(imgs[i]) & (C[i] > np.percentile(C[i], 40))
    pts.append(P[i][m])
pts = np.concatenate(pts)
rng = np.random.default_rng(0)
best = (0, None)
scale_guess = np.median(np.linalg.norm(pts - pts.mean(0), axis=1))
thr = 0.004 * scale_guess / 0.5
for _ in range(3000):
    a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
    n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
    if nn < 1e-9: continue
    n /= nn
    inl = np.abs((pts - a) @ n) < thr
    if inl.sum() > best[0]: best = (inl.sum(), inl)
inl = best[1]
q = pts[inl]; cen = q.mean(0)
_, _, vt = np.linalg.svd(q - cen, full_matrices=False); n = vt[2]
# up = towards the cameras (camera 0 centre is at origin in recon frame)
if np.dot(-cen, n) < 0: n = -n
print(f"table pts {len(pts)}, inliers {inl.sum()} ({inl.mean():.0%}), thr {thr:.4f}")

# rotation: z = n; x = projection of camera-0 x axis (image right) onto plane
x = np.array([1.0, 0, 0]); x = x - n * (x @ n); x /= np.linalg.norm(x)
y = np.cross(n, x)
R = np.stack([x, y, n])            # rows: table-frame axes in recon coords
to_t = lambda p: (p - cen) @ R.T

# all confident points, all frames
allp, allc = [], []
for i in range(S):
    m = C[i] > np.percentile(C[i], 30)
    allp.append(P[i][m]); allc.append(imgs[i][m])
allp = to_t(np.concatenate(allp)); allc = np.concatenate(allc)
tq = to_t(q)
print("table inlier extent x", np.percentile(tq[:, 0], [1, 99]), "y", np.percentile(tq[:, 1], [1, 99]))
json.dump({"R": R.tolist(), "origin": cen.tolist(), "plane_thr": thr}, open("work/align.json", "w"), indent=1)
np.savez_compressed("work/aligned_points.npz", p=allp.astype(np.float32), c=allc)

def ortho(points, colors, xr, yr, res, zmin, zmax, out):
    sel = (points[:, 2] > zmin) & (points[:, 2] < zmax)
    p, c = points[sel], colors[sel]
    order = np.argsort(p[:, 2]); p, c = p[order], c[order]       # higher points painted last
    w = int((xr[1] - xr[0]) / res); h = int((yr[1] - yr[0]) / res)
    img = np.full((h, w, 3), 255, np.uint8)
    u = ((p[:, 0] - xr[0]) / res).astype(int); v = ((yr[1] - p[:, 1]) / res).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img[v[ok], u[ok]] = c[ok]
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img

lo, hi = np.percentile(tq[:, :2], 0.5, 0), np.percentile(tq[:, :2], 99.5, 0)
pad = 0.15 * (hi - lo)
xr, yr = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
res = (xr[1] - xr[0]) / 1400
ortho(allp, allc, xr, yr, res, -0.02 * scale_guess, 1.0 * scale_guess, "work/topdown.png")
json.dump({"xr": xr, "yr": yr, "res": res}, open("work/topdown_meta.json", "w"))
print("topdown", xr, yr, res)
