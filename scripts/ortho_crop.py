import sys, json, numpy as np, cv2
a = np.load("work/aligned_points.npz"); p, c = a["p"], a["c"]
x0, x1, y0, y1, zmin, zmax, res = map(float, sys.argv[1:8]); out = sys.argv[8]
sel = (p[:, 2] > zmin) & (p[:, 2] < zmax) & (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
q, col = p[sel], c[sel]; o = np.argsort(q[:, 2]); q, col = q[o], col[o]
w, h = int((x1 - x0) / res), int((y1 - y0) / res)
img = np.full((h, w, 3), 255, np.uint8)
u = ((q[:, 0] - x0) / res).astype(int); v = ((y1 - q[:, 1]) / res).astype(int)
ok = (u >= 0) & (u < w) & (v >= 0) & (v < h); img[v[ok], u[ok]] = col[ok]
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
step = 0.01
for gx in np.arange(np.ceil(x0 / step) * step, x1, step):
    X = int((gx - x0) / res); big = abs(gx / 0.05 - round(gx / 0.05)) < 1e-6
    cv2.line(img, (X, 0), (X, h), (0, 0, 255) if big else (0, 200, 255), 1)
    if big: cv2.putText(img, f"{gx:.2f}", (X + 2, 12), 0, 0.4, (0, 0, 255), 1)
for gy in np.arange(np.ceil(y0 / step) * step, y1, step):
    Y = int((y1 - gy) / res); big = abs(gy / 0.05 - round(gy / 0.05)) < 1e-6
    cv2.line(img, (0, Y), (w, Y), (0, 0, 255) if big else (0, 200, 255), 1)
    if big: cv2.putText(img, f"{gy:.2f}", (2, Y - 2), 0, 0.4, (0, 0, 255), 1)
cv2.imwrite(out, img)
