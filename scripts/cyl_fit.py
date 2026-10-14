import sys, numpy as np, cv2
sys.path.insert(0, "scripts"); from pix2table import pix2table, S_METRIC
def bottom_arc(photo, obj, frac=0.7, lift_px=0):
    m = cv2.imread(f"masks/{photo}_{obj}.png", 0) > 127
    ys, xs = np.nonzero(m); x0, x1 = xs.min(), xs.max(); c, w = (x0 + x1) / 2, (x1 - x0)
    pts = []
    for x in range(int(c - frac * w / 2), int(c + frac * w / 2)):
        col = np.nonzero(m[:, x])[0]
        if len(col): pts.append((x + 0.5, col.max() + 1.0 - lift_px))
    return np.array(pts)
def fit_circle(p):
    A = np.c_[2 * p, np.ones(len(p))]; b = (p ** 2).sum(1)
    cx, cy, k = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([cx, cy]), np.sqrt(k + cx * cx + cy * cy)
for photo, obj in [("p09", "tea_tin"), ("p10", "tea_tin"), ("p08", "tea_tin"), ("p09", "bottle"), ("p10", "bottle"), ("p08", "bottle"),
                   ("p08", "jar"), ("p08", "jar_lid")]:
    arc = bottom_arc(photo, obj)
    P = pix2table(photo, arc)[:, :2] * S_METRIC
    c, r = fit_circle(P)
    res = np.abs(np.linalg.norm(P - c, axis=1) - r).max()
    print(f"{photo} {obj:8s} center {np.round(c, 4)}  diameter {2*r:.4f} m  maxres {res*1000:.1f} mm  n={len(P)}")
