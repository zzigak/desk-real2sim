"""Plane fits on lifted object points: laptop base / lid, monitor panel. Also floor height and cylinder heights."""
import json, glob, os, sys
import numpy as np, cv2
sys.path.insert(0, "scripts"); from pix2table import pix2table, cam_table, K, S_METRIC
S = S_METRIC
d = np.load("recon/predictions.npz"); al = json.load(open("work/align.json"))
R, O = np.array(al["R"]), np.array(al["origin"])
P, C = d["world_points"], d["depth_conf"]; names = [str(n).split(".")[0] for n in d["image_names"]]
def obj_pts(obj, photos=("p09", "p10", "p08"), q=20):
    L = []
    for ph in photos:
        f = f"masks/{ph}_{obj}.png"
        if not os.path.exists(f): continue
        i = names.index(ph)
        m = cv2.resize(cv2.imread(f, 0), P.shape[2:0:-1], interpolation=cv2.INTER_AREA) > 160
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (C[i] > np.percentile(C[i], q))
        L.append((P[i][m] - O) @ R.T)
    return np.concatenate(L) * S
def ransac_plane(p, thr=0.004, it=2000, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(it):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-12: continue
        n /= nn; inl = np.abs((p - a) @ n) < thr
        if best is None or inl.sum() > best.sum(): best = inl
    q = p[best]; c = q.mean(0); n = np.linalg.svd(q - c, full_matrices=False)[2][2]
    return n, c, best
# --- floor: all points far below the table
a = np.load("work/aligned_points.npz"); allp = a["p"] * S
low = allp[allp[:, 2] < -0.3]
h, e = np.histogram(low[:, 2], bins=120)
print("floor candidates (z m, count):", [(round(float(e[i]), 3), int(h[i])) for i in np.argsort(h)[-5:]])
# --- laptop: two planes
lp = obj_pts("laptop")
n1, c1, in1 = ransac_plane(lp)
rest = lp[~in1]; n2, c2, in2 = ransac_plane(rest)
for nm, n, c, k in [("plane1", n1, c1, in1.sum()), ("plane2", n2, c2, in2.sum())]:
    if n[2] < 0: n = -n
    print(f"laptop {nm}: n={np.round(n,3)} tilt from horizontal {np.degrees(np.arccos(abs(n[2]))):.1f} deg, centroid {np.round(c,3)}, inliers {k}")
np.savez("work/laptop_planes.npz", n1=n1, c1=c1, n2=n2, c2=c2, p1=lp[in1], p2=rest[in2])
# --- monitor panel
mp = obj_pts("monitor")
nm_, cm_, inm = ransac_plane(mp[mp[:, 2] > 0.2])
if nm_[1] > 0: nm_ = -nm_   # normal toward the user (-y)
q = mp[mp[:, 2] > 0.2][inm]
print(f"monitor panel: n={np.round(nm_,3)} tilt(back) {np.degrees(np.arcsin(-nm_[2] if False else nm_[2])):.1f} deg  yaw {np.degrees(np.arctan2(nm_[0], -nm_[1])):.1f}  centroid {np.round(cm_,3)}")
u = np.cross([0, 0, 1], nm_); u /= np.linalg.norm(u); v = np.cross(nm_, u)
qq = (q - cm_) @ np.c_[u, v]
print("  panel extent along width", np.round(np.percentile(qq[:, 0], [0.5, 99.5]), 3), "height", np.round(np.percentile(qq[:, 1], [0.5, 99.5]), 3),
      "z range", np.round(np.percentile(q[:, 2], [0.5, 99.5]), 3))
base = mp[mp[:, 2] < 0.03]
print("  monitor low points (base) xy pct", np.round(np.percentile(base[:, :2], [2, 50, 98], axis=0), 3), "n", len(base))
np.savez("work/monitor_plane.npz", n=nm_, c=cm_, q=q, base=base)
