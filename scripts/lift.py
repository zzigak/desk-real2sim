"""Lift SAM2 masks into the gravity-aligned table frame using VGGT-Omega world points.
Prints per-object footprint (min-area rect), z range, centroid, in reconstruction units."""
import json, glob, os
import numpy as np, cv2

d = np.load("recon/predictions.npz")
al = json.load(open("work/align.json")); R = np.array(al["R"]); O = np.array(al["origin"])
P, C = d["world_points"], d["depth_conf"]
names = [os.path.basename(str(n)).split(".")[0] for n in d["image_names"]]
H, W = P.shape[1:3]
objs = {}
for f in sorted(glob.glob("masks/*.png")):
    photo, obj = os.path.basename(f)[:-4].split("_", 1)
    if photo not in names: continue
    i = names.index(photo)
    m = cv2.resize(cv2.imread(f, 0), (W, H), interpolation=cv2.INTER_AREA) > 160
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    m &= C[i] > np.percentile(C[i], 20)
    pts = (P[i][m] - O) @ R.T
    objs.setdefault(obj, []).append(pts)
out = {}
for obj, L in sorted(objs.items()):
    p = np.concatenate(L)
    z1, z50, z99 = np.percentile(p[:, 2], [2, 50, 98])
    # footprint from points in the upper part only where it matters? use all points
    xy = p[:, :2].astype(np.float32)
    lo, hi = np.percentile(xy, 1, 0), np.percentile(xy, 99, 0)
    keep = np.all((xy >= lo) & (xy <= hi), 1)
    (cx, cy), (w, h), ang = cv2.minAreaRect(xy[keep] * 1000)
    out[obj] = dict(n=len(p), center=[cx / 1000, cy / 1000], size=[w / 1000, h / 1000], angle=ang,
                    z=[float(z1), float(z50), float(z99)])
    print(f"{obj:9s} n={len(p):6d} rect {w/1000:.3f} x {h/1000:.3f} @ ({cx/1000:+.3f},{cy/1000:+.3f}) ang {ang:6.1f}  z2/50/98 {z1:+.3f} {z50:+.3f} {z99:+.3f}")
json.dump(out, open("work/lift.json", "w"), indent=1)
