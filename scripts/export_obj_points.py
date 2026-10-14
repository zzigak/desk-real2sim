"""Dump per-object VGGT points (metric table frame, pc scale) with colours, for mesh yaw fitting."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, "scripts")
from planes import obj_pts  # noqa  (runs plane fits on import; cheap now)
d = np.load("recon/predictions.npz"); al = json.load(open("work/align.json"))
R, O = np.array(al["R"]), np.array(al["origin"]); P, C, I = d["world_points"], d["depth_conf"], d["images"]
names = [str(n).split(".")[0] for n in d["image_names"]]
out = {}
for obj in ("mouse", "keyboard", "bottle", "tea_tin"):
    pts, cols = [], []
    for ph in ("p09", "p10", "p08", "p02", "p03", "p06"):
        f = f"masks/{ph}_{obj}.png"
        if not os.path.exists(f): continue
        i = names.index(ph)
        m = cv2.resize(cv2.imread(f, 0), P.shape[2:0:-1], interpolation=cv2.INTER_AREA) > 160
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (C[i] > np.percentile(C[i], 20))
        pts.append((P[i][m] - O) @ R.T * 1.37); cols.append(I[i][m])
    out[obj + "_p"] = np.concatenate(pts).astype(np.float32); out[obj + "_c"] = np.concatenate(cols)
    print(obj, len(out[obj + "_p"]))
np.savez_compressed("work/obj_points.npz", **out)
