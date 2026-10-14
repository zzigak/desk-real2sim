"""Cast original-image pixels onto the table plane (z=0 of the aligned table frame). Library + CLI."""
import json, sys
import numpy as np

S_METRIC = 1.37  # m per reconstruction unit (keyboard + QcK pad, see notes)
_d = np.load("recon/predictions.npz")
_al = json.load(open("work/align.json"))
R, O = np.array(_al["R"]), np.array(_al["origin"])
NAMES = [str(n).split(".")[0] for n in _d["image_names"]]
K = {n: np.array(json.load(open("recon/cameras.json"))["frames"][i]["K_original"]) for i, n in enumerate(NAMES)}

def cam_table(photo):
    i = NAMES.index(photo); c2w = _d["c2w"][i]
    Rc = R @ c2w[:3, :3]; t = R @ (c2w[:3, 3] - O)
    return Rc, t

def pix2table(photo, uv, z=0.0):
    """uv: (N,2) original pixels -> (N,3) table-frame points in reconstruction units on plane z (units)."""
    uv = np.atleast_2d(np.asarray(uv, float))
    Rc, t = cam_table(photo)
    rays = (np.linalg.inv(K[photo]) @ np.c_[uv, np.ones(len(uv))].T).T @ Rc.T
    lam = (z - t[2]) / rays[:, 2]
    return t + lam[:, None] * rays

def quad_report(photo, quad, name=""):
    p = pix2table(photo, quad)[:, :2] * S_METRIC
    e = [np.linalg.norm(p[(k + 1) % 4] - p[k]) for k in range(4)]
    print(f"{name}: edges(m) {np.round(e, 4)}  center {np.round(p.mean(0), 4)}  "
          f"angle(deg) {np.degrees(np.arctan2(*(p[1] - p[0])[::-1])):.1f}")
    return p

if __name__ == "__main__":
    s = 1 / 1.8056
    disp = {"A": [(160, 120), (1000, 80), (1230, 675), (225, 755)],
            "B": [(175, 52), (1041, 147), (1080, 765), (60, 635)]}
    for k, q in disp.items():
        quad_report("p09", [(1180 + x * s, 1280 + y * s) for x, y in q], "sheet " + k)
