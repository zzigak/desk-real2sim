"""Which desk objects moved during a robot run besides the ones the demo handles (a hand or arm bumping something).
    python sim/dbg/bystanders.py out/robot/<run>/<demo>_states.npz [--handled a,b,...]"""
import argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("states", nargs="+"); ap.add_argument("--handled", default="")
a = ap.parse_args()
for path in a.states:
    d = np.load(path); t = d["t"]
    print("==", path)
    for k in d.files:
        if not k.startswith("pose__"):
            continue
        T = d[k]; dp = np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1)
        ang = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", T[:, :3, :3], T[0, :3, :3]) - 1) / 2, -1, 1)))
        if dp.max() > 0.002 or ang.max() > 1.0:
            f = int(np.argmax(dp > 0.002)) if (dp > 0.002).any() else int(np.argmax(ang > 1.0))
            print(f"  {k[6:]:16s} moved {1e3 * dp.max():6.1f} mm / {ang.max():5.1f} deg, first at t {t[f]:.2f}"
                  + ("" if k[6:] in a.handled.split(",") else "   <- not handled by the demo"))
