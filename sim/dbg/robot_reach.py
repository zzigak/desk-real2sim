"""Build the desk with the YAM cart and check IK reach for every grasp target of the fit demos (no stepping)."""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import robot_desk as R, scene as S
d = R.Desk(keep=(), cart_x=0.0)
H = S.H
for n in ("jar", "jar_lid", "tin", "tin_lid", "marker", "marker_cap"):
    T = d.pose(n); o = S.SC["objects"][n]
    print(n, "pose vs scene.json:", np.round(T[:3, 3] - o["pos"], 6))
targets = {
    "tin lid top-down (right)": ("right", (0.330, 0.148, H + 0.130)),
    "tin body side (left)": ("left", (0.330, 0.148 - 0.05, H + 0.06)),
    "jar lid (right)": ("right", (0.416, -0.318, H + 0.012)),
    "jar body side (left)": ("left", (0.531, -0.296 - 0.05, H + 0.06)),
    "jar mouth (right)": ("right", (0.531, -0.296, H + 0.20)),
    "marker cap (right)": ("right", (0.097, -0.218, H + 0.012)),
    "marker barrel (left)": ("left", (0.17, -0.233, H + 0.012)),
}
for bx in (0.0, 0.15, 0.3, 0.45):
    for k, (side, p) in targets.items():
        res = []
        for tilt in (0.0, 0.3, 0.6, 0.9, math.pi / 2):
            try:
                d.ik(side, p, tilt=tilt, base_x=bx); res.append(f"{tilt:.1f}")
            except RuntimeError:
                pass
        print(f"base_x {bx:+.2f} (cart x {bx:+.2f})  {k:26s} ok tilts: {res}")
