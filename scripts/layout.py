"""Assemble the measured desk layout into one metric world frame and write layout.json.

World frame: z up, floor z=0, table top at z=TABLE_H, x to the right along the desk, y toward the wall,
(0,0) = centre of the table top. Reconstruction units -> metres with S=1.37 (point-cloud scale, set by
the Filco TKL keyboard and QcK pad). Plane-casting measures ~4% large (focal/depth inconsistency in
VGGT), so lengths measured by casting are divided by CAST_BIAS; positions are not (pad centres from both
methods agree to 5 mm). Object SIZES come from catalogue dimensions where the product is identifiable.
"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from pix2table import pix2table, S_METRIC as S
from cast_plane import cast

CAST_BIAS = 1.04
TABLE = dict(x0=-0.405, x1=0.51, y0=-0.152, y1=0.40)          # units, from the table-height ortho slice
TC = np.array([(TABLE["x0"] + TABLE["x1"]) / 2, (TABLE["y0"] + TABLE["y1"]) / 2]) * S
TABLE_H = 0.775                                                # floor histogram peak, planes.py
W = lambda p: (np.asarray(p, float)[..., :2] - TC)             # pc-metric xy -> world xy
yaw_of = lambda a, b: float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
L = {}

L["table"] = dict(size=[(TABLE["x1"] - TABLE["x0"]) * S, (TABLE["y1"] - TABLE["y0"]) * S, 0.025], top_z=TABLE_H)

# --- flat things on the table, cast in p09 (pixel picks in work/z_*.jpg)
def cast_quad(photo, quad, z=0.0):
    return pix2table(photo, quad, z / S)[:, :2] * S
def quad_pose(P):
    c = P.mean(0); e01 = P[1] - P[0]; e32 = P[2] - P[3]
    return W(c), (yaw_of(P[0], P[1]) + yaw_of(P[3], P[2])) / 2
s = 1 / 1.8056
for k, q in {"paper_top": [(160, 120), (1000, 80), (1230, 675), (225, 755)],
             "paper_bottom": [(175, 52), (1041, 147), (1080, 765), (60, 635)]}.items():
    c, yaw = quad_pose(cast_quad("p09", [(1180 + x * s, 1280 + y * s) for x, y in q]))
    L[k] = dict(xy=c.tolist(), yaw=yaw, size=[0.2794, 0.2159], thickness=1e-4, note="US Letter, landscape")
s = 1 / 1.9118
c, yaw = quad_pose(cast_quad("p09", [(1820 + x * s, 1100 + y * s) for x, y in [(35, 46), (784, 46), (1264, 815), (285, 815)]]))
L["mousepad"] = dict(xy=c.tolist(), yaw=yaw, size=[0.27, 0.32], thickness=0.003, note="SteelSeries QcK medium, portrait")
# cloth: on-table part (back-left, back-right, front-right@edge, front-left@edge) + drape
s = 1 / 1.435
P = cast_quad("p09", [(680 + x * s, 1270 + y * s) for x, y in [(205, 30), (668, 48), (640, 660), (40, 645)]])
on_table_depth = (np.linalg.norm(P[3] - P[0]) + np.linalg.norm(P[2] - P[1])) / 2 / CAST_BIAS
width = (np.linalg.norm(P[1] - P[0]) + np.linalg.norm(P[2] - P[3])) / 2 / CAST_BIAS
back_mid = W((P[0] + P[1]) / 2)
L["cloth"] = dict(back_mid_xy=back_mid.tolist(), yaw=yaw_of(P[0], P[1]), width=width, on_table_depth=on_table_depth,
                  note="microfibre, folded; drapes over the front edge")
# marker: cast cap end / cap-barrel junction / barrel end at the axis height
s = 1 / 4.194
r_m = 0.0085
M = cast_quad("p09", [(1390 + x * s, 1420 + y * s) for x, y in [(95, 150), (518, 200), (1195, 300)]], z=0.0001 + r_m)
L["marker"] = dict(xy=W((M[0] + M[2]) / 2).tolist(), yaw=yaw_of(M[2], M[0]),  # +x of the marker = toward the cap
                   length=float(np.linalg.norm(M[2] - M[0]) / CAST_BIAS), cap_length=float(np.linalg.norm(M[1] - M[0]) / CAST_BIAS),
                   radius=r_m, rests_on="paper_top")

# --- cylinders from the cast bottom arcs (cyl_fit.py)
L["tea_tin"] = dict(xy=W(np.array([0.402, 0.328])).tolist(), diameter=0.087)
L["bottle"] = dict(xy=W(np.array([0.6395, 0.348])).tolist(), diameter=0.075)
L["jar_lid"] = dict(xy=W(np.array([0.4878, -0.1486])).tolist(), diameter=0.088, upside_down=True,
                    note="Ball wide-mouth storage cap lying open side up")
# jar: glass reflections spoil the arc fit; point-cloud front surface centre pushed back by 2r/pi toward the wall
lift = json.load(open("work/lift.json"))
jc = np.array(lift["jar"]["center"]) * S + np.array([0, 2 * 0.049 / np.pi])
L["jar"] = dict(xy=W(jc).tolist(), diameter=0.098, height=0.172, note="Ball wide-mouth quart")

# --- keyboard / mouse from point-cloud PCA
d = np.load("work/aligned_points.npz")
def pca_pose(obj):
    pts = []
    import os
    from planes import obj_pts
    return obj_pts(obj)
from planes import obj_pts
for obj in ("keyboard", "mouse"):
    p = obj_pts(obj); xy = p[:, :2]; c = np.median(xy, 0)
    ev, evec = np.linalg.eigh(np.cov((xy - c).T)); ax = evec[:, 1]
    yaw = float(np.degrees(np.arctan2(ax[1], ax[0])))
    yaw = (yaw + 90) % 180 - 90
    L[obj] = dict(xy=W(c).tolist(), yaw=yaw, z_top=float(np.percentile(p[:, 2], 98)))
L["keyboard"].update(size=[0.356, 0.138, 0.038], note="Filco Majestouch-2 TKL")
L["mouse"].update(size=[0.108, 0.070, 0.071], note="Logitech Lift vertical mouse; on the pad")

# --- laptop on stand (two-plane fit, planes.py)
lp = np.load("work/laptop_planes.npz")
n1, c1, n2, c2 = lp["n1"], lp["c1"], lp["n2"], lp["c2"]
if n1[2] < 0: n1 = -n1
if n2 @ np.r_[n1[:2], 0] < 0: n2 = -n2                 # screen normal faces the user side
hinge = np.cross(n1, n2); hinge /= np.linalg.norm(hinge)
if hinge[0] < 0: hinge = -hinge
fwd = np.cross(hinge, n1); fwd /= np.linalg.norm(fwd)  # down-slope direction on the deck (toward the user)
if fwd[2] > 0: fwd = -fwd
p1 = lp["p1"]; u = (p1 - c1) @ hinge; v = (p1 - c1) @ fwd
# deck extent along the slope: use robust percentiles (keyboard deck + palm rest)
v_lo, v_hi = np.percentile(v, [1, 99]); u_lo, u_hi = np.percentile(u, [1, 99])
base_c = c1 + hinge * (u_lo + u_hi) / 2 + fwd * (v_lo + v_hi) / 2
pitch = float(np.degrees(np.arcsin(-fwd[2])))
yaw_l = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 90   # 0 = facing -y (the user), CCW positive
open_angle = 180 - float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1))))
L["laptop"] = dict(base_center=[*W(base_c).tolist(), float(base_c[2])], yaw=yaw_l, pitch=pitch, lid_open_deg=open_angle,
                   measured_deck=[float(u_hi - u_lo), float(v_hi - v_lo)], size=[0.3557, 0.2481, 0.0168],
                   note="MacBook Pro 16 (2021+), German QWERTZ")
L["laptop_stand"] = dict(note="aluminium folding stand; holds the laptop at the measured pitch")

# --- monitor (panel plane + corner cast, cast_plane.py)
mp = np.load("work/monitor_plane.npz")
nm = mp["n"]; base = mp["base"]
L["monitor"] = dict(panel_center=[*W(np.array([0.217, 0.412])).tolist(), 0.361 / CAST_BIAS],
                    yaw=float(np.degrees(np.arctan2(nm[0], -nm[1]))), tilt_deg=float(np.degrees(np.arcsin(nm[2]))),
                    base_xy=W(np.median(base[:, :2], 0)).tolist(), screen_active=[0.698, 0.392], panel=[0.750, 0.459, 0.060],
                    tilt_range_deg=[-5, 21], note="Dell UltraSharp 32 (UP3214Q-like)")
json.dump(L, open("layout.json", "w"), indent=1)
for k, v in L.items():
    print(f"{k:13s}", {a: (np.round(b, 3).tolist() if isinstance(b, (list, float)) else b) for a, b in v.items() if a != "note"})
