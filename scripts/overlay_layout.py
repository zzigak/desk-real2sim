import json, numpy as np, cv2
L = json.load(open("layout.json")); meta = json.load(open("work/topdown_meta.json"))
S = 1.37; T = L["table"]; TC = np.array([(-0.405 + 0.51) / 2, (-0.152 + 0.40) / 2]) * S
img = cv2.imread("work/topdown.png")
xr, yr, res = meta["xr"], meta["yr"], meta["res"]
px = lambda p: (int(((p[0] + TC[0]) / S - xr[0]) / res), int((yr[1] - (p[1] + TC[1]) / S) / res))
def rect(c, size, yaw, col):
    a = np.radians(yaw); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    hw = np.array(size[:2]) / 2; cs = [R @ (s * hw) + c for s in ([-1, -1], [1, -1], [1, 1], [-1, 1])]
    cv2.polylines(img, [np.array([px(q) for q in cs])], True, col, 2)
def circ(c, d, col): cv2.circle(img, px(c), int(d / 2 / S / res), col, 2)
rect([0, 0], T["size"], 0, (0, 0, 0))
for k in ("paper_top", "paper_bottom"): rect(L[k]["xy"], L[k]["size"], L[k]["yaw"], (255, 0, 0))
rect(L["mousepad"]["xy"], L["mousepad"]["size"], L["mousepad"]["yaw"], (255, 0, 255))
rect(L["keyboard"]["xy"], L["keyboard"]["size"], L["keyboard"]["yaw"], (0, 0, 255))
rect(L["mouse"]["xy"], L["mouse"]["size"], L["mouse"]["yaw"], (0, 128, 255))
m = L["marker"]; rect(m["xy"], [m["length"], 2 * m["radius"]], m["yaw"], (0, 0, 0))
c = L["cloth"]; a = np.radians(c["yaw"]); fwd = np.array([np.sin(a), -np.cos(a)])
rect(np.array(c["back_mid_xy"]) + fwd * c["on_table_depth"] / 2, [c["width"], c["on_table_depth"]], c["yaw"], (0, 160, 0))
for k in ("tea_tin", "bottle", "jar", "jar_lid"): circ(L[k]["xy"], L[k]["diameter"], (0, 90, 200))
lp = L["laptop"]; foot = lp["size"][1] * np.cos(np.radians(lp["pitch"]))
rect(lp["base_center"][:2], [lp["size"][0], foot], lp["yaw"], (200, 0, 200))
mo = L["monitor"]; rect(mo["panel_center"][:2], [mo["panel"][0], mo["panel"][2]], mo["yaw"], (0, 0, 0))
cv2.imwrite("work/layout_overlay.png", img)
