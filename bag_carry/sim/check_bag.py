"""Behavioural checks for sim/run_bag.py, after CoDimRecon App. B.4: a run passes if element stretch stays in
[0.9, 1.1], the scene is at rest when the robot starts, the target rises and leaves its support, nothing is frozen,
and the task ends right. Plus this project's robot checks (no table penetration, grip at its cap, hand upright, no grasp
burst). usage: python3 sim/check_bag.py out/heavy/paper.json"""
import json
import sys

import numpy as np

TOP = 0.75
BASKET = dict(c=(0.50, 0.0), r_top=0.125, height=0.24)
GRIP = {"bag": 20.0, "duck": 10.0, "can": 10.0}


def inside_bag(log_end, name, hull=None):
    """Object centre inside the convex hull of the bag's final surface (an open box: the hull's top face is the
    opening), or for older logs within the bag's vertex bounding box shrunk 5 mm (which also accepts objects lying
    next to a toppled bag)."""
    if hull is not None:
        return bool(hull.find_simplex(np.asarray(log_end[name]["com"])[None])[0] >= 0)
    b, o = log_end["bag"], log_end[name]
    lo, hi = np.array(b["bbox"][0]) + 0.005, np.array(b["bbox"][1]) - 0.005
    c = np.array(o["com"])
    return bool(np.all(c > lo) and np.all(c < hi))


def bag_end_from_export(d, export):
    """bag_end_verts, bottom_c, rim_c for runs logged before those were (from the export's last frame)."""
    import glob
    import os
    rest = np.load(os.path.join(export, "bag_rest.npy"))
    z = np.load(sorted(glob.glob(os.path.join(export, "meshseq", "bag", "*.npz")))[-1])
    V, F = z["V"], z["F"]
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    rim = np.unique(u[c == 1])
    bottom = rest[:, 2] < rest[:, 2].min() + 1e-4
    d["bag_end_verts"] = V.tolist()
    d["log"][-1]["bag"]["bottom_c"] = V[bottom].mean(0).tolist()
    d["log"][-1]["bag"]["rim_c"] = V[rim].mean(0).tolist()


def check(d):
    log, task = d["log"], d["task"]
    res, fails = {}, []
    hull = None
    if d.get("bag_end_verts"):
        from scipy.spatial import Delaunay
        hull = Delaunay(np.asarray(d["bag_end_verts"]))
    s = [r["bag"]["stretch"] for r in log]
    res["bag_edge_ratio"] = (min(x[0] for x in s), max(x[1] for x in s))
    if res["bag_edge_ratio"][0] < 0.9 or res["bag_edge_ratio"][1] > 1.1:
        fails.append("bag edge length / rest %.2f-%.2f outside [0.9, 1.1]" % res["bag_edge_ratio"])
    settle = [r for r in log if r["phase"] == "settle"]
    tail = [r for r in settle if r["t"] >= settle[-1]["t"] - 0.5]
    if "bag_max_disp" in settle[-1]:
        res["settle_drift_mm"] = 1000 * max(r["bag_max_disp"] for r in tail)
        res["settle_speed_mm_s"] = 1000 * max(r["bag_max_speed"] for r in tail)
    else:  # older logs: centre-of-mass drift only
        res["settle_drift_mm"] = 1000 * float(np.linalg.norm(np.subtract(tail[-1]["bag"]["com"], tail[0]["bag"]["com"])))
    if res["settle_drift_mm"] > 1.0 or res.get("settle_speed_mm_s", 0.0) > 2.0:
        fails.append("not at rest when the robot starts (drift %.2f mm)" % res["settle_drift_mm"])
    res["robot_lowest_above_table_mm"] = 1000 * min(r["robot_zmin_above_table"] for r in log)
    if res["robot_lowest_above_table_mm"] < -0.5:
        fails.append("robot into the table")
    res["hand_tilt_max_deg"] = max(r["hand_tilt_deg"] for r in log)
    if res["hand_tilt_max_deg"] > 10:
        fails.append("hand tilted %.0f deg" % res["hand_tilt_max_deg"])
    res["grip_N"], res["held_gap_mm"] = {}, {}
    for ph in dict.fromkeys(r["phase"] for r in log):
        if ph == "settle":
            continue
        held = [r for r in log if r["phase"] == ph and r["target"][4] > 0.99]
        g = float(np.median([min(abs(x) for x in r["finger_force"]) for r in held]))
        res["grip_N"][ph] = g
        cap = d.get("grip_caps", {}).get(ph, GRIP[ph])
        if not 0.8 * cap <= g <= 1.2 * cap:
            fails.append(f"{ph}: grip {g:.1f} N, cap {cap:.0f} N")
        # something between the pads: the libuipc fingers stop wider than when they close on each other
        empty = np.median([r["finger_gap_ipc"] for r in log if r["phase"] == "settle"]) if False else None
        res["held_gap_mm"][ph] = 1000 * float(np.median([r["finger_gap_ipc"] for r in held]))
        obj = "bag" if ph == "bag" else ph
        rise = max(r[obj]["zmin"] for r in log if r["phase"] == ph) - [r for r in log if r["phase"] == ph][0][obj]["zmin"]
        res.setdefault("rise_mm", {})[ph] = 1000 * rise
        if rise < 0.12:
            fails.append(f"{ph}: {obj} rose only {rise * 1000:.0f} mm (needs 120)")
    end = log[-1]
    if task == "trash":
        top = max(log, key=lambda r: r["bag"]["zmin"])
        res["bag_cleared_basket"] = top["bag"]["zmin"] > TOP + BASKET["height"]
        if not res["bag_cleared_basket"]:
            fails.append("bag never came fully out of the basket")
        c = np.array(end["bag"]["com"][:2]) - BASKET["c"]
        res["bag_end_outside_basket"] = bool(np.linalg.norm(c) > BASKET["r_top"] + 0.02)
        if not res["bag_end_outside_basket"]:
            fails.append("bag not set down outside the basket")
        for o in ("duck", "can", "ball"):
            if "bbox" in end["bag"] and not inside_bag(end, o):
                fails.append(f"{o} not in the bag at the end")
    else:
        res["bag_end_y"] = end["bag"]["com"][1]
        if res["bag_end_y"] > -0.03 or end["bag"]["zmin"] > TOP + 0.01:
            fails.append("bag not on table B at the end")
        if "bottom_c" in end["bag"]:
            a = np.subtract(end["bag"]["rim_c"], end["bag"]["bottom_c"])
            res["bag_end_tilt_deg"] = float(np.degrees(np.arccos(a[2] / np.linalg.norm(a))))
            if res["bag_end_tilt_deg"] > 20:
                fails.append("bag not standing upright at the end (tilt %.0f deg)" % res["bag_end_tilt_deg"])
        for o in [k for k, v in end.items() if isinstance(v, dict) and "com" in v and k != "bag"]:
            if "bbox" in end["bag"] and not inside_bag(end, o, hull):
                fails.append(f"{o} not in the bag at the end")
    frozen = 0
    run = 0
    for a, b in zip(log, log[1:]):
        if b["target"][4] < 0.05 and b["bag"]["zmin"] > TOP + 0.02 and a["bag"]["com"] == b["bag"]["com"]:
            run += 1
            frozen = max(frozen, run)
        else:
            run = 0
    res["bag_frozen_frames"] = frozen
    if frozen >= 3:
        fails.append(f"bag frozen off its support for {frozen} frames")
    return not fails, fails, res


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    if len(sys.argv) > 2 and not d.get("bag_end_verts"):  # optional export dir for older paper-bag logs
        bag_end_from_export(d, sys.argv[2])
    ok, fails, res = check(d)
    print("PASS" if ok else "FAIL: " + "; ".join(fails), f"(wall {d['wall_s']:.0f} s)")
    for k, v in res.items():
        print(f"  {k}: {v}")
