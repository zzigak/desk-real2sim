"""Collision (and simple visual) meshes for the articulated / multi-part desk objects. Writes assets/*.obj + specs.json.
All dimensions in metres; each part is built in its own local frame with z=0 at its base."""
import json, os
import numpy as np, trimesh
from geom import revolve, check

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)

# ---------------- Ball wide-mouth quart jar + plastic storage cap (real helical thread) ----------------
# Fits are calibrated with sim/pull_test.py. Under IPC every mating gap below contact_d_hat (1 mm) is pre-loaded by the
# barrier, so a fit's holding force is set by how far below 1 mm its gap is and over how much area.
JAR = dict(r_body=0.049, r_root=0.0415, h=0.0030, p=0.0055, clear=0.0012, t_z0=0.1485, t_z1=0.1655, ramp=0.001,
           lip_top=0.170, r_mouth=0.0375)
JAR_LID = dict(r_out=0.0475, height=0.020, top=0.003, gap_above_lip=0.0015)


def make_jar(**over):
    """jar + lid. The lid bore is the jar thread grown by a uniform `clear` (geom.thread_envelope), so no flank sits
    inside the contact layer and the lid spins on its thread until its ceiling meets the lip. The neck has room for one
    turn of engagement at this pitch (like a real canning jar): a lid set higher lands on the run-out and jams."""
    J = {**JAR, **{k: v for k, v in over.items() if k in JAR}}; L = {**JAR_LID, **{k: v for k, v in over.items() if k in JAR_LID}}
    rr = J["r_root"]; rm = J["r_mouth"] if "r_mouth" not in over else over["r_mouth"]
    if "r_root" in over and "r_mouth" not in over:
        rm = rr - 0.004                                  # the lip and mouth follow a resized neck
    jar_prof = [(0, 0), (0.046, 0), (0.049, 0.004), (0.049, 0.128), (0.0465, 0.138), (0.0425, 0.144), (rr, 0.147),
                (rr, 0.1665), (rr - 0.001, 0.1690), (rr - 0.0025, J["lip_top"]), (rm, 0.1685),
                (rm, 0.146), (0.0425, 0.137), (0.0455, 0.128), (0.0455, 0.010), (0.043, 0.006), (0, 0.006)]
    jar = revolve(jar_prof, n=192, subdiv={6: 0.00025},
                  threads=[dict(seg=(6, 6), h=J["h"], p=J["p"], z0=J["t_z0"], z1=J["t_z1"], ramp=J["ramp"], rest=0.0)])
    # lid seated: ceiling 1.5 mm above the lip; bore = jar thread + clearance, threaded in jar coordinates via phase
    z_off = J["lip_top"] + L["gap_above_lip"] - (L["height"] - L["top"])      # jar z of the lid's skirt bottom when seated
    rb = J["r_root"]
    assert rb + J["h"] + J["clear"] < L["r_out"] - 0.001, "lid wall too thin for the thread"
    lid_prof = [(L["r_out"], 0), (L["r_out"], L["height"]), (0, L["height"]), (0, L["height"] - L["top"]),
                (rb, L["height"] - L["top"]), (rb, 0)]
    lid = revolve(lid_prof, n=192, subdiv={4: 0.00025},
                  threads=[dict(seg=(4, 5), h=J["h"], p=J["p"], clear=J["clear"], phase=z_off / J["p"], z0=0.0,
                                z1=L["height"] - L["top"], ramp=J["ramp"], rest=1.0)])
    return dict(jar=jar, jar_lid=lid), dict(**J, lid=L, lid_seated_z=z_off, mass=0.38, lid_mass=0.020)


# ---------------- tea tin: body with a stepped neck + slip-fit lid ----------------
TIN = dict(r=0.044, shoulder=0.098, r_neck=0.0410, neck_top=0.121, r_in=0.0390, bottom=0.003,
           lid_h=0.031, lid_top=0.002, lid_gap=0.00090, disc_r=0.031, disc_h=0.0025, seat_gap=0.0012,
           band_gap=None, band=0.002, band_ramp=0.0005)


def make_tin(**over):
    """slip-fit lid: a 2 cm skirt over the neck with radial gap lid_gap. With band_gap set, a `band`-tall rib at the
    skirt mouth closes the gap to band_gap and holds the lid (used at small d_hat, where lid_gap is a real clearance)."""
    T = {**TIN, **over}
    tin = revolve([(0, 0), (T["r"], 0), (T["r"], T["shoulder"]), (T["r_neck"], T["shoulder"]), (T["r_neck"], T["neck_top"]),
                   (T["r_in"], T["neck_top"]), (T["r_in"], T["bottom"]), (0, T["bottom"])], n=128)
    rli = T["r_neck"] + T["lid_gap"]                                          # skirt bore
    rib = [] if T["band_gap"] is None else [(rli, T["band"] + T["band_ramp"]), (T["r_neck"] + T["band_gap"], T["band"])]
    r0 = rli if T["band_gap"] is None else T["r_neck"] + T["band_gap"]
    tin_lid = revolve([(r0, 0), (T["r"], 0), (T["r"], T["lid_h"]), (T["disc_r"] + 0.002, T["lid_h"]),
                       (T["disc_r"], T["lid_h"] + T["disc_h"]), (0, T["lid_h"] + T["disc_h"]),
                       (0, T["lid_h"] - T["lid_top"]), (rli, T["lid_h"] - T["lid_top"])] + rib, n=128)
    return dict(tin=tin, tin_lid=tin_lid), dict(**T, lid_seated_z=T["shoulder"] + T["seat_gap"], mass=0.12, lid_mass=0.035)


# ---------------- whiteboard marker: barrel + snap cap ----------------
MARKER = dict(r=0.0085, barrel=0.086, r_tip=0.0060, tip_end=0.104, r_nib=0.0028, nib_end=0.114,
              cap_len=0.055, cap_r=0.0095, cap_depth=0.040, gap=0.0017, seat_gap=0.00070, band=0.003, band_ramp=0.0005, min_alone=0.0011)


def make_marker(**over):
    """the cap is held only where a ring on the barrel meets a rib inside the cap mouth (as on a real marker's click
    ring): there the gap is seat_gap, while ring-to-bore and rib-to-barrel stay above d_hat, so once the rib has slid
    past the ring the cap is free. The cap mouth sits 2.5 mm past the barrel shoulder when seated."""
    M = {**MARKER, **over}
    d = (M["gap"] - M["seat_gap"]) / 2; bore = M["r_tip"] + M["gap"]; z0 = M["barrel"] + 0.0025; b, rp = M["band"], M["band_ramp"]
    assert M["gap"] - d > M["min_alone"], "ring or rib alone would sit inside the contact layer"
    marker = revolve([(0, 0), (M["r"] - 0.001, 0), (M["r"], 0.0015), (M["r"], M["barrel"]), (M["r_tip"], M["barrel"] + 0.002),
                      (M["r_tip"] + d, z0), (M["r_tip"] + d, z0 + b - rp), (M["r_tip"], z0 + b),
                      (M["r_tip"], M["tip_end"]), (M["r_nib"], M["tip_end"] + 0.002), (M["r_nib"], M["nib_end"] - 0.001),
                      (0, M["nib_end"])], n=48)
    cap = revolve([(bore - d, 0), (M["cap_r"], 0), (M["cap_r"], M["cap_len"] - 0.002), (M["cap_r"] - 0.002, M["cap_len"]),
                   (0, M["cap_len"]), (0, M["cap_depth"]), (bore, M["cap_depth"]), (bore, b), (bore - d, b - rp)], n=48)
    return dict(marker=marker, marker_cap=cap), dict(**M, cap_bore=bore, ring=d, cap_seated_z=z0, mass=0.012, cap_mass=0.004)


# Fits for the robot (YAM) runs, which use the yam-arcade jars-station contact settings (d_hat 0.2 mm, see
# robot_desk.py): every clearance is real (> d_hat) except the retention bands, calibrated with pull_test.py --contact robot.
ROBOT_FITS = dict(jar=dict(r_root=0.040, h=0.0022, p=0.0045, clear=0.0004, r_out=0.0439),   # 87.8 mm lid: fits the gripper
                  tea_tin=dict(lid_gap=0.0005, band_gap=0.00020),
                  marker=dict(gap=0.0008, seat_gap=0.000205, min_alone=0.0003))
# The marker band is calibrated in the full robot scene (20 N grips, paper present), not the two-body rig: libuipc's
# preloaded contacts hold harder when other bodies are present. In the desk: 0.200 mm ~25 N, 0.205 mm 4.3 N,
# 0.210 mm 0.9 N (the rig gives 4.3 N at 0.17 mm and 0.4 N at 0.18 mm).


def min_gap(a, b, offset):
    """smallest distance between seated mating parts (b raised by offset), over their overlapping z band."""
    from scipy.spatial import cKDTree
    bb = b.copy(); bb.apply_translation([0, 0, offset])
    lo, hi = max(a.bounds[0, 2], bb.bounds[0, 2]) - 1e-3, min(a.bounds[1, 2], bb.bounds[1, 2]) + 1e-3
    worst = 1.0
    for src, dst in ((bb, a), (a, bb)):
        v = src.vertices[(src.vertices[:, 2] > lo) & (src.vertices[:, 2] < hi)]
        tree = cKDTree(dst.sample(600000))
        for i in range(0, len(v), 2000):
            worst = min(worst, tree.query(v[i:i + 2000])[0].min())
    return worst


if __name__ == "__main__":
    meshes, spec = {}, {}
    for key, fn, seat in (("jar", make_jar, "lid_seated_z"), ("tea_tin", make_tin, "lid_seated_z"), ("marker", make_marker, "cap_seated_z")):
        m, sp = fn(); meshes.update(m); spec[key] = sp
        a, b = list(m.values())
        print(f"seated {key:8s}: min surface distance {min_gap(a, b, sp[seat]) * 1000:.2f} mm")

    # ---------------- insulated bottle (open, no lid) + wooden stick ----------------
    B = dict(r=0.0385, body_top=0.125, r_neck=0.0345, top=0.155, r_in=0.0300, bottom=0.012)
    bottle = revolve([(0, 0), (B["r"] - 0.003, 0), (B["r"], 0.003), (B["r"], B["body_top"]), (B["r_neck"], B["body_top"] + 0.007),
                      (B["r_neck"], B["top"]), (B["r_in"], B["top"]), (B["r_in"], B["bottom"]), (0, B["bottom"])], n=128)
    S = dict(r=0.002, length=0.20)
    stick = trimesh.creation.cylinder(radius=S["r"], height=S["length"], sections=12)
    spec["bottle"] = dict(**B, mass=0.33); spec["stick"] = dict(**S, mass=0.002)
    meshes.update(bottle=bottle, stick=stick)
    for name, m in meshes.items():
        check(m, name)
        m.export(os.path.join(OUT, f"{name}_col.obj"))
    json.dump(spec, open(os.path.join(OUT, "specs.json"), "w"), indent=1)
