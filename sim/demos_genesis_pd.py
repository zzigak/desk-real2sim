"""Exercise each articulation / mechanism of the desk scene with PD-tracked 'virtual hands'.

A driven part is coupled 'two_way_soft_constraint': Genesis integrates it under PD targets on chosen dofs (the
hand), libuipc resolves every contact it makes; dofs with zero gain are left to physics (e.g. the jar lid's height is
free while it is twisted, so only the thread can raise or lower it). Joints (laptop hinge, monitor tilt) are driven by
a PD motor on the joint itself. Logs per-step metrics to <out>/<demo>.json and renders <out>/<demo>_<cam>.mp4."""
import argparse, json, os, time
import numpy as np
import genesis as gs
import scene as S

A = lambda t: np.array(t.cpu() if hasattr(t, "cpu") else t, dtype=float)

def ramp(t, t0, t1):
    return float(np.clip((t - t0) / (t1 - t0), 0, 1))

def smooth(t, t0, t1):
    s = ramp(t, t0, t1); return s * s * (3 - 2 * s)

class Hand:
    """PD 'hand' on a free body's 6 dofs. gains per dof; a dof with kp=0 is free."""
    def __init__(self, ent):
        self.e = ent; self.q0 = A(ent.get_dofs_position())
    def gains(self, kp, kv):
        self.e.set_dofs_kp(np.asarray(kp, float)); self.e.set_dofs_kv(np.asarray(kv, float))
    def go(self, q, dofs=None):
        dofs = list(range(6)) if dofs is None else dofs
        self.e.control_dofs_position(np.asarray(q, float)[dofs] if len(q) == 6 else np.asarray(q, float), dofs_idx_local=dofs)
    def q(self): return A(self.e.get_dofs_position())

# ------------------------------------------------------------------------------------------------ demos
def demo_laptop(t, st, E):
    """close the lid (0.5-2.5 s), hold, reopen to the photographed angle (3.5-5.5 s), driving the IPC-native hinge."""
    h = E["_hinges"]["laptop"]; q0 = S.SC["joints"]["laptop"]["q0"]
    closed = -q0 - 0.02                                               # a little past closed: the lid presses on the deck
    h.target = closed * (smooth(t, 0.5, 2.5) - smooth(t, 3.5, 5.5))
    b, bq = S.link_pos(E["laptop_base"]); d = 1000 * (b - st.setdefault("b0", b))
    return dict(hinge_target_deg=np.degrees(h.target + q0), lid_open_deg=np.degrees(h.angle + q0),
                base_dx_mm=d[0], base_dy_mm=d[1], base_dz_mm=d[2], stand_drift_mm=1000 * np.linalg.norm(
                    S.link_pos(E["laptop_stand"])[0] - st.setdefault("s0", S.link_pos(E["laptop_stand"])[0])))

def demo_monitor(t, st, E):
    """tilt back to +21 deg, forward to -5 deg, back to the photographed 1.4 deg (stand limits -5..21)."""
    h = E["_hinges"]["monitor"]; q0 = S.SC["joints"]["monitor"]["q0"]
    keys = [(0.0, 0.0), (0.5, 0.0), (2.0, np.radians(21) - q0), (2.5, np.radians(21) - q0), (4.5, np.radians(-5) - q0),
            (5.0, np.radians(-5) - q0), (6.0, 0.0)]
    ts, qs = zip(*keys); h.target = float(np.interp(t, ts, qs))
    return dict(tilt_target_deg=np.degrees(h.target + q0), tilt_deg=np.degrees(h.angle + q0))

def demo_jar(t, st, E):
    """virtual hand on the cap: lift it off the table, flip it, carry it over the jar, lower it onto the mouth, release
    height, screw on 2 turns clockwise, unscrew 2.5 turns, lift off."""
    lid, jar = E["jar_lid"], E["jar"]
    if "h" not in st:
        st["h"] = Hand(lid); st["jar_xy"] = A(S.link_pos(jar)[0])[:2]
        st["h"].gains([3e3, 3e3, 3e3, 40, 40, 40], [80, 80, 80, 1.5, 1.5, 1.5])
        st["lid_rest_z"] = S.SC["objects"]["jar"]["pos"][2] + S.SC_JAR["lid_seated_z"] + 0.004    # lid frame z when sitting on the thread
    h = st["h"]; q0 = h.q0; q = q0.copy()
    p_lift = smooth(t, 0.3, 1.0); p_flip = smooth(t, 1.0, 2.0); p_move = smooth(t, 2.0, 3.0); p_low = smooth(t, 3.0, 3.8)
    z_hi = q0[2] + 0.08
    q[2] = q0[2] + 0.08 * p_lift
    q[3] = q0[3] - np.pi * p_flip                                   # upside-down -> upright
    q[:2] = q0[:2] + (st["jar_xy"] + st.setdefault("dq", np.zeros(2)) - q0[:2]) * p_move
    if t >= 3.0:
        q[2] = z_hi + (st["lid_rest_z"] + 0.006 - z_hi) * p_low
    turns_on, turns_off = 2.0, 2.5
    yaw = q0[5] - 2 * np.pi * turns_on * ramp(t, 4.2, 8.2) + 2 * np.pi * turns_off * ramp(t, 9.0, 14.0)
    q[5] = yaw
    if t < 4.0:
        h.go(q)
    else:
        if "released" not in st:                                     # height is now free: only the thread moves it
            st["released"] = True; h.gains([3e3, 3e3, 0, 40, 40, 40], [80, 80, 0, 1.5, 1.5, 1.5])
        if t < 14.3:
            h.go(q, dofs=[0, 1, 3, 4, 5])
        else:                                                        # lift off
            if "lift0" not in st: st["lift0"] = h.q()[2]; h.gains([3e3, 3e3, 3e3, 40, 40, 40], [80, 80, 80, 1.5, 1.5, 1.5])
            q[2] = st["lift0"] + 0.06 * smooth(t, 14.3, 15.3); h.go(q)
    lp = S.link_pos(lid)[0]
    return dict(lid_z_above_mouth_mm=1000 * (lp[2] - st["lid_rest_z"] + 0.004), lid_yaw_turns=(A(lid.get_dofs_position())[5] - q0[5]) / (2 * np.pi),
                yaw_target_turns=(yaw - q0[5]) / (2 * np.pi), jar_drift_mm=1000 * np.linalg.norm(S.link_pos(jar)[0] - st.setdefault("j0", S.link_pos(jar)[0])))

def demo_tin(t, st, E):
    """hold the tin body down, pull the slip-fit lid straight up 6 cm, put it back."""
    tin, lid = E["tin"], E["tin_lid"]
    if "hb" not in st:
        st["hb"], st["hl"] = Hand(tin), Hand(lid)
        st["hb"].gains([2e3] * 3 + [30] * 3, [60] * 3 + [1] * 3); st["hl"].gains([2e3] * 3 + [20] * 3, [40] * 3 + [0.6] * 3)
    qb = st["hb"].q0.copy(); st["hb"].go(qb)
    ql = st["hl"].q0.copy(); ql[2] += 0.06 * (smooth(t, 0.5, 2.0) - smooth(t, 3.0, 4.5)); st["hl"].go(ql)
    return dict(lid_rise_mm=1000 * (st["hl"].q()[2] - st["hl"].q0[2]), lid_target_mm=1000 * (ql[2] - st["hl"].q0[2]),
                tin_rise_mm=1000 * (st["hb"].q()[2] - st["hb"].q0[2]))

def demo_marker(t, st, E):
    """hold the marker barrel, pull the cap off along the axis (5 cm), push it back on."""
    mk, cap = E["marker"], E["marker_cap"]
    if "hm" not in st:
        st["hm"], st["hc"] = Hand(mk), Hand(cap)
        st["hm"].gains([800] * 3 + [5] * 3, [20] * 3 + [0.1] * 3); st["hc"].gains([800] * 3 + [5] * 3, [20] * 3 + [0.1] * 3)
        a = np.radians(S.SC["objects"]["marker"]["euler"][2]); st["ax"] = np.array([np.cos(a), np.sin(a), 0])
    st["hm"].go(st["hm"].q0)
    s = 0.05 * (smooth(t, 0.5, 2.0) - smooth(t, 3.0, 4.5)); qc = st["hc"].q0.copy(); qc[:3] += st["ax"] * s; st["hc"].go(qc)
    off = (st["hc"].q() - st["hc"].q0)[:3] @ st["ax"]
    return dict(cap_off_mm=1000 * off, cap_target_mm=1000 * s)

def demo_stick(t, st, E):
    """lift the wooden stick straight out of the bottle, then let go above the mouth so it drops back in."""
    sk = E["stick"]
    if "h" not in st:
        st["h"] = Hand(sk); st["h"].gains([300] * 3 + [0.5] * 3, [6] * 3 + [0.01] * 3)
        st["b"] = A(S.link_pos(E["bottle"])[0])
    q = st["h"].q0.copy()
    if t < 2.6:
        q[2] += 0.20 * smooth(t, 0.3, 1.5); q[:2] += (st["b"][:2] - q[:2]) * smooth(t, 1.5, 2.5)
        q[3:5] = st["h"].q0[3:5] * (1 - smooth(t, 1.5, 2.5)); st["h"].go(q)
    elif "rel" not in st:
        st["rel"] = True; st["h"].gains([0] * 6, [0] * 6)
    p = S.link_pos(sk)[0]
    return dict(stick_z=p[2], stick_xy_from_axis_mm=1000 * np.linalg.norm(p[:2] - st["b"][:2]))


# ------------------------------------------------------------------------------------------------ deformable demos
# Fingers are free rigid boxes, coupled two-way and PD-driven in xyz (rotation held). A clamp closes 4 mm PAST the
# cloth so the overshoot becomes normal force (see memory: genesis-ipc-constraints).
FINGER = dict(size=(0.020, 0.020, 0.012), rho=800.0)
def fingers(n, park):
    def add(scene):
        out = {}
        for i in range(n):
            out[f"finger{i}"] = scene.add_entity(gs.morphs.Box(pos=park[i], size=FINGER["size"]),
                material=gs.materials.Rigid(coup_type="two_way_soft_constraint", coup_friction=1.2, rho=FINGER["rho"]),
                surface=gs.surfaces.Default(color=(0.85, 0.25, 0.20)))
        return out
    return add

def _finger_gains(f, kp=2e5, kv=2e4):
    f.set_dofs_kp(np.array([kp] * 3 + [kp / 50] * 3)); f.set_dofs_kv(np.array([kv] * 3 + [kv / 50] * 3))

def _path(t, keys):
    ts = [k[0] for k in keys]; P = np.array([k[1] for k in keys], float)
    return np.array([np.interp(t, ts, P[:, j]) for j in range(3)])

def demo_cloth(t, st, E):
    """settle 1.5 s; clamp the bottom corner of the hanging part (open-edge side) front/back; lift it up and lay it
    back over the table top; open the clamp."""
    cl, f0, f1 = E["cloth"], E["finger0"], E["finger1"]
    if "init" not in st:
        st["init"] = True; _finger_gains(f0); _finger_gains(f1); st["q0"] = [A(f0.get_dofs_position()), A(f1.get_dofs_position())]
    V = cl.get_state().pos[0].cpu().numpy()
    if t >= 1.5 and "corner" not in st:
        hang = V[V[:, 2] < S.H - 0.02]
        c = hang[np.argmin(hang[:, 2] - 0.3 * hang[:, 0])]          # lowest point toward the open (+x) edge
        st["corner"] = c + np.array([-0.012, 0, 0.015])              # grip a little in from the corner
        print("grip at", np.round(st["corner"], 3), flush=True)
    if "corner" in st:
        c = st["corner"]; y_front = c[1] - 0.012; y_back = c[1] + 0.012
        sq = 0.004 * smooth(t, 2.6, 3.0)                              # squeeze 4 mm past contact
        lift = [(3.0, c), (4.5, c + [0, 0.02, 0.22]), (6.0, c + [0, 0.30, 0.22]), (7.0, c + [0, 0.30, 0.05])]
        p = _path(t, [(0.0, c)] + lift) if t >= 3.0 else c
        open_ = smooth(t, 7.2, 7.6) * 0.03
        tgt0 = p + [0, -(0.012 + FINGER["size"][1] / 2) + 0.002 - (0.04 * (1 - smooth(t, 1.6, 2.6))) + sq - open_, 0]
        tgt1 = p + [0, +(0.012 + FINGER["size"][1] / 2) - 0.002 + (0.04 * (1 - smooth(t, 1.6, 2.6))) - sq + open_, 0]
        q0, q1 = st["q0"]
        if t < 1.6:
            tgt0 = q0[:3] + (tgt0 - q0[:3]) * 0; tgt1 = q1[:3] + (tgt1 - q1[:3]) * 0
        a0 = smooth(t, 1.5, 2.2)
        f0.control_dofs_position(np.r_[q0[:3] + (tgt0 - q0[:3]) * a0, q0[3:]]); f1.control_dofs_position(np.r_[q1[:3] + (tgt1 - q1[:3]) * a0, q1[3:]])
    else:
        f0.control_dofs_position(st["q0"][0]); f1.control_dofs_position(st["q0"][1])
    return dict(cloth_minz=float(V[:, 2].min()), cloth_on_table_frac=float((V[:, 2] > S.H - 0.005).mean()))

def demo_paper(t, st, E):
    """a fingertip presses on the top sheet's free corner and slides it 6 cm to the right; the bottom sheet should stay."""
    top, bot, f = E["paper_top"], E["paper_bottom"], E["finger0"]
    if "init" not in st:
        st["init"] = True; _finger_gains(f, 3e4, 3e3); st["q0"] = A(f.get_dofs_position())
        Vt = top.get_state().pos[0].cpu().numpy(); st["vt0"] = Vt.copy(); st["vb0"] = bot.get_state().pos[0].cpu().numpy().copy()
        st["pt"] = Vt[np.argmax(Vt[:, 0] - Vt[:, 1])] + np.array([-0.03, 0.03, 0])        # near the front-right corner
    pt = st["pt"]; zc = pt[2] + FINGER["size"][2] / 2 + 0.0015
    keys = [(0.0, st["q0"][:3]), (0.6, pt + [0, 0, 0.05]), (1.0, [pt[0], pt[1], zc - 0.003]),
            (3.0, [pt[0] + 0.06, pt[1], zc - 0.003]), (3.4, [pt[0] + 0.06, pt[1], zc + 0.05])]
    f.control_dofs_position(np.r_[_path(t, keys), st["q0"][3:]])
    Vt = top.get_state().pos[0].cpu().numpy(); Vb = bot.get_state().pos[0].cpu().numpy()
    return dict(top_shift_mm=1000 * float(np.mean(Vt[:, 0] - st["vt0"][:, 0])), bottom_shift_mm=1000 * float(np.mean(Vb[:, 0] - st["vb0"][:, 0])))

def demo_pad(t, st, E):
    """clamp the pad's front-right corner (above/below) and lift it 10 cm: the 3 mm pad should curl like thick cloth."""
    pad, f0, f1 = E["mousepad"], E["finger0"], E["finger1"]
    if "init" not in st:
        st["init"] = True; _finger_gains(f0); _finger_gains(f1); st["q0"] = [A(f0.get_dofs_position()), A(f1.get_dofs_position())]
        V = pad.get_state().pos[0].cpu().numpy(); st["c"] = V[np.argmax(V[:, 0] - V[:, 1])] + np.array([-0.012, 0.012, 0])
    c = st["c"]; h = FINGER["size"][2] / 2; q0, q1 = st["q0"]
    # lower jaw slides under the corner along +x from outside the pad edge, upper jaw comes down from above
    sq = 0.004 * smooth(t, 1.6, 2.0); up = 0.10 * smooth(t, 2.2, 4.0)
    lo = _path(t, [(0.0, q1[:3]), (0.8, c + [0.04, 0, -0.0]), (1.4, c + [0.0, 0, 0])]) if t < 1.4 else c.copy()
    lo = lo.copy(); lo[2] = S.H + h + 0.0005 if t < 2.2 else S.H + h + 0.0005 + up
    hi_ = c + [0, 0, 2 * h + 0.004 + 0.003 - sq + up]
    hi_ = _path(t, [(0.0, q0[:3]), (1.0, c + [0, 0, 0.05]), (1.5, hi_)]) if t < 1.5 else hi_
    f0.control_dofs_position(np.r_[hi_, q0[3:]]); f1.control_dofs_position(np.r_[lo, q1[3:]])
    V = pad.get_state().pos[0].cpu().numpy()
    return dict(pad_maxz_mm=1000 * float(V[:, 2].max() - S.H), mouse_z_mm=1000 * float(S.link_pos(E["mouse"])[0][2] - S.H))

def demo_mouse(t, st, E):
    """a fingertip pushes the mouse 8 cm along -x across the pad."""
    f, ms = E["finger0"], E["mouse"]
    if "init" not in st:
        st["init"] = True; _finger_gains(f, 3e4, 3e3); st["q0"] = A(f.get_dofs_position()); st["m0"] = S.link_pos(ms)[0]
    m = st["m0"]; z = S.H + 0.025
    keys = [(0.0, st["q0"][:3]), (0.8, [m[0] + 0.07, m[1], z]), (1.2, [m[0] + 0.05, m[1], z]), (3.5, [m[0] - 0.04, m[1], z]), (4.0, [m[0] - 0.04, m[1], z + 0.06])]
    f.control_dofs_position(np.r_[_path(t, keys), st["q0"][3:]])
    return dict(mouse_dx_mm=1000 * float(S.link_pos(ms)[0][0] - m[0]), mouse_dz_mm=1000 * float(S.link_pos(ms)[0][2] - m[2]))

DEMOS = dict(  # name: (fn, duration, driven entities -> coupling, cameras, skip)
    laptop=(demo_laptop, 6.0, {}, dict(pos=(-0.15, -0.95, 1.20), lookat=(-0.36, 0.03, 0.88), fov=34)),
    monitor=(demo_monitor, 6.5, {}, dict(pos=(0.95, -0.55, 1.15), lookat=(0.15, 0.28, 1.05), fov=40)),
    jar=(demo_jar, 15.5, {"jar_lid": "two_way_soft_constraint"}, dict(pos=(0.80, -0.70, 1.05), lookat=(0.50, -0.29, 0.86), fov=30)),
    tin=(demo_tin, 5.0, {"tin": "two_way_soft_constraint", "tin_lid": "two_way_soft_constraint"}, dict(pos=(0.55, -0.25, 1.00), lookat=(0.33, 0.15, 0.86), fov=30)),
    marker=(demo_marker, 5.0, {"marker": "two_way_soft_constraint", "marker_cap": "two_way_soft_constraint"}, dict(pos=(0.15, -0.55, 0.95), lookat=(0.14, -0.22, 0.79), fov=28)),
    stick=(demo_stick, 5.0, {"stick": "two_way_soft_constraint"}, dict(pos=(0.75, -0.25, 1.05), lookat=(0.57, 0.17, 0.90), fov=30)),
    cloth=(demo_cloth, 8.0, {}, dict(pos=(0.25, -1.05, 1.00), lookat=(-0.22, -0.30, 0.70), fov=40)),
    paper=(demo_paper, 3.6, {}, dict(pos=(0.30, -0.65, 1.05), lookat=(0.11, -0.22, 0.78), fov=32)),
    pad=(demo_pad, 4.5, {}, dict(pos=(0.95, -0.55, 1.00), lookat=(0.49, -0.08, 0.80), fov=34)),
    mouse=(demo_mouse, 4.0, {}, dict(pos=(0.40, -0.55, 0.95), lookat=(0.39, -0.07, 0.80), fov=30)),
)
NEEDS = {"cloth": ("cloth",), "paper": ("paper_top", "paper_bottom", "marker", "marker_cap"), "pad": ("mousepad", "mouse"), "mouse": ("mousepad", "mouse")}
PARK = {"cloth": [(-0.45, -0.70, 0.60), (-0.45, -0.30, 0.95)], "paper": [(0.30, -0.40, 0.90)], "pad": [(0.70, -0.30, 0.90), (0.75, -0.30, 0.80)],
        "mouse": [(0.55, -0.10, 0.90)]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=list(DEMOS)); ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--T", type=float, default=None); ap.add_argument("--res", type=int, nargs=2, default=(960, 720))
    ap.add_argument("--out", default=os.path.join(S.ROOT, "out", "demos")); ap.add_argument("--full", action="store_true",
                    help="keep every object (default drops the deformables far from the demo to save time)")
    ap.add_argument("--photo-cam", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fn, T, coup, cam = DEMOS[a.demo]; T = a.T or T
    gs.init(backend=gs.gpu, logging_level="warning")
    S.SC_JAR = json.load(open(os.path.join(S.A, "specs.json")))["jar"]
    keep = set(NEEDS.get(a.demo, ())) | ({"paper_top", "paper_bottom"} if a.demo == "marker" else set())
    skip = () if a.full else tuple(k for k in ("cloth", "paper_top", "paper_bottom", "mousepad") if k not in keep)
    extra = fingers(len(PARK[a.demo]), PARK[a.demo]) if a.demo in PARK else None
    scene, E, C = S.build(dt=a.dt, couple=coup, skip=skip, cameras=(a.photo_cam,) if a.photo_cam else (),
                          extra_cams={"demo": dict(res=tuple(a.res), **cam)}, extra=extra)
    t0 = time.time(); scene.build(); print(f"build {time.time()-t0:.1f}s", flush=True)
    for k, c in C.items(): c.start_recording(save_to_filename=os.path.join(a.out, f"{a.demo}_{k}.mp4"), fps=50)
    st, log = {}, []; n = int(T / a.dt); t0 = time.time()
    for i in range(n):
        t = i * a.dt
        row = fn(t, st, E); scene.step()
        if i % 5 == 0:
            row = {k: round(float(v), 4) for k, v in row.items()}; row["t"] = round(t, 3)
            row["ms_per_step"] = round((time.time() - t0) / (i + 1) * 1000); log.append(row)
            if i % 25 == 0: print(row, flush=True)
    for c in C.values(): c.stop_recording()
    json.dump(log, open(os.path.join(a.out, f"{a.demo}.json"), "w"), indent=1)
    print("done", a.demo, f"{(time.time()-t0)/n*1000:.0f} ms/step")
