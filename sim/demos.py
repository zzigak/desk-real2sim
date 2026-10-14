"""Exercise each mechanism of the desk scene. Everything is driven inside libuipc: hinges through their IPC-native
drives (ipc_joints.py), every other driven part through an IPC-native soft-transform 'hand' (ipc_hands.py) whose aim
follows a scripted trajectory; contact, threads, friction and the deformables respond physically. Logs metrics to
<out>/<demo>.json and records <out>/<demo>_<cam>.mp4."""
import argparse, json, os, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import genesis as gs
import scene as S
import ipc_grasps
import ipc_hands

SPEC = json.load(open(os.path.join(S.A, "specs.json")))


def smooth(t, t0, t1):
    s = float(np.clip((t - t0) / (t1 - t0), 0, 1)); return s * s * (3 - 2 * s)


def pose(p, R=np.eye(3)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M


def interp(t, keys):
    ts = [k[0] for k in keys]; P = np.array([k[1] for k in keys], float)
    return np.array([np.interp(t, ts, P[:, j]) for j in range(P.shape[1])])


def start_pose(name):
    o = S.SC["objects"][name]; return pose(o["pos"], Rot.from_euler("xyz", o["euler"], degrees=True).as_matrix())


def hand(name, obj, fn, **kw):
    m, I = S.mass_props(obj)   # mass properties as libuipc integrates the part: force logs and strength limits use them
    return ipc_hands.Hand(name, None, hold_then(None, fn), start_pose(obj), mass=m, I_link=I, **kw)


# person-like hands for the fit demos (ipc_hands.py): arm-like stiffness and a strength limit, so a fit that holds
# harder than a hand can pull stops it. The other demos keep the default stiff (eta 100) hands.
GRIP = dict(K=1000.0, f_max=20.0, tau_max=1.0)       # the hand doing the work
BRACE = dict(K=2000.0, f_max=30.0, tau_max=2.0)      # the hand holding the other part down


def hold_then(T0, fn):
    """trajectory that holds the start pose T0 (captured on the first call) and otherwise defers to fn(t, T0, Tcur)."""
    st = {}
    def traj(t, Tcur):
        st.setdefault("T0", Tcur.copy())
        return fn(t, st["T0"], Tcur)
    return traj



FAR = np.array([0.0, 0.0, -5.0])
FINGER = dict(size=(0.020, 0.020, 0.012), rho=4000.0, eta=20.0)


def finger_hand(E, i, fn, eta=None):
    eta = eta or FINGER["eta"]
    return ipc_hands.Hand(f"finger{i}", None, hold_then(None, fn), pose(E["_park"][i]), eta_p=eta, eta_a=eta * 5)


def verts(E, k):
    return E[k].get_state().pos[0].cpu().numpy()


# ------------------------------------------------------------------------------------------------ 1. paper
def setup_paper_fold(E):
    """dog-ear the top sheet: grip its front-right corner, lift it, carry it 8 cm diagonally inward and lay it down,
    tap a low-friction fingertip down three times along the fold, let go. The crease-plastic paper keeps the fold."""
    st = {}
    def corner(V):
        if "c" not in st:
            st["c"] = V[np.argmax(V[:, 0] - V[:, 1])].copy(); st["d"] = np.array([-0.08, 0.08, 0.0])
        return st["c"]
    def plan(t, V):
        if t < 0.4 or t > 4.6: return None
        c = corner(V); d = st["d"]; s = smooth(t, 1.0, 2.6)
        lift = 0.035 * np.sin(np.pi * s) + 0.02 * smooth(t, 0.4, 1.0) * (1 - smooth(t, 1.0, 1.4))
        return c, c + d * s + [0, 0, lift + 0.0006 * s]
    def press(t, T0, Tc):
        """three taps straight down along the fold line (no sliding, so the sheets are not dragged)."""
        if "c" not in st or t < 2.8: return T0
        c, d = st["c"], st["d"]; m = c + d / 2; u = np.array([1.0, 1.0, 0]) / np.sqrt(2)
        z = c[2] + FINGER["size"][2] / 2 + 0.0004 - 0.0015; up = z + 0.02
        keys = [(2.8, [*(m - 0.04 * u)[:2], up])]
        for k, off in enumerate((-0.04, 0.0, 0.04)):
            t0 = 2.9 + 0.5 * k; xy = (m + off * u)[:2]
            keys += [(t0, [*xy, up]), (t0 + 0.15, [*xy, z]), (t0 + 0.3, [*xy, z]), (t0 + 0.45, [*xy, up])]
        return pose(interp(t, keys), T0[:3, :3])
    E["_paper"] = st
    return [finger_hand(E, 0, press, eta=60.0)], [ipc_grasps.Grip("corner", E["paper_top"], plan, radius=0.010)]


def step_paper_fold(t, st, E):
    V = verts(E, "paper_top"); c = E["_paper"].get("c")
    if c is None: return dict(corner_lift_mm=0.0, fold_progress=0.0)
    i = st.setdefault("ci", int(np.argmin(np.linalg.norm(V - c, axis=1))))
    disp = V[i] - c
    return dict(corner_lift_mm=1000 * float(V[i, 2] - c[2]), fold_progress=float(disp[:2] @ E["_paper"]["d"][:2] / 0.0128))


# ------------------------------------------------------------------------------------------------ 2. cloth
def _hang_corners(V):
    hang = V[V[:, 2] < S.H - 0.05]
    lo = hang[hang[:, 2] < hang[:, 2].min() + 0.02]
    return lo[np.argmin(lo[:, 0])].copy(), lo[np.argmax(lo[:, 0])].copy()


def setup_cloth_fold(E):
    """settle the drape; grip the two bottom corners of the hanging part, swing them up over the table edge and lay
    them on the cloth's back edge: the cloth is folded in half on the table."""
    st = {}
    def plan_for(k):
        def plan(t, V):
            if t < 1.5 or t > 6.6: return None
            if "c" not in st: st["c"] = _hang_corners(V)
            c = st["c"][k]; back_y = float(V[:, 1].max()) if "back" not in st else st["back"]; st["back"] = back_y
            tgt = interp(t, [(1.5, c), (3.0, [c[0], c[1] - 0.05, S.H + 0.12]), (4.8, [c[0], back_y - 0.015, S.H + 0.08]),
                             (6.2, [c[0], back_y - 0.015, S.H + 0.012])])
            return c, tgt
        return plan
    return [], [ipc_grasps.Grip(f"c{k}", E["cloth"], plan_for(k), radius=0.014) for k in (0, 1)]


def step_cloth(t, st, E):
    V = verts(E, "cloth")
    return dict(cloth_minz_mm=1000 * float(V[:, 2].min() - S.H), on_table_frac=float((V[:, 2] > S.H - 0.005).mean()))


def setup_cloth_drop(E):
    """settle the drape; grip the middle of the cloth, lift it 35 cm so it hangs, carry it 5 cm in, let go: it falls
    and heaps on the table."""
    st = {}
    def plan(t, V):
        if t < 1.5 or t > 4.2: return None
        if "c" not in st:
            on = V[V[:, 2] > S.H]; ctr = on[:, :2].mean(0); top = on[on[:, 2] > on[:, 2].max() - 0.004]
            st["c"] = top[np.argmin(np.linalg.norm(top[:, :2] - ctr, axis=1))].copy()
        c = st["c"]
        return c, interp(t, [(1.5, c), (3.2, c + [0, 0.0, 0.35]), (4.0, c + [0, 0.05, 0.35])])
    return [], [ipc_grasps.Grip("mid", E["cloth"], plan, radius=0.025)]


# ------------------------------------------------------------------------------------------------ 3. marker
def setup_marker(E):
    """hold the barrel, pull the cap 5 cm off along the axis, lift it a little and let it drop onto the paper. Both hands
    have human strength (GRIP); the cap only comes off once the pull beats the click ring's hold."""
    a = np.radians(S.SC["objects"]["marker"]["euler"][2]); ax = np.array([np.cos(a), np.sin(a), 0]); E["_ax"] = ax
    barrel = hand("barrel", "marker", lambda t, T0, Tc: T0, **BRACE)
    cap = hand("cap", "marker_cap", lambda t, T0, Tc: None if t > 2.7 else
               pose(T0[:3, 3] + ax * 0.05 * smooth(t, 0.4, 1.9) + [0, 0, 0.02 * smooth(t, 1.9, 2.5)], T0[:3, :3]), **GRIP)
    return [barrel, cap], []


def step_marker(t, st, E):
    hb, hc = E["_hands"]; st.setdefault("c0", hc.T[:3, 3].copy()); st.setdefault("b0", hb.T[:3, 3].copy())
    return dict(cap_off_mm=1000 * float((hc.T[:3, 3] - st["c0"]) @ E["_ax"]), cap_z_mm=1000 * float(hc.T[2, 3] - st["c0"][2]),
                barrel_moved_mm=1000 * float(np.linalg.norm(hb.T[:3, 3] - st["b0"])),
                pull_N=float(hc.F @ E["_ax"]) if hc.is_active else 0.0, cap_saturated=float(hc.saturated[0]))


# ------------------------------------------------------------------------------------------------ 4. mouse pad
def setup_pad_pick(E):
    """grip the pad's two front corners and lift it 25 cm: it hangs like a heavy cloth and the mouse slides off."""
    st = {}
    def plan_for(k):
        def plan(t, V):
            if t < 0.6: return None
            if "c" not in st:
                front = V[V[:, 1] < V[:, 1].min() + 0.01]
                st["c"] = (front[np.argmin(front[:, 0])] + [0.012, 0.012, 0], front[np.argmax(front[:, 0])] + [-0.012, 0.012, 0])
            c = st["c"][k]
            return c, c + [0, 0, 0.25 * smooth(t, 0.8, 2.6) + 0.001]
        return plan
    return [], [ipc_grasps.Grip(f"p{k}", E["mousepad"], plan_for(k), radius=0.015) for k in (0, 1)]


def step_pad(t, st, E):
    V = verts(E, "mousepad")
    return dict(pad_maxz_mm=1000 * float(V[:, 2].max() - S.H), mouse_z_mm=1000 * float(S.link_pos(E["mouse"])[0][2] - S.H))


# ------------------------------------------------------------------------------------------------ 5. bottle + stick
def setup_bottle_tip(E):
    """a fingertip pushes the bottle near its rim toward -x until it tips over; the wooden stick spills out."""
    b = np.array(S.SC["objects"]["bottle"]["pos"]); r = SPEC["bottle"]["r"]; zc = b[2] + 0.13
    x0 = b[0] + r + FINGER["size"][0] / 2 + 0.003
    fn = lambda t, T0, Tc: pose(interp(t, [(0.0, T0[:3, 3]), (0.7, [x0 + 0.02, b[1], zc]), (1.0, [x0, b[1], zc]),
                                            (2.2, [x0 - 0.11, b[1], zc - 0.02]), (2.6, [x0 - 0.11, b[1], zc + 0.08])]), T0[:3, :3])
    return [finger_hand(E, 0, fn, eta=60.0)], []


def step_bottle(t, st, E):
    p, q = S.link_pos(E["bottle"]); st.setdefault("p0", p)
    tilt = np.degrees(np.arccos(np.clip(Rot.from_quat(q[[1, 2, 3, 0]]).apply([0, 0, 1])[2], -1, 1)))
    return dict(bottle_tilt_deg=tilt, stick_z_mm=1000 * float(S.link_pos(E["stick"])[0][2] - S.H))


# ------------------------------------------------------------------------------------------------ 6. jar
def setup_jar(E):
    """cap: lift off the table, flip upright, carry over the jar and set it on the thread (it cannot be pushed past the
    top crest, so it rests there), then turn it clockwise (seen from above) up to 2.5 turns with a 3 N push and a
    1.5 N m wrist: it catches, runs down the thread, seats on the rim, and the wrist stalls. The other hand holds the
    jar."""
    J = SPEC["jar"]; seated = np.array(S.SC["objects"]["jar"]["pos"]) + [0, 0, J["lid_seated_z"]]
    engage_z = seated[2] + J["p"] + 0.0003; st = {}
    jar_hold = hand("jar_body", "jar", lambda t, T0, Tc: T0, eta_p=50.0, eta_a=50.0, f_max=40.0, tau_max=3.0)
    def fn(t, T0, Tc):
        z_hi = max(T0[2, 3] + 0.10, seated[2] + 0.05)
        if t < 3.8:
            p = interp(t, [(0.3, T0[:3, 3]), (1.0, T0[:3, 3] + [0, 0, z_hi - T0[2, 3]]), (2.0, T0[:3, 3] + [0, 0, z_hi - T0[2, 3]]),
                           (3.0, [seated[0], seated[1], z_hi]), (3.8, [seated[0], seated[1], engage_z])])
            st["R_up"] = Rot.from_rotvec([np.pi * smooth(t, 1.0, 2.0), 0, 0]).as_matrix() @ T0[:3, :3]
            return pose(p, st["R_up"])
        if t > 9.9: return None
        yaw = -2 * np.pi * 0.5 * float(np.clip(t - 4.3, 0, 5.0)); st["yaw_cmd"] = yaw
        axis = jar_hold.T[:3, 3] + [0, 0, J["lid_seated_z"]]
        push = 3.0 / (cap.eta_p * cap.mass / cap.dt ** 2)    # height left to the thread; the aim sits 3 N below it
        return pose([axis[0], axis[1], Tc[2, 3] - push], Rot.from_rotvec([0, 0, yaw]).as_matrix() @ st["R_up"])
    E["_jar"] = dict(st=st, seated=seated)
    cap = hand("jar_cap", "jar_lid", fn, K=1000.0, K_rot=5.0, f_max=20.0, tau_max=1.5)   # wrist: 5 N m/rad, 1.5 N m
    return [cap, jar_hold], []


def step_jar(t, st, E):
    h = E["_hands"][0]; js = E["_jar"]["st"]
    yaw = Rot.from_matrix(h.T[:3, :3] @ js["R_up"].T).as_rotvec()[2] if "R_up" in js else 0.0
    st["yaw_u"] = st.get("yaw_u", 0.0) + ((yaw - st.get("yaw_p", yaw) + np.pi) % (2 * np.pi) - np.pi); st["yaw_p"] = yaw
    return dict(cap_above_seated_mm=1000 * (h.T[2, 3] - E["_jar"]["seated"][2]), cap_turns=st["yaw_u"] / (2 * np.pi),
                cmd_turns=js.get("yaw_cmd", 0.0) / (2 * np.pi), twist_Nm=-float(h.tau[2]) if h.is_active else 0.0,
                push_N=-float(h.F[2]) if h.is_active else 0.0, twist_saturated=float(h.saturated[1]))


# ------------------------------------------------------------------------------------------------ 7-9. laptop, monitor, key
def setup_laptop(E): return [], []
def step_laptop(t, st, E):
    """fold the lid from the photographed 113 deg to 50 deg and hold it there (IPC-native hinge drive)."""
    h = E["_hinges"]["laptop"]; q0 = S.SC["joints"]["laptop"]["q0"]
    h.target = (np.radians(50) - q0) * smooth(t, 0.5, 2.5)
    return dict(lid_target_deg=np.degrees(h.target + q0), lid_deg=np.degrees(h.angle + q0))


def setup_monitor(E): return [], []
def step_monitor(t, st, E):
    """tilt the panel back to +21 deg, forward to -5 deg, back to the photographed 1.4 deg."""
    h = E["_hinges"]["monitor"]; q0 = S.SC["joints"]["monitor"]["q0"]
    keys = [(0.0, [0.0]), (0.5, [0.0]), (2.0, [np.radians(21) - q0]), (2.5, [np.radians(21) - q0]), (4.5, [np.radians(-5) - q0]),
            (5.0, [np.radians(-5) - q0]), (6.0, [0.0])]
    h.target = float(interp(t, keys)[0])
    return dict(tilt_target_deg=np.degrees(h.target + q0), tilt_deg=np.degrees(h.angle + q0))


def setup_key(E):
    """a fingertip presses the laptop's space bar down and lets go; the key's spring drive returns it."""
    j = S.SC["joints"]["laptop_key"]; c = np.array(j["pivot"]); n = np.array(j["axis"])
    E["_hinges"]["laptop_key"].target = 0.0
    up = c + n * (0.0006 + FINGER["size"][2] / 2 + 0.0004)
    fn = lambda t, T0, Tc: pose(interp(t, [(0.0, T0[:3, 3]), (0.5, up + n * 0.03), (0.8, up), (1.3, up - n * 0.0025), (1.8, up - n * 0.0025),
                                            (2.1, up + n * 0.03)]), T0[:3, :3])
    return [finger_hand(E, 0, fn, eta=150.0)], []


def step_key(t, st, E):
    return dict(key_travel_mm=-1000 * E["_hinges"]["laptop_key"].angle)


# ------------------------------------------------------------------------------------------------ 10. tea tin
def setup_tin(E):
    """hold the tin, lift the slip-fit lid 6 cm off, carry it aside and set it down on the table (human-strength hands)."""
    body = hand("tin_body", "tin", lambda t, T0, Tc: None if t > 3.6 else T0, **BRACE)
    def fn(t, T0, Tc):
        if t > 3.6: return None
        side = np.array([-0.11, -0.02, 0.0])
        p = interp(t, [(0.4, T0[:3, 3]), (1.8, T0[:3, 3] + [0, 0, 0.06]), (2.6, T0[:3, 3] + side + [0, 0, 0.06]),
                       (3.5, T0[:3, 3] + side + [0, 0, -T0[2, 3] + S.H + 0.0025])])
        return pose(p, T0[:3, :3])
    return [body, hand("tin_lid", "tin_lid", fn, **GRIP)], []


def step_tin(t, st, E):
    hb, hl = E["_hands"]; st.setdefault("zb", hb.T[2, 3]); st.setdefault("pl", hl.T[:3, 3].copy())
    return dict(lid_rise_mm=1000 * (hl.T[2, 3] - st["pl"][2]), lid_moved_mm=1000 * float(np.linalg.norm(hl.T[:2, 3] - st["pl"][:2])),
                body_rise_mm=1000 * (hb.T[2, 3] - st["zb"]), lift_N=float(hl.F[2]) if hl.is_active else 0.0,
                lid_saturated=float(hl.saturated[0]))


# name: (setup, step, duration, deformables kept, finger parking spots, camera)
DEMOS = dict(
    paper_fold=(setup_paper_fold, step_paper_fold, 5.0, ("paper_top", "paper_bottom"), [(0.35, -0.40, 0.95)],
                dict(pos=(0.40, -0.62, 1.02), lookat=(0.14, -0.24, 0.78), fov=34)),
    cloth_fold=(setup_cloth_fold, step_cloth, 7.0, ("cloth",), None, dict(pos=(0.30, -1.05, 1.10), lookat=(-0.22, -0.28, 0.72), fov=40)),
    cloth_drop=(setup_cloth_drop, step_cloth, 7.0, ("cloth",), None, dict(pos=(0.30, -1.05, 1.15), lookat=(-0.22, -0.25, 0.85), fov=44)),
    marker=(setup_marker, step_marker, 4.0, ("paper_top", "paper_bottom"), None, dict(pos=(0.15, -0.55, 0.95), lookat=(0.12, -0.22, 0.79), fov=30)),
    pad_pick=(setup_pad_pick, step_pad, 4.5, ("mousepad",), None, dict(pos=(0.10, -0.80, 1.20), lookat=(0.46, -0.02, 0.92), fov=42)),
    bottle_tip=(setup_bottle_tip, step_bottle, 4.0, (), [(0.70, 0.17, 0.95)], dict(pos=(0.55, -0.35, 1.00), lookat=(0.52, 0.17, 0.83), fov=36)),
    jar=(setup_jar, step_jar, 10.5, (), None, dict(pos=(0.74, -0.60, 1.12), lookat=(0.53, -0.30, 0.88), fov=30)),
    laptop=(setup_laptop, step_laptop, 4.0, (), None, dict(pos=(-0.15, -0.95, 1.20), lookat=(-0.36, 0.03, 0.88), fov=34)),
    monitor=(setup_monitor, step_monitor, 6.5, (), None, dict(pos=(1.15, 0.12, 1.15), lookat=(0.15, 0.30, 1.02), fov=40)),
    key=(setup_key, step_key, 2.6, (), [(-0.30, -0.10, 1.00)], dict(pos=(-0.28, -0.30, 0.98), lookat=(-0.38, 0.0, 0.85), fov=26)),
    tin=(setup_tin, step_tin, 4.5, (), None, dict(pos=(0.55, -0.25, 1.00), lookat=(0.30, 0.14, 0.84), fov=32)),
)


def extras(park, n_marks, finger_friction=1.2):
    def add(scene):
        out = {}
        for i, p in enumerate(park or []):
            out[f"finger{i}"] = scene.add_entity(gs.morphs.Box(pos=p, size=FINGER["size"]),
                                                 material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=finger_friction, rho=FINGER["rho"]),
                                                 surface=gs.surfaces.Default(color=(0.85, 0.25, 0.20)))
        for i in range(n_marks):   # where a grip holds the sheet: drawn only, not simulated
            out[f"mark{i}"] = scene.add_entity(gs.morphs.Sphere(pos=tuple(FAR + [i * 0.1, 0, 0]), radius=0.008),
                                               material=gs.materials.Rigid(needs_coup=False, gravity_compensation=1.0),
                                               surface=gs.surfaces.Default(color=(0.85, 0.25, 0.20)))
        return out
    return add


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=list(DEMOS)); ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--T", type=float, default=None); ap.add_argument("--res", type=int, nargs=2, default=(960, 720))
    ap.add_argument("--out", default=os.path.join(S.ROOT, "out", "demos"))
    ap.add_argument("--newton-tol", type=float, default=None)   # libuipc newton/velocity_tol (default 0.05 m/s)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    setup, step_fn, T, keep, park, cam = DEMOS[a.demo]; T = a.T or T
    n_marks = 2 if a.demo in ("cloth_fold", "cloth_drop", "pad_pick", "paper_fold") else 0
    gs.init(backend=gs.gpu, logging_level="warning")
    skip = tuple(k for k in S.SC["deformables"] if k not in keep)
    scene, E, C = S.build(dt=a.dt, skip=skip, cameras=(), extra_cams={"demo": dict(res=tuple(a.res), **cam)}, newton_tol=a.newton_tol,
                          extra=extras(park, n_marks, finger_friction=0.2 if a.demo == "paper_fold" else 1.2))
    E["_park"] = park
    hands, grips = setup(E)
    owner = {"jar_cap": "jar_lid", "jar_body": "jar", "tin_body": "tin", "tin_lid": "tin_lid", "barrel": "marker", "cap": "marker_cap"}
    for h in hands: h.link = E[owner.get(h.name, h.name)].base_link; h.dt = a.dt
    E["_hands"] = hands; ipc_hands.install(hands); ipc_grasps.install(grips)
    t0 = time.time(); scene.build(); print(f"build {time.time()-t0:.1f}s", flush=True)
    for k, c in C.items(): c.start_recording(save_to_filename=os.path.join(a.out, f"{a.demo}.mp4"), fps=50)
    st, log = {}, []; n = int(T / a.dt); t0 = time.time()
    for i in range(n):
        t = i * a.dt
        for h in hands: h.t = t
        for g in grips: g.t = t
        scene.step()
        for j, g in enumerate(grips[:n_marks]):
            E[f"mark{j}"].set_pos(g.target if g.target is not None else FAR + [j * 0.1, 0, 0], relative=False, zero_velocity=True)
        row = step_fn(t, st, E)
        if i % 5 == 0:
            row = {k: round(float(v), 4) for k, v in row.items()}; row["t"] = round(t, 3)
            row["ms_per_step"] = round((time.time() - t0) / (i + 1) * 1000); log.append(row)
            if i % 25 == 0: print(row, flush=True)
    for c in C.values(): c.stop_recording()
    json.dump(log, open(os.path.join(a.out, f"{a.demo}.json"), "w"), indent=1)
    print("done", a.demo, f"{(time.time()-t0)/n*1000:.0f} ms/step")
