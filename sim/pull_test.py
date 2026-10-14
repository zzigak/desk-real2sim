"""Pull-off tests for the desk's fits, in a two-body scene with the demo contact settings (d_hat 1 mm, dt 0.01).

  marker / tin: the body is held still; a stiff hand draws the cap or lid straight off along its axis at constant speed
                and logs the force its spring applies. The peak, minus the part's weight, is the pull-off force.
  jar:          the lid is set on the thread one pitch above its seat and turned clockwise by a hand with the demo's
                strength (3 N push, 1.5 N m torque limit); logs torque, turns and height.
  mass:         the tin lid alone, held by a hand limited to half its weight: it must fall at g/2 (checks the mass
                properties the limits rely on).
Geometry comes from build_assets.make_* with --set overrides, e.g.  --set seat_gap=0.0006  (written to out/pull/<tag>/).

  python sim/pull_test.py marker --set seat_gap=0.0006 --tag m060"""
import argparse, json, os, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import genesis as gs
import build_assets as BA
import ipc_hands, ipc_patches
import scene as S

ap = argparse.ArgumentParser()
ap.add_argument("pair", choices=("marker", "tin", "jar", "mass"))
ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--tag", default=None)
ap.add_argument("--speed", type=float, default=0.025); ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--push", type=float, default=3.0); ap.add_argument("--tau-max", type=float, default=1.5)
ap.add_argument("--video", action="store_true")
ap.add_argument("--scene-pose", action="store_true")
ap.add_argument("--table", action="store_true")
ap.add_argument("--contact", choices=("desk", "robot"), default="desk")   # robot: yam-arcade jars-station settings
ap.add_argument("--robot-fits", action="store_true")                       # start from build_assets.ROBOT_FITS       # a fixed slab 1.5 mm under the body, as on the desk
ap.add_argument("--newton-tol", type=float, default=None)   # place both parts where scene.json has them
ap.add_argument("--part-K", type=float, default=None); ap.add_argument("--body-K", type=float, default=None)   # demo-like soft hands
a = ap.parse_args()
over = {k: float(v) for k, v in (kv.split("=") for kv in a.set)}
if a.robot_fits:
    over = {**BA.ROBOT_FITS[dict(marker="marker", tin="tea_tin", mass="tea_tin", jar="jar")[a.pair]], **over}
CONTACT = dict(contact_d_hat=S.DHAT, contact_resistance=1e7) if a.contact == "desk" else \
    dict(contact_d_hat=2e-4, contact_resistance=1e9, contact_eps_velocity=1e-3)
if a.contact == "robot" and a.newton_tol is None:
    a.newton_tol = 1e-3
tag = a.tag or a.pair + "".join(f"_{k}{v * 1000:g}" for k, v in over.items())
OUT = os.path.join(S.ROOT, "out", "pull", tag); os.makedirs(OUT, exist_ok=True)

fn, names, seat_key, masses, mu = dict(marker=(BA.make_marker, ("marker", "marker_cap"), "cap_seated_z", (0.012, 0.004), 0.4),
                                      tin=(BA.make_tin, ("tin", "tin_lid"), "lid_seated_z", (0.12, 0.035), 0.35),
                                      mass=(BA.make_tin, ("tin", "tin_lid"), "lid_seated_z", (0.12, 0.035), 0.35),
                                      jar=(BA.make_jar, ("jar", "jar_lid"), "lid_seated_z", (0.38, 0.020), (0.3, 0.25)))[a.pair]
meshes, spec = fn(**over)
rho = {}
for (nm, m), mass in zip(meshes.items(), masses):
    m.export(f"{OUT}/{nm}_col.obj"); rho[nm] = mass / m.volume
    open(f"{OUT}/{nm}.urdf", "w").write(
        f'<robot name="{nm}"><link name="{nm}"><inertial><origin xyz="0 0 0"/><mass value="{mass}"/>'
        f'<inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>'
        f'<visual><geometry><mesh filename="{nm}_col.obj"/></geometry></visual>'
        f'<collision><geometry><mesh filename="{nm}_col.obj"/></geometry></collision></link></robot>')
seat = spec[seat_key]
z_part = seat + (spec["p"] + 0.0003 if a.pair == "jar" else 0.0) + (0.10 if a.pair == "mass" else 0.0)

gs.init(backend=gs.gpu, logging_level="warning")
ipc_patches.patch_ipc_only_coupling_forces()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=a.dt, gravity=(0.0, 0.0, -9.81)),
                 rigid_options=gs.options.RigidOptions(enable_collision=False),
                 coupler_options=gs.options.IPCCouplerOptions(**CONTACT,
                                                              newton_semi_implicit_enable=False, enable_rigid_rigid_contact=True,
                                                              newton_tolerance=a.newton_tol),
                 vis_options=gs.options.VisOptions(show_world_frame=False), show_viewer=False)
mus = mu if isinstance(mu, tuple) else (mu, mu)
if a.table:
    scene.add_entity(gs.morphs.Box(pos=(0, 0, -0.0015 - 0.01), size=(0.4, 0.4, 0.02), fixed=True),
                     material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5))
ents = {}
def scene_T(nm):
    o = S.SC["objects"][nm]; M = np.eye(4); M[:3, :3] = Rot.from_euler("xyz", o["euler"], degrees=True).as_matrix(); M[:3, 3] = o["pos"]; return M
POSE = {}
for nm, z in ((names[0], 0.0), (names[1], z_part)):
    POSE[nm] = scene_T(nm) if a.scene_pose else np.eye(4)
    if not a.scene_pose: POSE[nm][2, 3] = z
for (nm, z), f in zip(((names[0], 0.0), (names[1], z_part)), mus):
    o = S.SC["objects"][nm]
    ents[nm] = scene.add_entity(gs.morphs.URDF(file=f"{OUT}/{nm}.urdf", pos=tuple(POSE[nm][:3, 3]),
                                               euler=tuple(o["euler"]) if a.scene_pose else (0, 0, 0), convexify=False, decimate=False,
                                               watertighten=0, merge_fixed_links=False),
                                material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=f, rho=rho[nm]))
cam = scene.add_camera(res=(640, 480), pos=(0.12, -0.16, z_part + 0.06), lookat=(0, 0, z_part), fov=40, GUI=False) if a.video else None
mp = {nm: S.mass_props(nm, f"{OUT}/{nm}_col.obj", rho[nm]) for nm in names}
body, part = names
t0p = 0.3                                     # settle, then act
Tb0, Tp0 = POSE[body].copy(), POSE[part].copy(); AX = Tp0[:3, 2].copy()   # pull along the part's own axis
if a.pair in ("marker", "tin"):
    def traj(t, T):
        p = Tp0.copy(); p[:3, 3] += AX * a.speed * max(t - t0p, 0.0); return p
    hp = ipc_hands.Hand("part", None, traj, Tp0, K=a.part_K, mass=mp[part][0], I_link=mp[part][1], dt=a.dt)
elif a.pair == "mass":
    hp = ipc_hands.Hand("part", None, lambda t, T: Tp0, Tp0, f_max=0.5 * mp[part][0] * 9.81, mass=mp[part][0], I_link=mp[part][1], dt=a.dt)
else:
    st = {}
    def traj(t, T):
        st.setdefault("R0", T[:3, :3].copy())
        k = hp.eta_p * hp.mass / a.dt ** 2   # = K once install() has resolved it
        yaw = -2 * np.pi * 0.5 * max(t - t0p, 0.0)
        return S_pose([0.0, 0.0, T[2, 3] - a.push / k], Rot.from_rotvec([0, 0, yaw]).as_matrix() @ st["R0"])
    hp = ipc_hands.Hand("part", None, traj, Tp0, K=1000.0, K_rot=5.0, f_max=20.0, tau_max=a.tau_max,
                        mass=mp[part][0], I_link=mp[part][1], dt=a.dt)
def S_pose(p, R):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M
hb = ipc_hands.Hand("body", None, lambda t, T: Tb0, Tb0, K=a.body_K, mass=mp[body][0], I_link=mp[body][1], dt=a.dt)
hb.link, hp.link = ents[body].base_link, ents[part].base_link
ipc_hands.install([hb, hp])
scene.build()
if cam: cam.start_recording()
T = dict(marker=t0p + 0.8, tin=t0p + 1.3, jar=t0p + 3.6, mass=0.4)[a.pair]
log, yaw_u, yaw_p = [], 0.0, None; tw = time.time()
for i in range(int(round(T / a.dt))):
    t = i * a.dt; hb.t = hp.t = t
    scene.step()
    if cam and i % 2 == 0: cam.render()
    # hand.T and hand.F are refreshed at the start of a step (in animate): the state at time t, the force over the step before
    R = Rot.from_matrix(hp.T[:3, :3]).as_matrix()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    if yaw_p is not None: yaw_u += (yaw - yaw_p + np.pi) % (2 * np.pi) - np.pi
    yaw_p = yaw
    log.append(dict(t=round(t, 3), dz_mm=1000 * float((hp.T[:3, 3] - Tp0[:3, 3]) @ AX), Fz=float(hp.F @ AX), Fxy=float(np.linalg.norm(hp.F[:2])),
                    tau_z=float(hp.tau[2]), turns=-yaw_u / (2 * np.pi), body_dz_mm=1000 * (hb.T[2, 3] - Tb0[2, 3]), body_Fz=float(hb.F[2]),
                    above_seat_mm=1000 * (hp.T[2, 3] - seat)))
    if i % 25 == 0: print({k: round(v, 4) for k, v in log[-1].items()}, f"{(time.time() - tw) / (i + 1) * 1000:.0f} ms/step", flush=True)
if cam: cam.stop_recording(save_to_filename=f"{OUT}/{a.pair}.mp4", fps=50)
W = hp.mass * 9.81
res = dict(pair=a.pair, over=over, mass=hp.mass, weight_N=W, log=log)
if a.pair in ("marker", "tin"):
    act = [r for r in log if r["t"] > t0p + 0.02]
    pk = max(act, key=lambda r: r["Fz"])
    free = [r["Fz"] for r in act if r["dz_mm"] > dict(marker=8, tin=26)[a.pair]]
    res.update(peak_N=pk["Fz"], peak_at_mm=pk["dz_mm"], pull_off_N=pk["Fz"] - W, free_N=float(np.mean(free)) if free else None)
    print(f"RESULT {tag}: peak hand force {pk['Fz']:.2f} N at {pk['dz_mm']:.1f} mm; minus weight {W:.3f} N -> pull-off {pk['Fz'] - W:.2f} N; "
          f"once free {res['free_N']}")
elif a.pair == "mass":
    zs = [(r["t"], r["dz_mm"]) for r in log]
    print("RESULT mass: m %.4f kg; z(t) vs -g/4 t^2:" % hp.mass, [(t, round(z, 2), round(-1000 * 9.81 / 4 * t ** 2, 2)) for t, z in zs[::5]])
else:
    run = [r for r in log if t0p + 0.3 < r["t"] and r["above_seat_mm"] > 1.0]
    res.update(running_tau_Nm=float(np.median([-r["tau_z"] for r in run])) if run else None,
               max_running_tau_Nm=float(max(-r["tau_z"] for r in run)) if run else None,
               end_turns=log[-1]["turns"], end_above_seat_mm=log[-1]["above_seat_mm"])
    print(f"RESULT {tag}: running torque median {res['running_tau_Nm']} max {res['max_running_tau_Nm']} N m; "
          f"end {res['end_turns']:.2f} turns, {res['end_above_seat_mm']:.2f} mm above design seat")
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
