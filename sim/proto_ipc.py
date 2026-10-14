"""IPC prototype: threaded jar lid unscrew, tin lid pull, marker cap pull, folded cloth drape over the table edge."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/franka-deformables"))
import genesis as gs
import genesis_cloth_fix
from cloth_mesh import folded_cloth

ap = argparse.ArgumentParser(); ap.add_argument("--dt", type=float, default=0.01); ap.add_argument("--T", type=float, default=4.0)
ap.add_argument("--out", default="../out/proto"); ap.add_argument("--no-cloth", action="store_true")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
A = os.path.join(os.path.dirname(__file__), "..", "assets"); SP = json.load(open(f"{A}/specs.json"))
H = 0.775; TH = 0.025
gs.init(backend=gs.gpu, logging_level="warning")
genesis_cloth_fix.patch(2e6 * 0.001, 0.3)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0, 0, -9.81)),
    rigid_options=gs.options.RigidOptions(enable_collision=False),          # all contact through IPC
    coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=0.001, contact_resistance=1e7,
        newton_semi_implicit_enable=False, enable_rigid_rigid_contact=True, enable_rigid_ground_contact=False),
    vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.4, 0.4, 0.4)), show_viewer=False)
table = scene.add_entity(gs.morphs.Box(pos=(0, 0, H - TH / 2), size=(0.8, 0.5, TH), fixed=True),
                         material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5),
                         surface=gs.surfaces.Default(color=(0.80, 0.62, 0.38)))
def mesh(name, pos, euler=(0, 0, 0), coup="ipc_only", fr=0.5, rho=1000, color=(0.7, 0.7, 0.7)):
    return scene.add_entity(gs.morphs.Mesh(file=f"{A}/{name}_col.obj", pos=pos, euler=euler, convexify=False, decimate=False, watertighten=0),
                            material=gs.materials.Rigid(coup_type=coup, coup_friction=fr, rho=rho),
                            surface=gs.surfaces.Default(color=color))
z0 = H + 0.0015
J = SP["jar"]; jar = mesh("jar", (-0.2, 0.1, z0), rho=2500, color=(0.75, 0.85, 0.85))
jl = mesh("jar_lid", (-0.2, 0.1, z0 + J["lid_seated_z"]), coup="two_way_soft_constraint", rho=700, color=(0.15, 0.15, 0.17))
T = SP["tea_tin"]; tin = mesh("tin", (0.05, 0.1, z0), rho=780, color=(0.9, 0.88, 0.8))
tl = mesh("tin_lid", (0.05, 0.1, z0 + T["lid_seated_z"]), coup="two_way_soft_constraint", rho=850, color=(0.9, 0.88, 0.8))
M = SP["marker"]; r = M["r"] + 0.0012
mk = mesh("marker", (0.15, -0.15, H + r), euler=(0, 90, 0), rho=550, color=(0.95, 0.95, 0.95))
cp = mesh("marker_cap", (0.15 + M["cap_seated_z"], -0.15, H + r), euler=(0, 90, 0), coup="two_way_soft_constraint",
          rho=450, color=(0.05, 0.05, 0.05))
cloth = None
if not args.no_cloth:
    cm = folded_cloth(0.20, 0.30, gap=0.010, h=0.0075, fold_segments=6); cm.export(f"{A}/proto_cloth.obj")
    # 0.12 m on the table, 0.18 m sticking out over the front edge (y = -0.25), starts flat
    cloth = scene.add_entity(gs.morphs.Mesh(file=f"{A}/proto_cloth.obj", pos=(-0.35, -0.25 - 0.18, H + 0.0012)),
        material=gs.materials.FEM.Cloth(E=2e6, nu=0.3, rho=300, thickness=0.001, bending_stiffness=0.66, friction_mu=0.6),
        surface=gs.surfaces.Default(color=(0.75, 0.92, 0.6)))
cam = scene.add_camera(res=(960, 600), pos=(0.2, -1.0, H + 0.45), lookat=(-0.05, 0.0, H + 0.05), fov=40, GUI=False)
t0 = time.time(); scene.build(); print(f"build {time.time()-t0:.1f}s", flush=True)

def drive(ent, kp_lin, kp_rot, kv_lin, kv_rot):
    ent.set_dofs_kp(np.r_[kp_lin, kp_rot]); ent.set_dofs_kv(np.r_[kv_lin, kv_rot])
q_jl = np.array(jl.get_dofs_position().cpu()); q_tl = np.array(tl.get_dofs_position().cpu()); q_cp = np.array(cp.get_dofs_position().cpu())
# jar lid: hold x, y, tilt; leave z FREE (the thread decides); twist about z
drive(jl, [2e3, 2e3, 0], [50, 50, 20], [50, 50, 0], [2, 2, 1])
# tin lid / cap: free until the pull starts
drive(tl, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]); drive(cp, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
cam.start_recording()
n = int(args.T / args.dt); log = []
t0 = time.time()
for i in range(n):
    t = i * args.dt
    if t > 0.5:                          # unscrew at 1 turn / 2 s (counter-clockwise from above)
        tgt = q_jl.copy(); tgt[5] = q_jl[5] + 2 * np.pi * 0.5 * (t - 0.5)
        jl.control_dofs_position(tgt[[0, 1, 3, 4, 5]], dofs_idx_local=[0, 1, 3, 4, 5])
    if t > 1.0 and i % 1 == 0:          # tin lid pull: 2 cm/s straight up, start from where it is
        if not hasattr(drive, "tl0"):
            drive.tl0 = np.array(tl.get_dofs_position().cpu()); drive(tl, [3e3, 3e3, 3e3], [30, 30, 30], [60, 60, 60], [1, 1, 1])
        tg = drive.tl0.copy(); tg[2] += 0.02 * (t - 1.0); tl.control_dofs_position(tg)
    if t > 1.0:                          # cap pull along +x at 2 cm/s
        if not hasattr(drive, "cp0"):
            drive.cp0 = np.array(cp.get_dofs_position().cpu()); drive(cp, [500, 500, 500], [5, 5, 5], [10, 10, 10], [0.1, 0.1, 0.1])
        tg = drive.cp0.copy(); tg[0] += 0.02 * (t - 1.0); cp.control_dofs_position(tg)
    scene.step()
    if i % 2 == 0: cam.render()
    if i % 10 == 0:
        a = lambda e: np.array(e.get_dofs_position().cpu())
        row = dict(t=round(t, 2), lid_z=a(jl)[2] - q_jl[2], lid_yaw=a(jl)[5], jar_z=a(jar)[2], tinlid_z=a(tl)[2] - q_tl[2],
                   tin_z=a(tin)[2], cap_x=a(cp)[0] - q_cp[0], marker_x=a(mk)[0])
        if cloth is not None:
            v = cloth.get_state().pos[0].cpu().numpy() if hasattr(cloth.get_state(), "pos") else None
            if v is not None: row["cloth_minz"] = float(v[:, 2].min()); row["cloth_miny"] = float(v[:, 1].min())
        log.append(row)
        print({k: (round(float(x), 4) if not isinstance(x, str) else x) for k, x in row.items()}, f"{(time.time()-t0)/(i+1)*1000:.0f} ms/step", flush=True)
cam.stop_recording(save_to_filename=f"{args.out}/proto.mp4", fps=int(0.5 / args.dt))
json.dump(log, open(f"{args.out}/log.json", "w"), indent=1)
