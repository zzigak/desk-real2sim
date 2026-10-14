"""Debug: one object (+ optional mate) on a fixed IPC table; print link pose / AABB before and after steps."""
import json, os, sys, time
import numpy as np
import genesis as gs
case = sys.argv[1]
A = os.path.expanduser("~/projects/desk-real2sim/assets"); SP = json.load(open(f"{A}/specs.json"))
H = 0.775
gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), rigid_options=gs.options.RigidOptions(enable_collision=False),
                 coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=0.001, contact_resistance=1e7, newton_semi_implicit_enable=False,
                                                              enable_rigid_rigid_contact=True, enable_rigid_ground_contact=False),
                 show_viewer=False)
scene.add_entity(gs.morphs.Box(pos=(0, 0, H - 0.0125), size=(0.8, 0.5, 0.025), fixed=True), material=gs.materials.Rigid(coup_type="ipc_only"))
def mesh(name, pos, euler=(0, 0, 0), coup="ipc_only", rho=1000):
    return scene.add_entity(gs.morphs.Mesh(file=f"{A}/{name}_col.obj", pos=pos, euler=euler, convexify=False, decimate=False, watertighten=0),
                            material=gs.materials.Rigid(coup_type=coup, rho=rho))
z0 = H + 0.0015; ents = {}
if case in ("jar", "jarlid"): ents["jar"] = mesh("jar", (0, 0, z0), rho=2500)
if case == "jarlid": ents["lid"] = mesh("jar_lid", (0, 0, z0 + SP["jar"]["lid_seated_z"]), rho=700)
if case in ("tin", "tinlid"): ents["tin"] = mesh("tin", (0, 0, z0), rho=780)
if case == "tinlid": ents["lid"] = mesh("tin_lid", (0, 0, z0 + SP["tea_tin"]["lid_seated_z"]), rho=850)
if case == "box": ents["box"] = scene.add_entity(gs.morphs.Box(pos=(0, 0, z0 + 0.02), size=(0.04, 0.04, 0.04)), material=gs.materials.Rigid(coup_type="ipc_only"))
scene.build()
def show(tag):
    for k, e in ents.items():
        aabb = np.array(e.get_AABB().cpu()) if hasattr(e, "get_AABB") else None
        print(tag, k, "pos", np.round(np.array(e.get_pos().cpu()), 4), "dofs", np.round(np.array(e.get_dofs_position().cpu())[:3], 4),
              "aabb z", None if aabb is None else np.round(aabb[:, 2], 4))
show("init ")
for i in range(5):
    t = time.time(); scene.step(); dt = time.time() - t
    show(f"step{i} {dt*1000:.0f}ms")
