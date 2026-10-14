"""Minimal repro for the libuipc 'row index out of submatrix range' crash: one cloth over a table, toggles."""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, genesis as gs
ap = argparse.ArgumentParser()
ap.add_argument("--plane", type=int, default=1); ap.add_argument("--table", default="urdf", choices=["urdf", "box"])
ap.add_argument("--nocoll", type=int, default=1); ap.add_argument("--patch", default="mine", choices=["mine", "franka", "none"])
ap.add_argument("--mesh", default="mine", choices=["mine", "towel"]); ap.add_argument("--rr", type=int, default=1)
ap.add_argument("--steps", type=int, default=5); ap.add_argument("--rg", type=int, default=1); ap.add_argument("--tfric", type=float, default=None); ap.add_argument("--pfric", type=float, default=None); ap.add_argument("--merge", type=int, default=1); ap.add_argument("--patch3", type=int, default=0)
a = ap.parse_args(); print(vars(a), flush=True)
A = os.path.expanduser("~/projects/desk-real2sim/assets"); H = 0.775
gs.init(backend=gs.gpu, logging_level="warning")
if a.patch == "mine":
    import cloth_fix; cloth_fix.patch({0.001: (2e3, 0.3), 1e-4: (1e5, 0.3), 0.003: (9e3, 0.4)} if a.patch3 else {0.001: (2e3, 0.3)})
elif a.patch == "franka":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import genesis_cloth_fix; genesis_cloth_fix.patch(2e3, 0.3)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01),
                 rigid_options=gs.options.RigidOptions(enable_collision=not a.nocoll),
                 coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=1e-3, contact_resistance=1e7, newton_semi_implicit_enable=False,
                                                              enable_rigid_rigid_contact=bool(a.rr), enable_rigid_ground_contact=bool(a.rg)), show_viewer=False)
if a.plane: scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only", **({"coup_friction": a.pfric} if a.pfric is not None else {})))
if a.table == "urdf":
    scene.add_entity(gs.morphs.URDF(file=f"{A}/urdf/table.urdf", fixed=True, convexify=False, decimate=False, watertighten=0, merge_fixed_links=bool(a.merge)),
                     material=gs.materials.Rigid(coup_type="ipc_only", rho=1688, **({"coup_friction": a.tfric} if a.tfric is not None else {})))
else:
    scene.add_entity(gs.morphs.Box(pos=(0, 0, H - 0.0125), size=(1.254, 0.756, 0.025), fixed=True), material=gs.materials.Rigid(coup_type="ipc_only"))
if a.mesh == "mine":
    m = gs.morphs.Mesh(file=f"{A}/urdf/cloth.obj")
else:
    m = gs.morphs.Mesh(file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "towel.obj"), pos=(0, 0, H + 0.004))
scene.add_entity(m, material=gs.materials.FEM.Cloth(E=2e6, nu=0.3, rho=300, thickness=0.001, bending_stiffness=0.66, friction_mu=0.6))
scene.build(); print("built", flush=True)
for i in range(a.steps): scene.step(); print("step", i, flush=True)
print("OK", flush=True)
