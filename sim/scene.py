"""Genesis + libuipc recreation of the desk. build(...) returns (scene, entities, cameras).

Physics split: Genesis' rigid solver does dynamics, joints and PD actuation with its own collision OFF; libuipc (IPC)
does every contact - rigid-rigid (threads, slip-fit lids, stick in bottle), rigid-cloth and cloth-cloth.
Coupling per rigid entity: 'ipc_only' for passive free objects (IPC integrates them), 'two_way_soft_constraint' for
anything a demo drives with PD targets (a stand-in for a hand), 'two_way_soft_constraint' for the articulations too (external_articulation gave NaN on the monitor).
"""
import json, os
import numpy as np
import genesis as gs
import cloth_fix
import ipc_joints
import ipc_patches

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
A = os.path.join(ROOT, "assets")
SC = json.load(open(os.path.join(A, "scene.json")))
H = SC["H"]
DHAT = 1e-3
# deformable materials (SI). Bending "stiffness" is Genesis' discrete-shell parameter; 0.66 = 2.4 cm bending length at
# 0.30 kg/m^2 (clamped-cantilever calibration, ~/projects/franka-deformables/calib). Paper and pad: own clamped
# cantilevers (calib/cantilever_shell.py, calib/fit_shells.json) to B = 2.5e-4 N m (80 gsm) and 3e-3 N m (rubber pad).
FABRIC = {
    "cloth": dict(E=2e6, nu=0.3, rho=300.0, thickness=0.001, bending_stiffness=0.66, friction_mu=0.6),      # microfibre, ~300 gsm
    # paper: Genesis' crease-plastic shell (StrainPlasticDiscreteShellBending) - a fold past the yield angle becomes a
    # permanent crease. 0.1 rad of hinge angle on the 7 mm mesh: 0.5 rad let a pressed dog-ear spring back open.
    "paper": dict(E=1e9, nu=0.3, rho=800.0, thickness=1e-4, bending_stiffness=3.0, friction_mu=0.35,
                  bending_yield_strain=0.1, bending_hardening_modulus=0.0),                               # 80 gsm, target Lb 6.8 cm
    "pad":   dict(E=3e6, nu=0.4, rho=467.0, thickness=0.003, bending_stiffness=18.57, friction_mu=0.9),      # rubber base + cloth top, 1.4 kg/m^2, target Lb 6 cm, elastic
    # diagnostic: PAD_T overrides the pad's thickness (contact radius t/2) to test thick-shell friction
}
if os.environ.get("PAD_T"):
    FABRIC["pad"] = {**FABRIC["pad"], "thickness": float(os.environ["PAD_T"])}


def shift(name):
    """per-run layout change for one demo (env DESK_SHIFT_<name>="dx,dy,dz", metres); zero by default."""
    v = os.environ.get(f"DESK_SHIFT_{name}")
    return np.array([float(c) for c in v.split(",")]) if v else np.zeros(3)


def euler(name, default):
    """per-run orientation change for one demo (env DESK_EULER_<name>="rx,ry,rz", degrees); the scene's by default."""
    v = os.environ.get(f"DESK_EULER_{name}")
    return tuple(float(c) for c in v.split(",")) if v else tuple(default)


def fabric(kind):
    """FABRIC[kind] with a demo's own material override (env DESK_FABRIC_<kind>="E=2e5,bending_stiffness=0.08"), read
    when the scene is built."""
    f = dict(FABRIC[kind])
    for kv in filter(None, os.environ.get(f"DESK_FABRIC_{kind}", "").split(",")):
        k, v = kv.split("=")
        f[k.strip()] = float(v)
    return f


def ipc_fabric(kind, **override):
    """FABRIC values are physical. libuipc uses a shell's `thickness` as a contact RADIUS (distance check: shell-rigid
    = t, shell-shell = 2 t), so a 3 mm pad would collide as 6 mm. Pass t/2 and double rho and E: same area density
    (rho t), same membrane stiffness (E t), contact faces where the real faces are. Bending is set separately."""
    f = {**fabric(kind), **override}
    return {**f, "thickness": f["thickness"] / 2, "rho": f["rho"] * 2, "E": f["E"] * 2}


DEFAULT_COUP = {}
FREE = set(os.environ.get("DESK_FREE", "").split(",")) - {""}   # diagnostic: objects to build free instead of fixed   # every desk object is ipc_only; hinges are IPC-native (ipc_joints.py)


def build(dt=0.01, couple=None, skip=(), cameras=("p10",), res=None, extra_cams=None, newton_tol=None, vis=True, extra=None,
          coupler_kw=None, urdf_override=None):
    """coupler_kw overrides the IPC coupler options (the robot runs use the yam-arcade jars-station contact settings);
    urdf_override {name: dict(urdf=path, rho=...)} swaps an object's URDF (robot-run fit geometry, robot_desk.py)."""
    couple = {**DEFAULT_COUP, **(couple or {})}
    ipc_patches.patch_ipc_only_coupling_forces()
    cloth_fix.patch({f["thickness"] / 2: (f["E"] * f["thickness"], f["nu"]) for f in map(fabric, FABRIC)})
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(**{
            **dict(contact_d_hat=DHAT, contact_resistance=1e7, newton_semi_implicit_enable=False, newton_tolerance=newton_tol,
                   enable_rigid_rigid_contact=True), **(coupler_kw or {})}),
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.45, 0.45, 0.45), background_color=(0.86, 0.85, 0.81),
                                          lights=[dict(type="directional", dir=(0.3, 0.6, -1.0), color=(1.0, 0.98, 0.94), intensity=3.0)]),
        show_viewer=False,
    )
    ents = {}
    ents["floor"] = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.6),
                                     surface=gs.surfaces.Default(color=(0.30, 0.30, 0.32)))
    for name, o in SC["objects"].items():
        if name in skip: continue
        coup = couple.get(name, "ipc_only")
        surf = gs.surfaces.Default(color=tuple(o["rgba"])) if "rgba" in o else gs.surfaces.Default()
        ents[name] = scene.add_entity(
            gs.morphs.URDF(file=(urdf_override or {}).get(name, {}).get("urdf", os.path.join(A, o["urdf"])), pos=tuple(np.asarray(o["pos"]) + shift(name)), euler=euler(name, o["euler"]),
                           fixed=bool(o.get("fixed", False)) and name not in FREE, convexify=False, decimate=False, watertighten=0,
                           merge_fixed_links=False),
            material=gs.materials.Rigid(coup_type=coup, coup_friction=o["friction"], rho=(urdf_override or {}).get(name, {}).get("rho", o["rho"])),
            surface=surf)
    for name, d in SC["deformables"].items():
        if name in skip: continue
        ents[name] = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(A, d["mesh"]), pos=tuple(np.asarray(d["pos"]) + shift(name)), euler=tuple(d["euler"])),
            material=gs.materials.FEM.Cloth(**ipc_fabric(d["kind"])), surface=gs.surfaces.Default(color=tuple(d["rgba"])))
    if extra is not None:
        ents.update(extra(scene))
    hinges = {}
    for jn, j in SC.get("joints", {}).items():
        if j["a"] in ents and j["b"] in ents:
            hinges[jn] = ipc_joints.Hinge(jn, ents[j["a"]].base_link, ents[j["b"]].base_link, np.array(j["pivot"]), np.array(j["axis"]),
                                          lo=j["lo"] - j["q0"], hi=j["hi"] - j["q0"], kind=j.get("kind", "revolute"),
                                          drive_strength=float(os.environ.get(f"DESK_DRIVE_{jn}", j.get("drive", 10.0))),
                                          slip=float(os.environ.get(f"DESK_SLIP_{jn}", j.get("slip", 0.02))))
    ipc_joints.install(list(hinges.values()))
    ents["_hinges"] = hinges
    cams = {}
    for c in cameras:
        cc = SC["cameras"][c]
        cams[c] = scene.add_camera(res=tuple(res or cc["res"]), pos=tuple(cc["pos"]), lookat=tuple(cc["lookat"]), up=tuple(cc["up"]),
                                   fov=cc["fov"], GUI=False)
    for k, cc in (extra_cams or {}).items():
        cams[k] = scene.add_camera(GUI=False, **cc)
    return scene, ents, cams


def mass_props(name, col=None, rho=None):
    """(mass, inertia about the centre of mass in link axes) of a rigid object as libuipc integrates it: its collision
    mesh at the object's density (the hand strength limits and force logs in ipc_hands.py need these)."""
    import trimesh
    m = trimesh.load(col or os.path.join(A, "urdf", f"{name}_col.obj"), force="mesh")
    m.density = rho if rho is not None else SC["objects"][name]["rho"]
    return float(m.mass), np.array(m.moment_inertia)


def link_pos(ent, link=None):
    l = ent.get_link(link) if link else ent.base_link
    return np.array(l.get_pos().cpu()), np.array(l.get_quat().cpu())
