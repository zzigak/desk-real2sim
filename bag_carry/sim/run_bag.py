"""Bag tasks in Genesis + libuipc, set up following CoDimRecon (arXiv 2609.36024): bags are manifold thin shells
(midsurface + scalar thickness), containers are separate rigid solids, nothing is pinned, the robot is the only
scripted thing, and libuipc runs with the paper's tight Newton tolerance. See ../README.md.

  --task paper  A brown paper bag with a flat bottom and two twisted-paper rope handles stands on table A.
                --load light|heavy: objects drop into it while it settles, then the Franka pinches the front handle,
                carries the bag to table B, stands it back up there and lets go.
                --load robot: the robot first puts a rubber duck and a soda can into the empty bag.
  --task trash  A kitchen bin lined with a plastic bag; a duck, a can and a paper ball fall in; the robot pinches a
                fold of the liner, pulls the bag out over the rim and sets it down (work in progress, see README).

Materials (CoDimRecon Tab. 2 and App. B.4):
  plastic bag: rho 920 kg/m3, E 60 MPa, nu 0.45, h 0.03 mm, mu 0.45
  paper:       rho 761.9 kg/m3 (80 g/m2), E 2.25 GPa, nu 0.15, h 0.105 mm, mu 0.4, plastic hinges (yield 25 /m)
Solver (App. B.4): up to 100 Newton iterations at 2e-3 m/s, PCG tolerance 1e-6, 24 line-search iterations,
contact resistance 1e8, d_hat 0.5 mm, dt 0.01 s, no damping.

    cd ~/simquarry/genesis-world && .venv/bin/python <repo>/bag_carry/sim/run_bag.py --task paper --load heavy \
        --finger-armature 2.0 --eps-v 1e-3 --export exp --out <repo>/bag_carry/out/heavy
"""
import argparse
import json
import math
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import franka_ipc as F  # noqa: E402

import genesis as gs  # noqa: E402
import genesis.utils.mesh as mu  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

from franka_ipc import ARM_LINKS, FINGER_CLOSE, FINGER_OPEN, FINGER_SPEED, interp, ipc_link_centroids, quat_yaw  # noqa: E402

B = os.path.join(F.ASSETS, "bag")
S = F.ASSETS  # table/, duck/
TOP = 0.75
META = json.load(open(os.path.join(B, "meta.json")))
TABLE_EXT = json.load(open(os.path.join(S, "table", "table.json")))["extents"]  # (x, y, z) of the table mesh
# CoDimRecon App. B.4 "Plastic bag": rho 920, E 60 MPa (membrane and bending), nu 0.45, h 0.03 mm, mu 0.45. Its hinge
# rigidity E h^3/12(1-nu^2) = 1.7e-7 N m (bending length 8.6 mm) is below what the 1 cm mesh resolves (~1.4 cm), so
# the bending setting is just small.
PLASTIC = dict(E=60e6, nu=0.45, rho=920.0, thickness=3e-5, friction=0.45, bend=1e-3)
# Tab. 2 / App. B.4 "Paper": rho 761.9 (80 g/m2), E 2.25 GPa (membrane and bending), nu 0.15, h 0.105 mm, mu 0.4,
# plastic hinges with yield curvature 25/m and no hardening. bending_stiffness 1.8 reproduces E h^3/12(1-nu^2) =
# 2.2e-4 N m (bending length 6.6 cm) on this 1 cm mesh (out/bag/calib: 1.2 -> 0.41 cm, 4.8 -> 0.11 cm tip drop vs
# the elastica's 0.28 cm); the yield curvature over 1 cm hinges is a 0.25 rad dihedral change.
PAPER = dict(E=2.25e9, nu=0.15, rho=761.9, thickness=1.05e-4, friction=0.4, bend=1.8, yield_angle=0.25)
# twisted kraft-paper rope handles (not in the paper): 5 mm, ~6 g/m, bending modulus 100 MPa (EI 2.7e-3 N m^2)
ROPE = dict(radius=0.0025, linear_density=0.006, E=1e8, bend=1e8, friction=0.5)
CAN = dict(radius=0.033, height=0.122, mass=0.015, rgb=(0.80, 0.10, 0.12))
BALL = dict(radius=0.03, mass=0.005, rgb=(0.93, 0.93, 0.90))
GRIP = {"bag": 20.0, "rigid": 10.0}  # App. B.4: "independent finger PD controllers are limited to 20 N"


def contact_hook(franka, fixed_off, arm_contact, ropes=None):
    """Before libuipc's world.init: switch off contact between fixed bodies that touch (bin on table: nothing ever has
    to be resolved between two fixed bodies, and libuipc rejects bodies that start in contact), and give the arm links
    their own contact element that collides with everything except the robot itself (see run_sort.contact_hook)."""
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot

    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
        tab = self._ipc_contact_tabular
        if ropes is not None:
            # each rope's tucked ends lie on bag wall vertices: glue them there with zero-length libuipc vertex stitches
            # (like handles glued inside a real bag), and switch rope-bag contact off (the glued part lies on the wall)
            from uipc.constitution import SoftVertexStitch
            rods, xyz, bag, offset = ropes
            fem = list(self.fem_solver.entities)
            slots = {}
            for slot in SceneVisitor(self._ipc_scene).geometries():
                if not isinstance(slot, SimplicialComplexSlot):
                    continue
                meta = slot.geometry().meta()
                st = meta.find("solver_type")
                if st is not None and st.view()[0] in ("rod", "cloth"):
                    slots[int(meta.find("entity_idx").view()[0])] = slot
            bag_slot = slots[fem.index(bag)]
            bv = bag_slot.geometry().positions().view().squeeze(-1)
            for rod, pts in zip(rods, xyz):
                rslot = slots[fem.index(rod)]
                rv = rslot.geometry().positions().view().squeeze(-1)
                n_t = 3
                pairs = []
                for i in list(range(n_t)) + list(range(len(pts) - n_t, len(pts))):
                    j = int(np.argmin(np.linalg.norm(bv - rv[i], axis=1)))
                    assert np.linalg.norm(bv[j] - rv[i]) < 1e-6, "rope end not on a bag vertex"
                    pairs.append([i, j])
                svs = SoftVertexStitch()
                self._ipc_constitution_tabular.insert(svs)
                geo = svs.create_geometry((rslot, bag_slot), np.array(pairs, dtype=np.int32), kappa=1e5)
                self._ipc_objects.create(f"rope_stitch_{fem.index(rod)}").geometries().create(geo)
                tab.insert(self._ipc_rod_contacts[rod], self._ipc_cloth_contacts[bag], 0.0, 0.0, False)
        for a, b in fixed_off:
            tab.insert(self._ipc_abd_contacts[a], self._ipc_abd_contacts[b], 0.0, 0.0, False)
        if arm_contact:
            arm = tab.create("robot_arm")
            arm_idx = {franka.get_link(n).idx for n in ARM_LINKS}
            n = 0
            for slot in SceneVisitor(self._ipc_scene).geometries():
                if not isinstance(slot, SimplicialComplexSlot):
                    continue
                meta = slot.geometry().meta()
                st = meta.find("solver_type")
                if st is not None and st.view()[0] == "rigid" and int(meta.find("link_idx").view()[0]) in arm_idx:
                    arm.apply_to(slot.geometry())
                    n += 1
            assert n == len(ARM_LINKS), f"found {n} arm link geometries, expected {len(ARM_LINKS)}"
            robot = self._ipc_abd_contacts[franka]
            tab.insert(arm, arm, 0.0, 0.0, False)
            tab.insert(arm, robot, 0.0, 0.0, False)
            tab.insert(arm, self._ipc_no_collision_contact, 0.0, 0.0, False)
            others = ([e for ent, e in self._ipc_abd_contacts.items() if ent is not franka]
                      + list(self._ipc_cloth_contacts.values()) + list(self._ipc_fem_contacts.values())
                      + list(self._ipc_ground_contacts.values()))
            for e in others:
                tab.insert(arm, e, 0.5, self.options.contact_resistance, True)
        orig(self)

    IPCCoupler._finalize_ipc = _finalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("trash", "paper"), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=(960, 720))
    # App. B.4: "up to 100 Newton iterations (velocity tolerance 2e-3 m/s), 4,000 PCG iterations (tolerance rate 1e-6),
    # and 24 line-search iterations", contact resistance 1e8; Tab. 2: contact activation distance 0.5 mm for paper
    ap.add_argument("--newton-tol", type=float, default=2e-3)
    ap.add_argument("--newton-iters", type=int, default=100)
    ap.add_argument("--ls-iters", type=int, default=24)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--d-hat", type=float, default=0.0005)
    ap.add_argument("--resistance", type=float, default=1e8)
    ap.add_argument("--coup", choices=("two_way", "external"), default="two_way",
                    help="robot coupling: Genesis two-way soft constraint per link, or joint-level external articulation "
                         "(the robot articulated inside libuipc, as in the paper)")
    ap.add_argument("--finger-armature", type=float, default=0.2)
    ap.add_argument("--load", choices=("robot", "light", "heavy"), default="robot",
                    help="paper task: the robot loads duck + can (robot), or a set of objects is dropped into the bag "
                         "while it settles and the robot only carries it (light: ~0.15 kg, heavy: ~1.2 kg)")
    ap.add_argument("--film-grip", type=float, default=20.0, help="finger force cap on the plastic liner, N (paper: <= 20)")
    ap.add_argument("--eps-v", type=float, default=None, help="libuipc friction smoothing velocity, m/s (default 0.01)")
    ap.add_argument("--settle-only", action="store_true")
    ap.add_argument("--export", default=None, help="also write every logged frame for the Blender renderer "
                    "(recorder.Recorder format, as in zzigak/yam-arcade: robot link poses, bag surface, rope centrelines, object poses) "
                    "to this directory (relative: under --out)")
    ap.add_argument("--ramp", type=float, default=0.05)
    args = ap.parse_args()
    out = args.out or os.path.join(F.HERE, "out", args.task)
    os.makedirs(out, exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")
    F.patch_ipc_only_coupling_forces()  # free ipc_only bodies next to a two-way coupled robot
    F.use_neohookean_shell()  # isotropic membrane, the nearest libuipc has to the paper's StVK
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),  # libuipc does all contact
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=args.d_hat, contact_resistance=args.resistance,
            constraint_strength_translation=100.0, constraint_strength_rotation=1000.0,
            newton_semi_implicit_enable=False, newton_tolerance=args.newton_tol, newton_max_iterations=args.newton_iters,
            n_linesearch_iterations=args.ls_iters, linear_system_tolerance=args.pcg_tol, contact_eps_velocity=args.eps_v,
            enable_rigid_rigid_contact=True, enable_rigid_ground_contact=True,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.35, 0.35, 0.35)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only"),
                     surface=gs.surfaces.Default(color=(0.52, 0.42, 0.33)))
    fixed_off = []

    def add_table(cx, cy, yaw_deg):
        scene.add_entity(gs.morphs.Mesh(file=os.path.join(S, "table", "table.obj"), pos=(cx, cy, 0.0), euler=(0, 0, yaw_deg),
                                        fixed=True, collision=False), surface=gs.surfaces.Default())
        ex, ey = (TABLE_EXT[0], TABLE_EXT[1]) if yaw_deg == 0 else (TABLE_EXT[1], TABLE_EXT[0])
        return scene.add_entity(gs.morphs.Box(pos=(cx, cy, TOP - 0.015), size=(ex, ey, 0.03), fixed=True, visualization=False),
                                material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5))

    scene.add_entity(gs.morphs.Box(pos=(-0.02, 0.0, TOP / 2), size=(0.26, 0.26, TOP), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.25, 0.26, 0.28)))
    franka = scene.add_entity(
        gs.morphs.MJCF(file=F.panda_mjcf(), pos=(0.0, 0.0, TOP)),
        material=(gs.materials.Rigid(coup_type="two_way_soft_constraint", coup_links=ARM_LINKS + ("hand", "left_finger", "right_finger"),
                                     coup_friction=1.0, gravity_compensation=1.0) if args.coup == "two_way"
                  else gs.materials.Rigid(coup_type="external_articulation", coup_friction=1.0, gravity_compensation=1.0)),
    )
    duck_col = trimesh.load(os.path.join(S, "duck", "duck_col.obj"), process=False)

    def add_duck(pos):
        return scene.add_entity(
            gs.morphs.URDF(file=os.path.join(S, "duck", "duck.urdf"), pos=pos, fixed=False, convexify=False, decimate=False,
                           watertighten=0, align=False),
            material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.8, rho=0.03 / duck_col.volume),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=os.path.join(S, "duck", "duck_diff.jpg"))))

    def add_can(pos, euler=(0, 0, 0)):
        vol = math.pi * CAN["radius"] ** 2 * CAN["height"]
        return scene.add_entity(gs.morphs.Cylinder(pos=pos, euler=euler, radius=CAN["radius"], height=CAN["height"]),
                                material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5, rho=CAN["mass"] / vol),
                                surface=gs.surfaces.Default(color=CAN["rgb"]))

    objs, shell = {}, None
    if args.task == "trash":
        table = add_table(0.51, 0.0, 0)
        bx, by = 0.50, 0.0
        bin_ = scene.add_entity(gs.morphs.Mesh(file=os.path.join(B, "basket.obj"), pos=(bx, by, TOP), fixed=True,
                                               convexify=False, decimate=False),
                                material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5),
                                surface=gs.surfaces.Default(color=(0.42, 0.26, 0.14)))
        fixed_off.append((bin_, table))
        p = PLASTIC
        # loose liner folded over the rim (make_bag_assets.liner), as in CoDimRecon Fig. 1
        shell = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(B, "liner.obj"), pos=(bx, by, TOP)),
            material=gs.materials.FEM.Cloth(E=p["E"], nu=p["nu"], rho=p["rho"], thickness=p["thickness"],
                                            bending_stiffness=p["bend"], friction_mu=p["friction"]),
            surface=gs.surfaces.Default(color=(0.90, 0.90, 0.88)))
        # dropped from above the opening at staggered heights, a few cm apart
        objs["duck"] = add_duck((bx + 0.01, by + 0.03, TOP + 0.33))
        objs["can"] = add_can((bx - 0.02, by - 0.04, TOP + 0.45), euler=(90, 0, 20))
        vb = 4 / 3 * math.pi * BALL["radius"] ** 3
        objs["ball"] = scene.add_entity(gs.morphs.Sphere(pos=(bx + 0.03, by - 0.01, TOP + 0.55), radius=BALL["radius"]),
                                        material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.6, rho=BALL["mass"] / vb),
                                        surface=gs.surfaces.Default(color=BALL["rgb"]))
        cam_pos, cam_look = (1.05, -0.62, 1.32), (0.46, 0.0, 0.92)
        home = (0.30, 0.30, TOP + 0.45)
    else:
        ta = add_table(0.70, 0.36, 90)
        tb = add_table(0.70, -0.36, 90)
        p = PAPER
        bag_c = (0.50, 0.30)
        shell = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(B, "rope_bag.obj"), pos=(*bag_c, TOP + 0.003)),
            material=gs.materials.FEM.Cloth(E=p["E"], nu=p["nu"], rho=p["rho"], thickness=p["thickness"],
                                            bending_stiffness=p["bend"], bending_yield_strain=p["yield_angle"],
                                            friction_mu=p["friction"]),
            surface=gs.surfaces.Default(color=(0.66, 0.50, 0.32)))
        ropes_xyz = np.load(os.path.join(B, "rope_handles.npy"), allow_pickle=True)
        rho_rope = ROPE["linear_density"] / (math.pi * ROPE["radius"] ** 2)
        ropes = [scene.add_entity(
            gs.morphs.Polyline(points=[tuple(pt + np.array([*bag_c, TOP + 0.003])) for pt in r], radius=ROPE["radius"]),
            material=gs.materials.FEM.Rod(E=ROPE["E"], bending_stiffness=ROPE["bend"], rho=rho_rope, friction_mu=ROPE["friction"]),
            surface=gs.surfaces.Default(color=(0.55, 0.40, 0.24))) for r in ropes_xyz]
        if args.load == "robot":
            objs["duck"] = add_duck((0.38, 0.13, TOP + 0.003))
            objs["can"] = add_can((0.64, 0.14, TOP + 0.003 + CAN["height"] / 2))
        else:
            # dropped into the bag from above its opening while everything settles, staggered so nothing starts in
            # contact; all within +-4 cm of the bag's centre line, clear of the rope handles' planes (+-6 cm)
            rim = TOP + 0.003 + META["rope_bag"]["h"]
            bx, by = bag_c
            heavy = args.load == "heavy"

            def prim(name, morph, mass, vol, rgb, mu=0.5):
                objs[name] = scene.add_entity(morph, material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=mu, rho=mass / vol),
                                              surface=gs.surfaces.Default(color=rgb))

            r, hh = (0.04, 0.10) if heavy else (0.033, 0.122)
            prim("jar" if heavy else "can", gs.morphs.Cylinder(pos=(bx - 0.045, by, rim + 0.10), radius=r, height=hh),
                 0.45 if heavy else 0.015, math.pi * r * r * hh, (0.70, 0.82, 0.86) if heavy else CAN["rgb"])
            prim("full_can" if heavy else "ball", gs.morphs.Cylinder(pos=(bx + 0.045, by, rim + 0.12), radius=0.033, height=0.122)
                 if heavy else gs.morphs.Sphere(pos=(bx + 0.045, by, rim + 0.08), radius=0.03),
                 0.38 if heavy else 0.005, math.pi * 0.033 ** 2 * 0.122 if heavy else 4 / 3 * math.pi * 0.03 ** 3,
                 CAN["rgb"] if heavy else BALL["rgb"])
            if heavy:
                prim("orange", gs.morphs.Sphere(pos=(bx, by, rim + 0.30), radius=0.04), 0.20, 4 / 3 * math.pi * 0.04 ** 3, (0.95, 0.50, 0.08))
            prim("tea_box", gs.morphs.Box(pos=(bx - 0.035, by, rim + (0.45 if heavy else 0.30)), size=(0.08, 0.05, 0.11)),
                 0.15 if heavy else 0.06, 0.08 * 0.05 * 0.11, (0.20, 0.45, 0.30))
            objs["duck"] = add_duck((bx + 0.04, by, rim + (0.60 if heavy else 0.45)))
        cam_pos, cam_look = (1.55, -0.05, 1.35), (0.55, 0.0, 0.85)
        home = (0.45, 0.0, TOP + 0.40)
    contact_hook(franka, fixed_off, arm_contact=True,
                 ropes=(ropes, ropes_xyz, shell, np.array([*bag_c, TOP + 0.003])) if args.task == "paper" else None)
    # debug=True so the rope tubes (debug meshes, see below) are rendered
    cam = scene.add_camera(res=tuple(args.res), pos=cam_pos, lookat=cam_look, fov=45, GUI=False, debug=True)
    scene.build()

    hand = franka.get_link("hand")
    fl = [franka.get_link(n) for n in ("left_finger", "right_finger")]
    motors, fingers = slice(0, 7), slice(7, 9)
    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 400.0, 400.0])
    franka.set_dofs_kv([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 5.0, 5.0])
    franka.set_dofs_armature([args.finger_armature] * 2, fingers)
    # Seed the IK on the standard elbow-up branch, base turned toward the target. Unseeded, its random restarts picked
    # a folded-over branch (base -2.06 rad, joint 3 at its limit) that later ran the base joint into its -2.9 rad limit
    # and flipped the hand over (run 4: 96-117 deg).
    seed = np.array([math.atan2(home[1], home[0]), -0.3, 0.0, -2.3, 0.0, 2.0, 0.785, FINGER_OPEN, FINGER_OPEN])
    q = A(franka.inverse_kinematics(link=hand, pos=np.array(home), quat=quat_yaw(0.0), init_qpos=seed, max_samples=1,
                                    max_solver_iters=200))
    q[fingers] = FINGER_OPEN
    franka.set_qpos(q)
    pad_off = float(A(hand.get_pos())[2]) - min(float(A(l.get_AABB())[0, 2]) for l in fl)
    q = A(franka.inverse_kinematics(link=hand, pos=np.array([home[0], home[1], home[2] + pad_off]), quat=quat_yaw(0.0),
                                    init_qpos=q, max_samples=1, max_solver_iters=200))
    print("home q", np.round(q[:7], 2), flush=True)
    q[fingers] = FINGER_OPEN
    franka.set_qpos(q)
    franka.control_dofs_position(q)

    bag_mesh = trimesh.load(os.path.join(B, "liner.obj" if args.task == "trash" else "rope_bag.obj"), process=False)
    edges = np.unique(np.sort(bag_mesh.edges, axis=1), axis=0)
    e_all, cnt = np.unique(np.sort(bag_mesh.edges, axis=1), axis=0, return_counts=True)
    rim = np.unique(e_all[cnt == 1])
    rest = A(shell.get_state().pos).reshape(-1, 3)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    bottom_idx = np.where(rest[:, 2] < rest[:, 2].min() + 1e-4)[0]

    exp = make_export(args, out, scene, franka, objs, shell, bag_mesh, ropes if args.task == "paper" else [],
                      fps=1.0 / (max(1, int(round(1.0 / (args.fps * args.dt)))) * args.dt)) if args.export else None

    def verts(name):
        if name == "bag":
            return A(shell.get_state().pos).reshape(-1, 3)
        e = objs[name]
        if name == "duck":
            pos, quat = A(e.get_pos()).reshape(3), A(e.get_quat()).reshape(4)
            R = trimesh.transformations.quaternion_matrix(quat)[:3, :3]
            return np.array(duck_col.vertices) @ R.T + pos
        return A(e.get_verts()).reshape(-1, 3)

    every = max(1, int(round(1.0 / (args.fps * args.dt))))
    frames, log = [], []
    st = dict(step=0, finger_target=FINGER_OPEN, in_contact=False, q=q, cur=(*home, 0.0, 0.0))
    t_wall = time.time()

    def run(keys, label, grip_cap):
        franka.set_dofs_force_range([-grip_cap] * 2, [20.0] * 2, fingers)
        n = int(round(sum(k[0] for k in keys) / args.dt))
        for i in range(n):
            t = i * args.dt
            x, y, z, yaw, g = interp(keys, t)
            qq, err = franka.inverse_kinematics(link=hand, pos=np.array([x, y, z + pad_off]), quat=quat_yaw(yaw),
                                                init_qpos=st["q"], max_samples=1, max_solver_iters=100, return_error=True)
            # an unreachable step keeps the last good joint target instead of jumping to whatever the solver returns
            if float(np.abs(A(err)).max()) < 0.02:
                st["q"] = A(qq)
            else:
                st["ik_fail"] = st.get("ik_fail", 0) + 1
            franka.control_dofs_position(st["q"][motors], motors)
            closing = g > 0.0 and interp(keys, t + args.dt)[4] >= g
            qf = float(A(franka.get_dofs_position(fingers)).mean())
            if closing or g > 0.99:
                if not st["in_contact"] and qf - st["finger_target"] > 0.002 and st["finger_target"] < FINGER_OPEN - 0.005:
                    st["in_contact"] = True
                st["finger_target"] = max(FINGER_CLOSE, st["finger_target"] - (args.ramp if st["in_contact"] else FINGER_SPEED) * args.dt)
            else:
                st["in_contact"] = False
                st["finger_target"] = min(FINGER_OPEN, max(st["finger_target"], qf) + FINGER_SPEED * args.dt)
            franka.control_dofs_position(np.full(2, st["finger_target"]), fingers)
            scene.step()
            if st["step"] % every == 0:
                rec = dict(t=st["step"] * args.dt, phase=label, target=[x, y, z, yaw, g])
                rec["finger_force"] = [float(f) for f in A(franka.get_dofs_control_force(fingers))]
                lp, rp = (A(l.get_pos()) for l in fl)
                rec["finger_gap_drawn"] = float(np.linalg.norm(lp - rp))
                ipc = ipc_link_centroids(scene, fl)
                rec["finger_gap_ipc"] = float(np.linalg.norm(ipc["left_finger"] - ipc["right_finger"]))
                rec["robot_zmin_above_table"] = min(float(A(l.get_AABB())[0, 2]) for l in [hand] + fl) - TOP
                hz = trimesh.transformations.quaternion_matrix(A(hand.get_quat()).reshape(4))[:3, 2]
                rec["hand_tilt_deg"] = float(np.degrees(np.arccos(np.clip(-hz[2], -1.0, 1.0))))
                for name in ["bag"] + list(objs):
                    v = verts(name)
                    r = dict(com=v.mean(0).tolist(), zmin=float(v[:, 2].min()), zmax=float(v[:, 2].max()))
                    if name == "bag":
                        ratio = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1) / rest_len
                        r["stretch"] = [float(ratio.min()), float(ratio.max())]
                        r["rim_zmax"] = float(v[rim, 2].max())
                        # upright check: bottom-panel centre -> rim centre should stay near vertical on the table
                        r["bottom_c"], r["rim_c"] = v[bottom_idx].mean(0).tolist(), v[rim].mean(0).tolist()
                        r["bbox"] = [v.min(0).tolist(), v.max(0).tolist()]
                        # App. B.4 settling check: largest vertex displacement since the last logged frame, and speed
                        if st.get("prev_bag") is not None:
                            dv = np.linalg.norm(v - st["prev_bag"][0], axis=1)
                            dmax = float(dv.max())
                            rec["bag_max_disp"] = dmax
                            rec["bag_max_vidx"] = int(dv.argmax())
                            rec["bag_max_speed"] = dmax / max(rec["t"] - st["prev_bag"][1], 1e-9)
                        st["prev_bag"] = (v.copy(), rec["t"])
                    rec[name] = r
                log.append(rec)
                if args.task == "paper":
                    # Genesis draws a FEM.Rod as a zero-radius line; draw each rope as a tube through its centreline
                    for n_ in st.setdefault("tubes", []):
                        scene.clear_debug_object(n_)
                    st["tubes"] = []
                    for rope in ropes:
                        cp = A(rope.get_state().pos).reshape(-1, 3)
                        tv, tf = mu.create_tube(cp, ROPE["radius"], 12, False)
                        tube = trimesh.Trimesh(tv, tf, process=False)
                        tube.visual.vertex_colors = (np.array([0.55, 0.40, 0.24, 1.0]) * 255).astype(np.uint8)
                        st["tubes"].append(scene.draw_debug_mesh(tube))
                    rec["ropes_zmax"] = [float(A(r.get_state().pos).reshape(-1, 3)[:, 2].max()) for r in ropes]
                frames.append(cam.render()[0])
                if exp is not None:
                    exp.frame(rec["t"])
            st["step"] += 1
            if st["step"] % 200 == 0:
                print(f"step {st['step']}  t={st['step'] * args.dt:.2f}s  {label}  wall {time.time() - t_wall:.0f}s", flush=True)
        st["cur"] = tuple(keys[-1][1:5]) + (0.0,)

    def pick_place(grasp, yaw, lift_z, place_xy, place_z, label, cap):
        x0, y0, z0, yaw0, _ = st["cur"]
        keys = [(0.01, x0, y0, z0, yaw0, 0.0),
                (1.4, grasp[0], grasp[1], max(z0, grasp[2] + 0.12), yaw, 0.0),
                (1.2, grasp[0], grasp[1], grasp[2] + 0.12, yaw, 0.0),
                (1.2, *grasp, yaw, 0.0),
                (1.6, *grasp, yaw, 1.0),
                (1.6, grasp[0], grasp[1], lift_z, yaw, 1.0),
                (2.2, place_xy[0], place_xy[1], lift_z, yaw, 1.0),
                (1.4, place_xy[0], place_xy[1], place_z, yaw, 1.0),
                (0.8, place_xy[0], place_xy[1], place_z, yaw, 0.0),
                (1.0, place_xy[0], place_xy[1], place_z + 0.15, yaw, 0.0)]
        run(keys, label, cap)

    def carry_bag(grasp, lift_z, place_xy, touch_z, cap):
        """Carry by the rope handle, set down by feel, let go without snagging the handle.

        Run carryx_heavy lowered to a fixed height 12 mm above the expected touch-down: the heavier, tilted bag landed
        early, the hand kept pushing, the slack rope rode up over the right fingertip pad and the rising open hand
        hooked it and pulled the bag back up. Run carryy lowered until the bottom panel lay flat: that pushed even
        further (both loads hooked again; the heavy bag, hanging ~50 deg off one handle, fell on its side and the tilted
        handle loop ended up around one open finger). Run carryz let go at first contact + 3 cm with the bag still 22 deg
        over on one bottom edge; both bags toppled. Now: fast to 4 cm above the expected touch-down, 2 cm/s down to
        first contact, then walk the hand back to the pose it grasped from (above the handle wall's bottom edge, at the
        grasp height) until the bag stands within 6 deg of upright; wait, open, centre on the rope, slide 7 cm along
        it (the loop is 8 cm wide, the open fingers straddle it) to leave the loop sideways, and only then rise."""
        x0, y0, z0, yaw0, _ = st["cur"]
        px, py = place_xy
        run([(0.01, x0, y0, z0, yaw0, 0.0),
             (1.4, grasp[0], grasp[1], max(z0, grasp[2] + 0.12), 0.0, 0.0),
             (1.2, grasp[0], grasp[1], grasp[2] + 0.12, 0.0, 0.0),
             (1.2, *grasp, 0.0, 0.0),
             (1.6, *grasp, 0.0, 1.0),
             (1.6, grasp[0], grasp[1], lift_z, 0.0, 1.0),
             (2.2, px, py, lift_z, 0.0, 1.0),
             (1.2, px, py, touch_z + 0.04, 0.0, 1.0)], "bag", cap)
        x, y, z = px, py, touch_z + 0.04
        # 1) straight down at 2 cm/s until the bag first touches the table (a real arm: the wrist load drops)
        first = None
        while z > grasp[2] - 0.03:
            if verts("bag")[:, 2].min() <= TOP + 0.0015:
                first = z
                break
            run([(0.01, x, y, z, 0.0, 1.0), (0.1, x, y, z - 0.002, 0.0, 1.0)], "bag", cap)
            z -= 0.002
        # 2) stand it up: carried by one handle the bag hangs 20-50 deg over and lands on one bottom edge. Upright, the
        # handle's apex stands above the bottom edge of its own wall, at the height it was grasped at on table A; walk
        # the hand there (<= 4 mm and 2 mm per 0.1 s, re-measured every step) and only let go once the bag is upright.
        wall_y = rest[:, 1].min() if np.asarray(ropes_xyz[0].tolist(), float)[:, 1].mean() < 0 else rest[:, 1].max()
        w_edge = bottom_idx[np.abs(rest[bottom_idx, 1] - wall_y) < 1e-4]
        tilt, moved, it = None, 0.0, 0
        for it in range(120):
            v = verts("bag")
            a = v[rim].mean(0) - v[bottom_idx].mean(0)
            tilt = float(np.degrees(np.arccos(a[2] / np.linalg.norm(a))))
            dxy = v[w_edge].mean(0)[:2] - np.array([x, y])
            dz = grasp[2] - z
            if np.linalg.norm(dxy) < 0.005 and abs(dz) < 0.002 and tilt < 6.0:
                break
            sxy = dxy * min(1.0, 0.004 / max(np.linalg.norm(dxy), 1e-9))
            sz = float(np.clip(dz, -0.002, 0.002))
            run([(0.01, x, y, z, 0.0, 1.0), (0.1, x + sxy[0], y + sxy[1], z + sz, 0.0, 1.0)], "bag", cap)
            x, y, z = x + sxy[0], y + sxy[1], z + sz
            moved += float(np.linalg.norm(sxy))
        st["touch_z"], st["first_contact_z"], st["righting_move"] = z, first, moved
        st["release_tilt_deg"], st["righting_steps"] = tilt, it
        print(f"set-down: first contact {first}, righting {it} steps, moved {moved:.3f} m, tilt at release {tilt:.1f} deg",
              flush=True)
        run([(0.01, x, y, z, 0.0, 1.0), (0.3, x, y, z, 0.0, 1.0), (0.8, x, y, z, 0.0, 0.0)], "bag", cap)
        # centre the open hand on the rope: the rope points between the fingertips, in the hand's closing axis
        R = trimesh.transformations.quaternion_matrix(A(hand.get_quat()).reshape(4))[:3, :3]
        hp = A(hand.get_pos()).reshape(3)
        rp = np.concatenate([(A(r.get_state().pos).reshape(-1, 3) - hp) @ R for r in ropes])
        near = rp[(np.abs(rp[:, 0]) < 0.03) & (np.abs(rp[:, 1]) < 0.07) & (rp[:, 2] > 0.04) & (rp[:, 2] < 0.13)]
        shift = R @ np.array([0.0, near[:, 1].mean(), 0.0]) if len(near) else np.zeros(3)
        side = R @ np.array([-0.07, 0.0, 0.0])  # along the rope (hand x), toward the robot
        st["release_shift"] = shift.tolist()
        print("release: rope offset in the hand", np.round(near[:, 1].mean(), 4) if len(near) else None, flush=True)
        xs, ys = x + shift[0], y + shift[1]
        run([(0.01, x, y, z, 0.0, 0.0), (0.6, xs, ys, z, 0.0, 0.0), (0.9, xs + side[0], ys + side[1], z, 0.0, 0.0),
             (1.0, xs + side[0], ys + side[1], z + 0.15, 0.0, 0.0)], "bag", cap)

    def rigid_grasp(name, pad_z_above_table=0.010):
        """Top-down across the narrow axis. The palm sits 5.6 cm above the pad bottoms, so the pads go no lower than
        4.8 cm under the object's top (run 2's duck and can grasps stopped on the palm and closed on nothing)."""
        v = verts(name)
        pad_z_above_table = max(pad_z_above_table, v[:, 2].max() - 0.048 - TOP)
        xy = v[:, :2] - v[:, :2].mean(0)
        w, U = np.linalg.eigh(xy.T @ xy)
        minor = U[:, 0]
        yaw = (math.atan2(minor[1], minor[0]) - math.pi / 2 + math.pi / 2) % math.pi - math.pi / 2
        return (*v[:, :2].mean(0), TOP + pad_z_above_table), yaw, v

    # settle until at rest (App. B.4 check: drift < 1 mm, speed < 2 mm/s); the liner swings for a while after the drops
    # loaded bags: the front handle's stitched ends sway for a while after the drops (undamped, 4 mm/s at 3 s)
    run([((1.5 if args.load == "robot" else 5.0) if args.task == "paper" else 5.0, *home, 0.0, 0.0)], "settle", GRIP["rigid"])
    plans = []
    if not args.settle_only and args.task == "trash":
        # As in CoDimRecon Fig. 1: reach into the basket and pinch the liner's slack. The liner hangs in 7 inward pleats;
        # take the pleat facing the robot, 4 cm below the rim (the palm stays above the rim and the cuff), fingers
        # closing across the pleat (tangentially), pads 4 mm out from its tip (as for a cloth edge; 15 mm out, run 3
        # squeezed the fold's root flat and caught nothing).
        v = verts("bag")
        H = META["basket"]["height"]
        zg = TOP + H - 0.04
        band = v[np.abs(v[:, 2] - zg) < 0.006]
        rel = band[:, :2] - np.array([0.50, 0.0])
        r = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        # the liner's fold facing the robot (make_bag_assets.LINER: a 4 cm deep fin at -x); its tip is the innermost
        # band vertex within 0.3 rad of -x
        cand = np.where(np.abs(np.angle(np.exp(1j * (ang - np.pi)))) < 0.3)[0]
        k = cand[np.argmin(r[cand])]
        radial = rel[k] / r[k]
        tang = np.array([-radial[1], radial[0]])
        # pads (1.7 cm across radially) centred 12 mm out from the tip: the fin's two layers are ~7 mm apart there
        g = (0.50 + radial[0] * (r[k] + 0.012), radial[1] * (r[k] + 0.012), zg)
        yaw = (math.atan2(tang[1], tang[0]) - math.pi / 2 + math.pi / 2) % math.pi - math.pi / 2
        plans.append(dict(object="bag", grasp=list(map(float, g)), yaw=float(yaw), crest_r=float(r[k])))
        print("plan", plans[-1], flush=True)
        pick_place(g, yaw, TOP + 0.70, (0.42, -0.34), TOP + 0.28, "bag", args.film_grip)
    elif not args.settle_only:
        bag_c = np.array([0.50, 0.30])
        for name in (("duck", "can") if args.load == "robot" else ()):
            g, yaw, v = rigid_grasp(name, 0.035 if name == "can" else 0.010)
            # carry with the object's bottom 4 cm above the handle bars (9 cm over the rim), lower it between the two
            # handles and let go 4 cm above the rim
            bottom_below_pads = g[2] - v[:, 2].min()
            rim_z = TOP + PAPER_H
            lift = rim_z + bottom_below_pads + 0.13
            plans.append(dict(object=name, grasp=list(map(float, g)), yaw=float(yaw)))
            print("plan", plans[-1], flush=True)
            pick_place(g, yaw, lift, bag_c, rim_z + bottom_below_pads + 0.04, name, GRIP["rigid"])
        # carry the bag by its front rope handle: pinch the rope's apex, fingers closing across the handle's plane
        rv = A(ropes[0].get_state().pos).reshape(-1, 3)
        apex = rv[np.argmax(rv[:, 2])]
        v = verts("bag")
        g = (apex[0], apex[1], apex[2] - 0.009)  # pads 1.7 cm tall, centred on the rope
        plans.append(dict(object="bag", grasp=list(map(float, g)), yaw=0.0, apex=list(map(float, apex))))
        print("plan", plans[-1], flush=True)
        dz = g[2] - v[:, 2].min()
        carry_bag(g, TOP + 0.50, (0.50, -0.30), TOP + dz, GRIP["bag"])
    wall = time.time() - t_wall
    imageio.mimwrite(os.path.join(out, f"{args.task}.mp4"), frames, fps=round(1.0 / (every * args.dt)), quality=8, macro_block_size=8)
    json.dump(dict(task=args.task, wall_s=wall, pad_offset=pad_off, plans=plans, ik_fail_steps=st.get("ik_fail", 0),
                   bag_end_verts=np.round(verts("bag"), 5).tolist(),
                   touch_z=st.get("touch_z"), first_contact_z=st.get("first_contact_z"), righting_move=st.get("righting_move"),
                   release_tilt_deg=st.get("release_tilt_deg"), righting_steps=st.get("righting_steps"),
                   release_shift=st.get("release_shift"),
                   grip_caps={"bag": args.film_grip if args.task == "trash" else GRIP["bag"], "duck": GRIP["rigid"], "can": GRIP["rigid"]},
                   load=args.load, masses={k: float(A(e.get_mass())) if hasattr(e, "get_mass") else None for k, e in objs.items()}, log=log), open(os.path.join(out, f"{args.task}.json"), "w"), default=float)
    if exp is not None:
        exp.close()
    print(f"wrote {out}  wall {wall:.0f}s")


PAPER_H = META["rope_bag"]["h"]


class make_export:
    """Per-frame export for render/render_bag.py (Blender), in the YAM-arcade Recorder format: one mesh per robot link
    and colour (in the link frame) plus the link's pose each frame, each free object's visual mesh plus its pose, the
    bag's simulated surface every frame, the ropes' simulated centrelines. Nothing here feeds back into the sim."""

    def __init__(self, args, out, scene, franka, objs, shell, bag_mesh, ropes, fps):
        from recorder import Recorder
        d = args.export if os.path.isabs(args.export) else os.path.join(out, args.export)
        self.rec = Recorder(d, station=f"bag_{args.task}_{args.load}", fps=fps)
        src = os.path.join(d, "_src")
        os.makedirs(src, exist_ok=True)
        self.shell, self.faces, self.ropes = shell, np.asarray(bag_mesh.faces, np.int32), ropes
        self.links = {}  # export name -> link

        def to_link(link, vg):
            lp, R = A(link.get_pos()).reshape(3), trimesh.transformations.quaternion_matrix(A(link.get_quat()).reshape(4))[:3, :3]
            return (A(vg.get_vverts()).reshape(-1, 3) - lp) @ R, np.asarray(vg.init_vfaces)

        colours = {}
        for link in franka.links:
            groups = {}
            for vg in link.vgeoms:
                c = tuple(np.round(np.asarray(vg.surface.diffuse_texture.color, float)[:3], 3))
                groups.setdefault(c, []).append(to_link(link, vg))
            for k, (c, parts) in enumerate(groups.items()):
                V = np.concatenate([p[0] for p in parts])
                off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
                F = np.concatenate([p[1] + o for p, o in zip(parts, off)])
                f = os.path.join(src, f"franka_{link.name}_{k}.obj")
                trimesh.Trimesh(V, F, process=False).export(f)
                name = f"robot/{link.name}#{k}"
                # menagerie panda colours: white / off-white shells, dark grey joints and flanges
                mat = (dict(kind="paint", color=(0.93, 0.93, 0.92), roughness=0.32) if min(c) > 0.85 else
                       dict(kind="plastic", color=(0.07, 0.07, 0.075), roughness=0.42) if max(c) < 0.3 else
                       dict(kind="emissive", color=c, strength=2.0))  # link6's pilot lights (light blue, green)
                self.rec.add_rigid(name, f, mat)
                self.links[name] = link
                colours[name] = c
        for oname, e in objs.items():
            link = e.links[0]
            for k, vg in enumerate(link.vgeoms):
                V, F = to_link(link, vg)
                f = os.path.join(src, f"{oname}_{k}.obj")
                tm = trimesh.Trimesh(V, F, process=False)
                tex = getattr(vg.surface, "diffuse_texture", None)
                if vg.uvs is not None and getattr(tex, "image_path", None):
                    from PIL import Image
                    uv = np.asarray(vg.uvs).copy()
                    uv[:, 1] = 1.0 - uv[:, 1]  # Genesis stores v flipped (OpenGL image rows); OBJ wants it back
                    tm.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(tex.image_path))
                tm.export(f)
                name = oname if k == 0 else f"{oname}#{k}"
                self.rec.add_rigid(name, f, dict(kind="rubber") if oname == "duck" else dict(kind="plastic"))
                self.links[name] = link
        self.rec.add_meshseq("bag", dict(kind="paper", color=(0.66, 0.50, 0.32)))
        for i in range(len(ropes)):
            self.rec.add_rod(f"rope_{i}", ROPE["radius"], dict(kind="paper", color=(0.55, 0.40, 0.24)))
        self.rest = A(shell.get_state().pos).reshape(-1, 3)
        np.save(os.path.join(d, "bag_rest.npy"), self.rest.astype(np.float32))
        self.extra = dict(top=TOP, task=args.task, load=args.load, robot_colours={k: list(v) for k, v in colours.items()},
                          tables=[dict(c=[0.70, 0.36], size=[TABLE_EXT[1], TABLE_EXT[0]]), dict(c=[0.70, -0.36], size=[TABLE_EXT[1], TABLE_EXT[0]])]
                          if args.task == "paper" else [dict(c=[0.51, 0.0], size=[TABLE_EXT[0], TABLE_EXT[1]])],
                          pedestal=dict(c=[-0.02, 0.0, TOP / 2], size=[0.26, 0.26, TOP]),
                          object_masses={k: float(A(e.get_mass())) if hasattr(e, "get_mass") else None for k, e in objs.items()})

    def frame(self, t):
        rigid = {n: (A(l.get_pos()).reshape(3), A(l.get_quat()).reshape(4)) for n, l in self.links.items()}
        rods = {f"rope_{i}": A(r.get_state().pos).reshape(-1, 3) for i, r in enumerate(self.ropes)}
        self.rec.frame(t, rigid=rigid, rods=rods, meshes={"bag": (A(self.shell.get_state().pos).reshape(-1, 3), self.faces)})

    def close(self):
        self.rec.close(self.extra)
        import shutil
        shutil.rmtree(os.path.join(self.rec.out_dir, "_src"), ignore_errors=True)  # the Recorder copied them to meshes/

if __name__ == "__main__":
    main()
