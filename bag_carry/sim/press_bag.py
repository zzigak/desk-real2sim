"""Two-hand press on the paper bag (Genesis + libuipc), a behavioural test in the spirit of CoDimRecon App. B.4 (their
cushion press and paper fold): two palm-sized paddles come in from the bag's two ends at mid-height, squeeze it by
`--squeeze` each, hold, back off, and the bag is left alone. Run once with plastic hinges and once with
`--elastic` to see what plasticity does, the paper's controlled rerun ("replaying the same trajectory with plasticity
disabled isolates its effect").

The paddles are free rigid bodies moved kinematically from Genesis (set_pos every step, gravity compensated), their
libuipc copies following through the two-way soft constraint; the bag (and its glued rope handles) is not touched
otherwise. Bag, materials and solver settings as run_bag.py --task paper.

    cd ~/simquarry/genesis-world && .venv/bin/python <repo>/bag_carry/sim/press_bag.py [--elastic]
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

from run_bag import B, META, PAPER, ROPE, S, TABLE_EXT, TOP  # noqa: E402

PADDLE = (0.02, 0.10, 0.12)  # x (thickness) by y (width) by z (height), m: a flat palm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(F.HERE, "out", "press"))
    ap.add_argument("--elastic", action="store_true", help="no plastic hinges (control run)")
    ap.add_argument("--squeeze", type=float, default=0.045, help="each paddle moves this far past first contact, m")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--res", type=int, nargs=2, default=(960, 720))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")
    F.use_neohookean_shell()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.0005, contact_resistance=1e8,
            constraint_strength_translation=1e4, constraint_strength_rotation=1e4,  # paddles track their path tightly
            newton_semi_implicit_enable=False, newton_tolerance=2e-3, newton_max_iterations=100,
            n_linesearch_iterations=24, linear_system_tolerance=1e-6,
            enable_rigid_rigid_contact=False, enable_rigid_ground_contact=True,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.35, 0.35, 0.35)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only"),
                     surface=gs.surfaces.Default(color=(0.52, 0.42, 0.33)))
    cx, cy = 0.0, 0.0
    scene.add_entity(gs.morphs.Mesh(file=os.path.join(S, "table", "table.obj"), pos=(cx, cy, 0.0), fixed=True, collision=False),
                     surface=gs.surfaces.Default())
    scene.add_entity(gs.morphs.Box(pos=(cx, cy, TOP - 0.015), size=(TABLE_EXT[0], TABLE_EXT[1], 0.03), fixed=True,
                                   visualization=False),
                     material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5))
    p = PAPER
    base = np.array([cx, cy, TOP + 0.003])
    bag = scene.add_entity(
        gs.morphs.Mesh(file=os.path.join(B, "rope_bag.obj"), pos=tuple(base)),
        material=gs.materials.FEM.Cloth(E=p["E"], nu=p["nu"], rho=p["rho"], thickness=p["thickness"], bending_stiffness=p["bend"],
                                        bending_yield_strain=None if args.elastic else p["yield_angle"], friction_mu=p["friction"]),
        surface=gs.surfaces.Default(color=(0.66, 0.50, 0.32)))
    ropes_xyz = np.load(os.path.join(B, "rope_handles.npy"), allow_pickle=True)
    rho_rope = ROPE["linear_density"] / (math.pi * ROPE["radius"] ** 2)
    ropes = [scene.add_entity(
        gs.morphs.Polyline(points=[tuple(pt + base) for pt in r], radius=ROPE["radius"]),
        material=gs.materials.FEM.Rod(E=ROPE["E"], bending_stiffness=ROPE["bend"], rho=rho_rope, friction_mu=ROPE["friction"]),
        surface=gs.surfaces.Default(color=(0.55, 0.40, 0.24))) for r in ropes_xyz]
    hb = META["rope_bag"]
    zc = TOP + 0.003 + 0.5 * hb["h"]
    x_start = 0.5 * hb["w"] + 0.5 * PADDLE[0] + 0.012  # 12 mm clear of each end wall
    vol = PADDLE[0] * PADDLE[1] * PADDLE[2]
    paddles = [scene.add_entity(gs.morphs.Box(pos=(cx + s * x_start, cy, zc), size=PADDLE, fixed=False),
                                material=gs.materials.Rigid(coup_type="two_way_soft_constraint", gravity_compensation=1.0,
                                                            rho=2.0 / vol, coup_friction=0.6),
                                surface=gs.surfaces.Default(color=(0.88, 0.74, 0.62)))  # skin-ish
               for s in (-1, 1)]
    _install_rope_glue(ropes, ropes_xyz, bag)  # rope ends glued to the bag, as in run_bag
    cams = [scene.add_camera(res=tuple(args.res), pos=(cx + 0.05, cy - 0.62, TOP + 0.30), lookat=(cx, cy, TOP + 0.11), fov=40,
                             GUI=False, debug=True),
            scene.add_camera(res=(480, 360), pos=(cx + 0.001, cy - 0.001, TOP + 0.75), lookat=(cx, cy, TOP + 0.05), fov=40,
                             GUI=False, debug=True)]
    scene.build()

    m = trimesh.load(os.path.join(B, "rope_bag.obj"), process=False)
    edges = np.unique(np.sort(m.edges, axis=1), axis=0)
    rest = A(bag.get_state().pos).reshape(-1, 3)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    mid = np.abs(rest[:, 2] - zc) < 0.006  # the band of the bag at the paddles' height
    w0 = float(np.ptp(rest[mid, 0]))
    starts = [A(pd.get_pos()).reshape(3) for pd in paddles]

    # schedule (s): settle 1.0, approach to contact 0.8 (12 mm), press 2.0, hold 1.0, back off 1.2 (to start), watch 2.0
    phases = [("settle", 1.0), ("approach", 0.8), ("press", 2.0), ("hold", 1.0), ("release", 1.2), ("watch", 2.0)]
    t_end = np.cumsum([d for _, d in phases])

    def inward(t):
        """How far each paddle has moved in from its start, m."""
        t0 = 0.0
        travel = 0.012 + args.squeeze
        for (name, d), te in zip(phases, t_end):
            if t < te:
                a = (t - t0) / d
                s = 0.5 - 0.5 * math.cos(math.pi * a)  # smooth start and stop
                return {"settle": 0.0, "approach": 0.012 * s, "press": 0.012 + args.squeeze * s, "hold": travel,
                        "release": travel * (1 - s), "watch": 0.0}[name], name
            t0 = te
        return 0.0, "watch"

    frames, log, tubes = [], [], []
    n = int(round(t_end[-1] / args.dt))
    t_wall = time.time()
    for i in range(n):
        t = i * args.dt
        d, ph = inward(t)
        for k, (pd, s0) in enumerate(zip(paddles, starts)):
            sgn = 1.0 if k == 0 else -1.0  # paddle 0 starts at -x and moves +x
            pd.set_pos(s0 + np.array([sgn * d, 0.0, 0.0]), relative=False, zero_velocity=True)
            pd.set_quat(np.array([1.0, 0.0, 0.0, 0.0]), relative=False, zero_velocity=True)  # flat palms stay upright
        scene.step()
        if i % 3 == 0:
            v = A(bag.get_state().pos).reshape(-1, 3)
            ratio = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1) / rest_len
            band = v[np.abs(v[:, 2] - zc) < 0.02]
            log.append(dict(t=t, phase=ph, inward=d, width=float(np.ptp(band[:, 0])), depth=float(np.ptp(band[:, 1])),
                            zmax=float(v[:, 2].max()), zmin=float(v[:, 2].min()), stretch=[float(ratio.min()), float(ratio.max())],
                            paddle_gap=float(np.linalg.norm(A(paddles[0].get_pos()) - A(paddles[1].get_pos()))) - PADDLE[0]))
            for n_ in tubes:
                scene.clear_debug_object(n_)
            tubes = []
            for rope in ropes:
                tv, tf = mu.create_tube(A(rope.get_state().pos).reshape(-1, 3), ROPE["radius"], 12, False)
                tube = trimesh.Trimesh(tv, tf, process=False)
                tube.visual.vertex_colors = (np.array([0.55, 0.40, 0.24, 1.0]) * 255).astype(np.uint8)
                tubes.append(scene.draw_debug_mesh(tube))
            a, b = cams[0].render()[0], cams[1].render()[0]
            pad = np.zeros((a.shape[0] - b.shape[0], b.shape[1], 3), dtype=a.dtype) + 255
            frames.append(np.concatenate([a, np.concatenate([b, pad], 0)], 1))
        if i % 100 == 0:
            print(f"step {i}/{n} t={t:.2f} {ph} inward {d * 1000:.1f} mm  wall {time.time() - t_wall:.0f}s", flush=True)
    wall = time.time() - t_wall
    tag = "elastic" if args.elastic else "plastic"
    imageio.mimwrite(os.path.join(args.out, f"press_{tag}.mp4"), frames, fps=round(1 / (3 * args.dt)), quality=8, macro_block_size=8)
    json.dump(dict(tag=tag, squeeze=args.squeeze, wall_s=wall, width0=w0, log=log), open(os.path.join(args.out, f"press_{tag}.json"), "w"),
              default=float)
    hold = [r for r in log if r["phase"] == "hold"]
    end = log[-1]
    print(f"{tag}: width {w0 * 100:.1f} cm at rest, {min(r['width'] for r in hold) * 100:.1f} cm squeezed, "
          f"{end['width'] * 100:.1f} cm after release ({(w0 - end['width']) * 1000:.0f} mm kept); edge length / rest "
          f"{min(r['stretch'][0] for r in log):.3f}-{max(r['stretch'][1] for r in log):.3f}; wall {wall:.0f} s")


def _install_rope_glue(ropes, xyz, bag):
    """Glue each rope's tucked ends to the bag wall (zero-length libuipc vertex stitches) and switch rope-bag contact
    off, as run_bag.contact_hook does for the carry task."""
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    from uipc.backend import SceneVisitor
    from uipc.constitution import SoftVertexStitch
    from uipc.geometry import SimplicialComplexSlot

    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
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
        for rod, pts in zip(ropes, xyz):
            rslot = slots[fem.index(rod)]
            rv = rslot.geometry().positions().view().squeeze(-1)
            pairs = []
            for i in list(range(3)) + list(range(len(pts) - 3, len(pts))):
                j = int(np.argmin(np.linalg.norm(bv - rv[i], axis=1)))
                assert np.linalg.norm(bv[j] - rv[i]) < 1e-6
                pairs.append([i, j])
            svs = SoftVertexStitch()
            self._ipc_constitution_tabular.insert(svs)
            geo = svs.create_geometry((rslot, bag_slot), np.array(pairs, dtype=np.int32), kappa=1e5)
            self._ipc_objects.create(f"rope_stitch_{fem.index(rod)}").geometries().create(geo)
            self._ipc_contact_tabular.insert(self._ipc_rod_contacts[rod], self._ipc_cloth_contacts[bag], 0.0, 0.0, False)
        orig(self)

    IPCCoupler._finalize_ipc = _finalize


if __name__ == "__main__":
    main()
