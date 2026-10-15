"""Franka + libuipc helpers the bag scripts share (taken from franka-deformables / zzigak/cloth-genesis, where the same
robot set-up runs the towel and cable tasks).

Genesis 1.3.1, branch ipc_rod_and_mpm of simquarry/genesis-world. Nothing in Genesis or libuipc is edited on disk:
the two patches below replace one method / one class reference at run time, before scene.build().
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bag_carry/
ASSETS = os.path.join(HERE, "assets")

ARM_LINKS = tuple(f"link{i}" for i in range(8))
DOWN_QUAT = {"y": np.array([0.0, 1.0, 0.0, 0.0])}  # wxyz: hand z pointing down, fingers closing along world y
FINGER_OPEN, FINGER_CLOSE = 0.04, -0.05  # far enough past contact that kp 400 saturates at the force cap
FINGER_SPEED = 0.05  # m/s per finger


def panda_mjcf():
    """assets/panda_hand_box.xml with its meshdir pointed at the Franka meshes that ship with the installed Genesis
    (genesis/assets/xml/franka_emika_panda/assets); written next to it as _panda_hand_box.local.xml."""
    import genesis as gs

    src = os.path.join(ASSETS, "panda_hand_box.xml")
    dst = os.path.join(ASSETS, "_panda_hand_box.local.xml")
    meshdir = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "franka_emika_panda", "assets")
    xml = open(src).read().replace('meshdir="FRANKA_MESHDIR"', f'meshdir="{meshdir}"')
    if not os.path.exists(dst) or open(dst).read() != xml:
        tmp = dst + f".{os.getpid()}"
        open(tmp, "w").write(xml)
        os.replace(tmp, dst)
    return dst


def quat_yaw(yaw):
    """Hand pointing down, yawed so the fingers close along (cos(yaw + 90 deg), sin(yaw + 90 deg))."""
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    qz = np.array([c, 0.0, 0.0, s])
    w1, v1 = qz[0], qz[1:]
    w2, v2 = DOWN_QUAT["y"][0], DOWN_QUAT["y"][1:]
    return np.concatenate([[w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)])


def interp(keys, t):
    """keys: (duration, x, y, z, yaw, grip 0/1) from the previous pose. Returns (x, y, z, yaw, grip in [0, 1])."""
    t0, prev = 0.0, keys[0]
    for k in keys:
        if t < t0 + k[0]:
            a = (t - t0) / k[0]
            return tuple((1 - a) * p + a * c for p, c in zip(prev[1:6], k[1:6]))
        t0 += k[0]
        prev = k
    return tuple(keys[-1][1:6])


def ipc_link_centroids(scene, links):
    """Centroid of each rigid link's copy inside libuipc, which is what actually makes contact. With
    two_way_soft_constraint the drawn Genesis link is tied to it by a spring, so the two can differ."""
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot, apply_transform, merge

    want = {link.idx: link for link in links}
    out = {}
    for slot in SceneVisitor(scene.sim.coupler._ipc_scene).geometries():
        if not isinstance(slot, SimplicialComplexSlot):
            continue
        geom = slot.geometry()
        meta = geom.meta()
        st = meta.find("solver_type")
        if st is None or st.view()[0] != "rigid":
            continue
        (idx,) = map(int, meta.find("link_idx").view())
        if idx in want:
            g = merge(apply_transform(geom)) if geom.instances().size() >= 1 else geom
            out[want[idx].name] = g.positions().view().squeeze(-1).mean(0)
    return out


def patch_ipc_only_coupling_forces():
    """Genesis bug workaround (ipc_rod_and_mpm branch, 1.3.1): IPCCoupler._apply_abd_coupling_forces applies the
    two-way soft-constraint reaction to EVERY retrieved affine-body link, including free 'ipc_only' bodies, which have
    no soft constraint at all. Next to a two_way_soft_constraint robot, a free body's drawn Genesis copy got a spurious
    force every step and diverged to NaN within a few steps (libuipc's copy was fine). This re-implements the method
    with those links' forces zeroed; libuipc already sets their pose directly."""
    from genesis.engine.couplers.ipc_coupler import coupler as C

    def _apply(self):
        if (not self.options.two_way_coupling or C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT not in self._entities_by_coup_type
                or not self._abd_data_by_link):
            return
        d = self._coupling_data
        C.update_coupling_forces(d.ipc_transforms, d.aim_transforms, d.links_mass, d.links_inertia_i,
                                 self._constraint_strength_translation_scaled, self._constraint_strength_rotation_scaled,
                                 d.out_forces, d.out_torques)
        if not hasattr(self, "_soft_link_mask"):
            soft = set(self._entities_by_coup_type[C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT])
            self._soft_link_mask = np.array([link.entity in soft for link in d.links])
        d.out_forces[:, ~self._soft_link_mask] = 0.0
        d.out_torques[:, ~self._soft_link_mask] = 0.0
        if np.isnan(d.out_forces).any() or np.isnan(d.out_torques).any():
            raise RuntimeError("NaN coupling forces")
        f, tq = (d.out_forces, d.out_torques) if self.sim.n_envs > 0 else (d.out_forces[0], d.out_torques[0])
        self.rigid_solver.apply_links_external_force(f, links_idx=d.links_idx, local=False)
        self.rigid_solver.apply_links_external_torque(tq, links_idx=d.links_idx, local=False)

    C.IPCCoupler._apply_abd_coupling_forces = _apply


def use_neohookean_shell():
    """Make Genesis's IPC cloth use libuipc's isotropic NeoHookeanShell membrane (one (lambda, mu) pair, scaled by
    thickness) instead of StrainLimitingBaraffWitkinShell. CoDimRecon models plastic film and paper with an isotropic
    St. Venant-Kirchhoff membrane; NeoHookeanShell is the nearest isotropic membrane libuipc has, and the two agree at
    small strain. Swaps the class the coupler instantiates, before scene.build()."""
    from genesis.engine.couplers.ipc_coupler import coupler as C
    from uipc.constitution import NeoHookeanShell

    C.StrainLimitingBaraffWitkinShell = NeoHookeanShell
