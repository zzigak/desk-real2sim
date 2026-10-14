"""IPC-native grasps on shells (cloth, paper, mouse pad): libuipc SoftPositionConstraint on the shell's vertices.

A Grip models a fingertip/suction contact: when its plan first returns a pick point, the vertices within `radius` of it
are captured; while active they are pulled (soft, strength `strength`) to the moving target plus their captured
offsets, so the rest of the sheet hangs, drapes and folds physically. A plan returning None releases them."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

import uipc
from uipc.backend import SceneVisitor
from uipc.constitution import SoftPositionConstraint
from uipc.geometry import SimplicialComplexSlot


@dataclass
class Grip:
    name: str
    entity: object                 # Genesis FEM entity
    plan: Callable                 # (t, V current verts) -> None or (pick point, target point)
    radius: float = 0.012
    t: float = 0.0
    idx: np.ndarray | None = None
    offs: np.ndarray | None = None
    target: np.ndarray | None = None


def install(grips, strength=100.0):
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
        by_entity = {}
        for g in grips: by_entity.setdefault(g.entity, []).append(g)
        if by_entity:
            spc = SoftPositionConstraint(); self._ipc_constitution_tabular.insert(spc)
        for ent, gs_ in by_entity.items():
            i_e = list(self.fem_solver.entities).index(ent)
            (obj,) = self._ipc_objects.find(f"cloth_{i_e}_0")
            for slot in SceneVisitor(self._ipc_scene).geometries():
                if not isinstance(slot, SimplicialComplexSlot): continue
                meta = slot.geometry().meta(); st = meta.find("solver_type")
                if st is not None and st.view()[0] == "cloth" and int(meta.find("entity_idx").view()[0]) == i_e:
                    spc.apply_to(slot.geometry(), strength)
            self._ipc_animator.insert(obj, _animator(gs_))
        orig(self)

    IPCCoupler._finalize_ipc = _finalize


def _animator(grips):
    def animate(info):
        geo = info.geo_slots()[0].geometry()
        V = np.array(geo.positions().view())[..., 0]
        isc = uipc.view(geo.vertices().find(uipc.builtin.is_constrained)); aim = uipc.view(geo.vertices().find(uipc.builtin.aim_position))
        isc[:] = 0
        for g in grips:
            r = g.plan(g.t, V)
            if r is None:
                g.idx = None; g.target = None; continue
            pick, target = np.asarray(r[0], float), np.asarray(r[1], float)
            if g.idx is None:
                g.idx = np.nonzero(np.linalg.norm(V - pick, axis=1) < g.radius)[0]
                g.offs = V[g.idx] - pick
            g.target = target
            isc[g.idx] = 1
            aim[g.idx] = (target + g.offs)[..., None]
    return animate
