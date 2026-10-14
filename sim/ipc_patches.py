"""Genesis IPC coupler workaround, copied from ~/projects/franka-deformables/genesis_cable.py.

IPCCoupler._apply_abd_coupling_forces applies the two-way soft-constraint reaction to EVERY affine-body link, including
free 'ipc_only' bodies that have no soft constraint. With any two_way_soft_constraint entity in the scene (here the
laptop and monitor) every free object's Genesis copy gets a spurious force each step and diverges (seen: laptop stand
1e20 m within 0.1 s). libuipc sets those bodies' poses itself, so their forces are zeroed."""
import numpy as np


def patch_ipc_only_coupling_forces():
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
