"""IPC-native hinges (libuipc AffineBodyRevoluteJoint + AffineBodyDrivingRevoluteJoint) between two Genesis 'ipc_only'
links, added to the libuipc scene right before world.init.

Why not a Genesis articulation: a free articulated entity coupled two-way (soft constraint) went unstable as soon as it
rested on another IPC body (the laptop on its stand drifted ~1 m/s); keeping the whole mechanism inside libuipc avoids
the Genesis/IPC split for these parts.

A Hinge has two modes, chosen per step by its owner:
  - friction (default): the drive's aim angle stays put until the joint is pushed more than `slip` rad away from it,
    then follows at that offset: a stick-slip friction hinge whose holding torque is the drive stiffness times `slip`.
  - track: `target` (rad, relative to the start pose) is the aim angle; demos drive the joint this way.
Angles are relative to the photographed start pose and positive by the right-hand rule about `axis`."""
import os
from dataclasses import dataclass, field

import numpy as np

import uipc
from uipc.constitution import (AffineBodyDrivingPrismaticJoint, AffineBodyDrivingRevoluteJoint, AffineBodyPrismaticJoint,
                               AffineBodyRevoluteJoint)


@dataclass
class Hinge:
    name: str
    link_a: object
    link_b: object
    pivot: np.ndarray
    axis: np.ndarray
    strength: float = 100.0
    drive_strength: float = 10.0          # 100 was stiff enough that closing the lid lifted the laptop base off its stand
    slip: float = 0.02
    lo: float = -np.inf
    hi: float = np.inf
    target: float | None = None
    aim: float = 0.0
    angle: float = 0.0
    kind: str = "revolute"            # or "prismatic": then angle/aim/target are distances along `axis` (m)
    log: list = field(default_factory=list)

    def animate(self, info):
        q, aim_name = ("angle", "aim_angle") if self.kind == "revolute" else ("distance", "aim_distance")
        for slot in info.geo_slots():
            geo = slot.geometry()
            self.angle = float(uipc.view(geo.edges().find(q))[0])
            if self.target is not None:
                self.aim = float(np.clip(self.target, self.lo, self.hi))
            elif abs(self.angle - self.aim) > self.slip:
                self.aim = self.angle - np.sign(self.angle - self.aim) * self.slip
            uipc.view(geo.edges().find(aim_name))[:] = self.aim
            uipc.view(geo.edges().find("driving/is_constrained"))[:] = 1


def install(hinges):
    """Hook IPCCoupler._finalize_ipc so each hinge's joint geometry is created between the two links' affine-body
    geometry slots before libuipc initializes the world."""
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler

    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
        cons = dict(revolute=(AffineBodyRevoluteJoint(), AffineBodyDrivingRevoluteJoint()),
                    prismatic=(AffineBodyPrismaticJoint(), AffineBodyDrivingPrismaticJoint()))
        for c in cons.values():
            for k in c: self._ipc_constitution_tabular.insert(k)
        for h in hinges:
            rev, drv = cons[h.kind]
            slot_a = self._abd_slots_by_link[h.link_a][0]
            slot_b = self._abd_slots_by_link[h.link_b][0]
            p0 = np.asarray(h.pivot, np.float32)[None]; p1 = (np.asarray(h.pivot) + np.asarray(h.axis))[None].astype(np.float32)
            strength = float(os.environ.get("HINGE_STRENGTH", h.strength)); drive = float(os.environ.get("HINGE_DRIVE", h.drive_strength))
            geo = rev.create_geometry(p0.astype(np.float64), p1.astype(np.float64), [slot_a], np.array([0], np.int32), [slot_b], np.array([0], np.int32), np.array([strength]))
            drv.apply_to(geo, drive)
            if os.environ.get("HINGE_PASSIVE"):
                uipc.view(geo.edges().find("is_passive"))[:] = int(os.environ["HINGE_PASSIVE"])
            obj = self._ipc_objects.create(f"hinge_{h.name}")
            obj.geometries().create(geo)
            self._ipc_animator.insert(obj, h.animate)
        orig(self)

    IPCCoupler._finalize_ipc = _finalize
