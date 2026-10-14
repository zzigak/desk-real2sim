"""IPC-native 'virtual hands': a libuipc SoftTransformConstraint on an 'ipc_only' link whose aim pose comes from a Python
trajectory each frame. The constraint is soft (strength ratios eta_p for the centre-of-mass position, eta_a for
rotation), so contact can still push back; returning None from the trajectory releases the link.

Used instead of Genesis' two_way_soft_constraint for every driven part: that path integrates a second copy of the body
in Genesis with its own collision off, and a driven body in resting contact drifted with it (see ipc_joints.py).

Partial control: a trajectory can copy coordinates of the CURRENT pose into the aim (e.g. the jar lid's height while it
is twisted), which leaves that coordinate to contact alone.

Strength. The constraint energy sits on the kinetic-term scale, so it acts as a spring of k = eta_p * m / dt^2 on the
centre of mass and eta_a * I / dt^2 on rotation (a held part sags g dt^2 / eta: 9.8 um at eta 100, dt 0.01). At the
default eta 100 that is stiff enough to drag a part through any fit. A hand that should behave like a person's is
given a stiffness (K in N/m, K_rot in N m/rad about the link z axis, converted to eta) and a strength (f_max N,
tau_max N m): each step its aim is kept within f_max / K (tau_max / K_rot) of where the part is, so it never pulls
harder than that. While the cap is active the hand acts like a muscle at full effort, whose force drops with speed
(f_max - K dt v). Two things that did not work: capping a stiff (eta 100) spring this way makes it a damper of
k dt and the part barely moves; swapping the spring for libuipc's AffineBodyExternalBodyForce at the cap gives a
4-20 g part a full step of 20 N, so it overshoots and the force flips sign every step.
Every hand with mass properties logs the force and torque it applied over the previous step (F, tau)."""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import uipc
from uipc.constitution import SoftTransformConstraint


@dataclass
class Hand:
    """trajectory(t, T_link) -> aim pose of the LINK frame (URDF origin) or None. libuipc's affine-body transform sits at
    the body's centre of mass, so the hand measures the fixed offset D = T_ipc^-1 T_link on its first frame (T_link0 is
    the link's start pose) and converts both ways; T is always the link-frame pose."""
    name: str
    link: object
    trajectory: Callable
    T_link0: np.ndarray
    eta_p: float = 100.0
    eta_a: float = 100.0
    t: float = 0.0
    T: np.ndarray = field(default_factory=lambda: np.eye(4))
    D: np.ndarray | None = None
    is_active: bool = False
    f_max: float | None = None          # strength limit (N) on the centre-of-mass spring
    tau_max: float | None = None        # strength limit (N m) on the rotation spring
    mass: float | None = None           # mass properties as libuipc integrates them (needed for limits and force logs)
    I_link: np.ndarray | None = None    # inertia about the centre of mass, in link axes
    dt: float = 0.01
    F: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(3))
    K: float | None = None              # translational stiffness (N/m); sets eta_p
    K_rot: float | None = None          # rotational stiffness about the link z axis (N m/rad); sets eta_a
    saturated: tuple = (False, False)   # (force, torque) limit active on the last step
    _last: np.ndarray | None = None     # aim (libuipc frame) sent for the previous step

    def __post_init__(self):
        assert self.mass is not None or (self.f_max is None and self.tau_max is None and self.K is None and self.K_rot is None), \
            "a stiffness or strength limit needs mass properties"

    def resolve(self):
        """turn K / K_rot into strength ratios (needs mass properties and dt)."""
        if self.K is not None:
            self.eta_p = self.K * self.dt ** 2 / self.mass
        if self.K_rot is not None:
            self.eta_a = self.K_rot * self.dt ** 2 / self.I_link[2, 2]

    def animate(self, info):
        geo = info.geo_slots()[0].geometry()
        T_ipc = np.array(uipc.view(geo.transforms())[0])
        if self.D is None:
            self.D = np.linalg.inv(T_ipc) @ self.T_link0
        self.T = T_ipc @ self.D
        aim = self.trajectory(self.t, self.T)
        self.is_active = aim is not None
        uipc.view(geo.instances().find(uipc.builtin.is_constrained))[0] = int(self.is_active)
        if self.mass is None:
            if self.is_active:
                uipc.view(geo.instances().find(uipc.builtin.aim_transform))[0] = aim @ np.linalg.inv(self.D)
            return
        Rc = Rot.from_matrix(T_ipc[:3, :3]).as_matrix(); I_w = Rc @ self.I_link @ Rc.T
        k_p, k_a = self.eta_p * self.mass / self.dt ** 2, self.eta_a / self.dt ** 2
        if self._last is not None:   # force/torque the spring applied at the end of the previous step
            self.F = k_p * (self._last[:3, 3] - T_ipc[:3, 3])
            self.tau = k_a * I_w @ Rot.from_matrix(self._last[:3, :3] @ Rc.T).as_rotvec()
        else:
            self.F, self.tau = np.zeros(3), np.zeros(3)
        self._last = None; sf = sa = False
        if self.is_active:
            A = aim @ np.linalg.inv(self.D)
            if self.f_max is not None:
                d = A[:3, 3] - T_ipc[:3, 3]; n = np.linalg.norm(d)
                if k_p * n > self.f_max:
                    A[:3, 3] = T_ipc[:3, 3] + d * self.f_max / (k_p * n); sf = True
            if self.tau_max is not None:
                rv = Rot.from_matrix(A[:3, :3] @ Rc.T).as_rotvec(); tq = np.linalg.norm(k_a * I_w @ rv)
                if tq > self.tau_max:
                    A[:3, :3] = Rot.from_rotvec(rv * self.tau_max / tq).as_matrix() @ Rc; sa = True
            uipc.view(geo.instances().find(uipc.builtin.aim_transform))[0] = A
            self._last = A
        self.saturated = (sf, sa)


def install(hands):
    """Hook IPCCoupler._finalize_ipc (chained with any earlier hook) so each hand's soft transform constraint and
    animator exist before libuipc initializes the world."""
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler

    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
        stc = SoftTransformConstraint()
        self._ipc_constitution_tabular.insert(stc)
        for h in hands:
            h.resolve()
            stc.apply_to(self._abd_slots_by_link[h.link][0].geometry(), np.array([[h.eta_p], [h.eta_a]]))
            # the coupler names each link's libuipc object rigid_link_<link idx>_<env>; animators are per object
            (obj,) = self._ipc_objects.find(f"rigid_link_{h.link.idx}_0")
            self._ipc_animator.insert(obj, h.animate)
        orig(self)

    IPCCoupler._finalize_ipc = _finalize
