"""The desk with a wheeled bimanual I2RT YAM cart (~/projects/yam-arcade/robots/yam_cart.py) parked at its front edge,
for robot-hand versions of the fit demos (robot_demos.py).

Contact settings are the yam-arcade jars station's robot run (stations/jars/robot_demo.py ROBOT_COUPLER, NOTES.md):
d_hat 0.2 mm, contact resistance 1e9, friction smoothing and Newton tolerance 1e-3 m/s, dt 1/150, robot coupling
translation 10 / rotation 40 with the fingertip spring measured at the pad centroid. At this d_hat the fits use
build_assets.ROBOT_FITS: every clearance is real (> d_hat); only the short retention bands sit inside it.

Only joint PD targets (IK + min-jerk joint paths), the base PD target and gripper commands drive anything; the desk
objects are free libuipc bodies. The arm links themselves are not collision geometry (only the four fingertip pads
and the two palms are coupled to libuipc), so paths are planned to keep the arms clear of the desk objects.
"""
import math, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh

YA = os.environ.get("YAM_ARCADE", os.path.expanduser("~/projects/yam-arcade"))
for p in (YA, os.path.join(YA, "robots")):
    if p not in sys.path:
        sys.path.insert(0, p)

import scene as S
import build_assets as BA
from objwriter import mesh_inertial

VARIANT = "yam"
DT = 1.0 / 150.0
CONTACT = dict(contact_d_hat=2e-4, contact_resistance=1e9, contact_eps_velocity=1e-3, newton_tolerance=1e-3,
               constraint_strength_translation=10.0, constraint_strength_rotation=40.0, two_way_coupling=True)
# the yam-arcade cart is built for a 0.75 m table with the arm bases 4.5 cm behind its near edge and the base 0.27 m
# behind it; the desk top is 0.775 m, so the cart stands 2.5 cm higher (its base joints are virtual: no wheel contact)
RU = os.path.join(S.A, "urdf_robot")


def _near_edge_y():
    import json
    L = json.load(open(os.path.join(S.ROOT, "layout.json")))
    return -0.5 * L["table"]["size"][1]


def cart_pos(x):
    return (float(x), _near_edge_y() - 0.27, S.H - 0.75)


# ------------------------------------------------------------------------------------------------ fit assets
def robot_assets():
    """Collision (+ simple visual) meshes and URDFs for the fit parts at the robot runs' d_hat, in assets/urdf_robot.
    Returns the scene.build urdf_override {name: dict(urdf, rho)}."""
    import hashlib, json
    test = json.loads(os.environ.get("ROBOT_FITS_JSON", "{}"))   # diagnostic overrides, e.g. {"marker": {"seat_gap": 0.00025}}
    ru = RU if not test else os.path.join(RU, "var_" + hashlib.md5(json.dumps(test, sort_keys=True).encode()).hexdigest()[:8])
    if os.environ.get("SLURM_JOB_ID"):                   # on sc several runs start together on a network filesystem: private copies
        import socket
        ru = os.path.join(ru, f"run_{socket.gethostname().split('.')[0]}_{os.getpid()}")
    os.makedirs(ru, exist_ok=True)
    parts = {}
    for key, fn in (("jar", BA.make_jar), ("tea_tin", BA.make_tin), ("marker", BA.make_marker)):
        m, sp = fn(**{**BA.ROBOT_FITS[key], **test.get(key, {})})
        parts.update(m)
    vis = {n: os.path.relpath(os.path.join(S.A, "urdf", f"{n}_vis.ply"), ru) for n in ("tin", "tin_lid")}
    out = {}

    def atomic(path, write):                             # several runs start at once on sc and share these files
        tmp = f"{path}.{os.getpid()}.tmp{os.path.splitext(path)[1]}"
        write(tmp); os.replace(tmp, path)
    for name, m in parts.items():
        o = S.SC["objects"][name]
        atomic(os.path.join(ru, f"{name}_col.obj"), m.export)
        if name not in vis:
            v = m.copy(); rgba = o.get("rgba", (0.10, 0.10, 0.11, 1.0))
            c = np.tile((np.array(rgba) * 255).astype(np.uint8), (len(v.vertices), 1))
            if name == "jar_lid":   # the pale wedge that makes the turning lid visible (build_scene_assets.proc_obj)
                ang = np.degrees(np.arctan2(v.vertices[:, 1], v.vertices[:, 0])); rad = np.linalg.norm(v.vertices[:, :2], axis=1)
                c[(np.abs(ang) < 12) & (rad > 0.6 * rad.max()), :3] = (215, 212, 200)
            v.visual.vertex_colors = c; atomic(os.path.join(ru, f"{name}_vis.ply"), v.export)
        vfile = vis.get(name, f"{name}_vis.ply")
        text = (f'<robot name="{name}"><link name="{name}">{mesh_inertial(m, o["mass"])}'
                f'<visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{vfile}"/></geometry></visual>'
                f'<collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{name}_col.obj"/></geometry></collision>'
                f'</link></robot>')
        atomic(os.path.join(ru, f"{name}.urdf"), lambda f: open(f, "w").write(text))
        out[name] = dict(urdf=os.path.join(ru, f"{name}.urdf"), rho=float(o["mass"] / m.volume))
    return out


# ------------------------------------------------------------------------------------------------ coupling patches
_PATCH = {}


def patch_coupling():
    """yam-arcade stations/jars/station.patch_coupling_at_geometry_centroid (copied): Genesis computes the two-way
    soft-constraint reaction on a coupled robot link at the link ORIGIN, libuipc's constraint pulls at the IPC
    geometry's centroid; on the YAM fingertips the origin is 70-90 mm from the pad and a grasp on a short part
    diverged. Spring measured and applied at the centroid; non-robot links get no reaction. Also stores the
    reaction on every robot link in FORCES (N, on the Genesis copy; the hand pushes on the world with minus that)."""
    from genesis.engine.couplers.ipc_coupler import coupler as C
    if _PATCH:
        C.IPCCoupler._apply_abd_coupling_forces = _PATCH["fn"]
        return
    cache = {}

    def _apply(self):
        if (not self.options.two_way_coupling or C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT not in self._entities_by_coup_type
                or not self._abd_data_by_link):
            return
        d = self._coupling_data
        if id(self) not in cache:
            soft = set(self._entities_by_coup_type[C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT])
            mask = np.array([link.entity in soft for link in d.links])
            cents = np.zeros((len(d.links), 3))
            for i, link in enumerate(d.links):
                if mask[i]:
                    geo = self._abd_slots_by_link[link][0].geometry()
                    cents[i] = np.asarray(geo.positions().view()).reshape(-1, 3).mean(0)
            cache[id(self)] = (mask, cents, [l.name for l in d.links])
        mask, cents, names = cache[id(self)]
        C.update_coupling_forces(d.ipc_transforms, d.aim_transforms, d.links_mass, d.links_inertia_i,
                                 self._constraint_strength_translation_scaled, self._constraint_strength_rotation_scaled,
                                 d.out_forces, d.out_torques)
        Ti, Ta = d.ipc_transforms, d.aim_transforms
        c_ipc = np.einsum("blij,lj->bli", Ti[..., :3, :3], cents) + Ti[..., :3, 3]
        c_aim = np.einsum("blij,lj->bli", Ta[..., :3, :3], cents) + Ta[..., :3, 3]
        F = self._constraint_strength_translation_scaled * d.links_mass[None, :, None] * (c_ipc - c_aim)
        d.out_torques[:] = d.out_torques + np.cross(c_aim - Ta[..., :3, 3], F)
        d.out_forces[:] = F
        d.out_forces[:, ~mask] = 0.0
        d.out_torques[:, ~mask] = 0.0
        if np.isnan(d.out_forces).any() or np.isnan(d.out_torques).any():
            raise RuntimeError("NaN IPC coupling forces")
        for i in np.nonzero(mask)[0]:
            FORCES[names[i]] = np.array(d.out_forces[0, i], float)
        f, tq = (d.out_forces, d.out_torques) if self.sim.n_envs > 0 else (d.out_forces[0], d.out_torques[0])
        self.rigid_solver.apply_links_external_force(f, links_idx=d.links_idx, local=False)
        self.rigid_solver.apply_links_external_torque(tq, links_idx=d.links_idx, local=False)

    _PATCH["fn"] = _apply
    C.IPCCoupler._apply_abd_coupling_forces = _apply


FORCES = {}

# affine-body stiffness (MPa) of the fitted parts. Genesis gives every rigid link 100 MPa, which lets a 10 N gripper
# squeeze a marker cap uniformly enough to close its click band onto the barrel and lock the fit (in-scene the cap then
# held 12.6 N even at a band gap equal to d_hat). Plastic is 1.5-3 GPa, glass and steel far stiffer.
KAPPA_MPA = dict(marker=1e4, marker_cap=1e4, tin=1e4, tin_lid=1e4, jar=1e4, jar_lid=1e4)


def install_kappa(links_kappa):
    """set libuipc's per-instance affine stiffness `kappa` on the given links before the world initialises."""
    import uipc
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
        for link, k in links_kappa.items():
            for slot in self._abd_slots_by_link[link]:
                uipc.view(slot.geometry().instances().find("kappa"))[:] = k * 1e6
        orig(self)

    IPCCoupler._finalize_ipc = _finalize


def hand_force(side):
    """force (N, world) that one hand's pads and palm push on the world with, from the coupling springs."""
    return -sum((FORCES.get(f"{side}_{n}", np.zeros(3)) for n in ("tip_left", "tip_right", "gripper")), np.zeros(3))


# ------------------------------------------------------------------------------------------------ environment
def min_jerk(n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Impedance:
    """Cartesian impedance on one arm's tool centre point (TCP), the usual way a robot pulls or inserts a fitted part:
    joint torques tau = J^T [K (x* - x) - D v ; Kr rot(R* R^T) - Dr w], with stiffness and damping per axis of `frame`
    (columns = world directions of the task axes), and the force and torque magnitudes capped at f_max / tau_max (the
    arm's usable strength). The cart's gravity compensation still applies. Move the target with .pos / .R."""

    def __init__(self, d, side, k=(1500.0, 1500.0, 1500.0), kr=(3.0, 3.0, 3.0), frame=None, f_max=20.0, tau_max=1.5,
                 m_eff=0.5, i_eff=3e-3, zeta=0.8):
        self.d, self.side = d, side
        self.k, self.kr = np.asarray(k, float), np.asarray(kr, float)
        self.c, self.cr = 2 * zeta * np.sqrt(self.k * m_eff), 2 * zeta * np.sqrt(self.kr * i_eff)
        self.frame = np.eye(3) if frame is None else np.asarray(frame, float)
        self.f_max, self.tau_max = f_max, tau_max
        self.pos, self.R = self.tcp()
        self.F, self.M = np.zeros(3), np.zeros(3)

    def tcp(self):
        from yam_cart import mat_from_quat
        p, q = self.d.cart.tcp_pose(self.side)
        return p, mat_from_quat(q)

    def update(self):
        from genesis.utils.misc import tensor_to_array
        cart = self.d.cart; e = cart.entity; dofs = cart.arm_dofs[self.side]
        x, R = self.tcp()
        J = tensor_to_array(e.get_jacobian(cart.ee_link[self.side], local_point=cart.tcp_offset))[:, dofs]
        qd = tensor_to_array(e.get_dofs_velocity(dofs))
        v, w = J[:3] @ qd, J[3:] @ qd
        Fl = self.frame.T
        F = self.frame @ (self.k * (Fl @ (self.pos - x)) - self.c * (Fl @ v))
        M = self.frame @ (self.kr * (Fl @ Rot.from_matrix(self.R @ R.T).as_rotvec()) - self.cr * (Fl @ w))
        if np.linalg.norm(F) > self.f_max: F *= self.f_max / np.linalg.norm(F)
        if np.linalg.norm(M) > self.tau_max: M *= self.tau_max / np.linalg.norm(M)
        self.F, self.M = F, M
        e.control_dofs_force(J[:3].T @ F + J[3:].T @ M - 0.3 * qd, dofs)


class Desk:
    """scene + cart + pose readout + recording + the motion primitives of yam-arcade's jars Demo."""

    def __init__(self, keep=(), cart_x=0.0, cams=None, res=(960, 720), dt=DT, fps=30, nyx=None, lift=0.0, back=0.0, drop=()):
        """lift: the cart's lift carriage (deck + both arms) lowered by this much (m) from arm bases level with the
        tabletop; back: the cart parked this much further from the desk's front edge (m)."""
        import genesis as gs
        import yam_cart
        import robot_plan as RP
        from yam_cart import add_cart
        self.gs, self.dt = gs, dt
        self.lift, self.back = float(lift), float(back)
        self.cart_xml = RP.cart_xml(self.lift)
        if self.lift:                                     # the lowered model and its export meshes live in this repo
            yam_cart.ASSETS = os.path.dirname(self.cart_xml)
        skip = tuple(k for k in S.SC["deformables"] if k not in keep) + tuple(drop)   # drop: rigid objects this demo removes
        # diagnostic: DESK_ONLY=a,b,... keeps only these rigid objects (plus the table)
        only = set(os.environ.get("DESK_ONLY", "").split(",")) - {""}
        if only:
            skip += tuple(o for o in S.SC["objects"] if o != "table" and o not in only)
        self.cart_x0 = cart_x
        holder = {}

        def extra(scene):
            holder["cart"] = add_cart(scene, gs, VARIANT, ipc=True, pos=np.asarray(cart_pos(cart_x)) - (0.0, self.back, 0.0))
            patch_coupling()
            out = {}
            if nyx:
                from nyx_cam import add_nyx
                out["_nyx"] = add_nyx(scene, **nyx)
            return out

        self._override = robot_assets()
        self.scene, self.E, self.C = S.build(dt=dt, skip=skip, cameras=(), extra_cams=cams or {}, res=res, extra=extra,
                                             coupler_kw=CONTACT, urdf_override=self._override)
        self.cart = holder["cart"]
        install_kappa({self.E[n].base_link: k for n, k in KAPPA_MPA.items() if n in self.E})
        self.scene.build()
        self.cart.after_build()
        self.t, self.k, self.fps = 0.0, 0, fps
        self.every = max(1, int(round(1.0 / (fps * dt))))
        self.base = np.zeros(3)
        self.on_step = []
        self.imp = {}                                     # side -> active Impedance controller
        self.log, self.events = [], []
        self._D = {}
        for n in S.SC["objects"]:   # libuipc body frames vs URDF link frames, measured before the first step
            if n in self.E and self.scene.sim.coupler._abd_data_by_link and self.E[n].base_link in self.scene.sim.coupler._abd_data_by_link:
                self._frame(n)
        # collision-free planning (robot_plan.py) on the same MJCF; the ready pose is set relative to the arm bases
        self.P = RP.Planner(self.cart.base_pos, self.cart_xml, objects=[n for n in S.SC["objects"] if n in self.E])
        self.P.home = self._ready_pose()
        self.cart.home_qpos = self.P.home.copy()
        self.cart.set_qpos(self.P.home)

    def _ready_pose(self):
        """both TCPs 20 cm above their arm bases, 17.5 cm in front of them and 4.5 cm inboard, pointing down (the
        yam-arcade ready pose, measured from the arm bases), collision-free."""
        from yam_cart import down_quat
        q = self.cart.home_qpos.copy()
        self._sync()
        for side, sx in (("left", -1.0), ("right", 1.0)):
            b = self.cart.base_pos + (sx * 0.305 - sx * 0.045, 0.225 + 0.175, 0.0)
            for z in (S.H + 0.20, S.H - self.lift + 0.20):      # 20 cm above the desk, or above the lowered arm bases
                try:
                    q[self.P.arm[side]] = self.P.ik(side, (b[0], b[1], z), down_quat(0.0), q, restarts=40); break
                except RuntimeError:
                    if z == S.H - self.lift + 0.20:
                        raise
        return q

    def _sync(self):
        """desk object poses into the planner; an object within 6 cm of a TCP whose fingers are not wide open is in
        that hand (or about to be), and its own contact with the hand is libuipc's, so it is left out."""
        poses = {n: self.pose_any(n) for n in self.P.col.objects}
        held = set()
        for side in ("left", "right"):
            if self.cart.gripper_width(side) < 0.09:
                p = self.tcp(side)
                held |= {n for n, T in poses.items() if np.linalg.norm(T[:3, 3] - p) < 0.06}
        self.P.col.set_objects(poses, ignore=held)
        from genesis.utils.misc import tensor_to_array
        self.P.col.sheets = {n: np.asarray(tensor_to_array(self.E[n].get_state().pos)[0], float)
                             for n in S.SC["deformables"] if self.E.get(n) is not None}
        # a hand holds a sheet only when closed on it (a few mm); an opened hand next to a sheet it just let go of must not
        # drag through it
        self.P.col.sheet_held = {s for s in ("left", "right") if self.cart.gripper_width(s) < 0.015}
        return held

    def q_ref(self, base_x=None, assume=None):
        """full config for planning: the arm and base PD targets, the fingers where they are, and an arm under
        Cartesian impedance where it is (its joint target is stale while the impedance drives it)."""
        q = self.cart.q_target.copy(); qa = self.cart.qpos()
        fd = np.concatenate([self.cart.finger_dofs[s] for s in ("left", "right")])
        q[fd] = qa[fd]
        for s in self.imp:
            q[self.cart.arm_dofs[s]] = qa[self.cart.arm_dofs[s]]
        for s, q6 in (assume or {}).items():              # plan against an arm pose that is yet to come
            q[self.cart.arm_dofs[s]] = q6
        if base_x is not None:
            q[self.cart.base_dofs[0]] = base_x
        return q

    # -- state
    def _T_ipc(self, name):
        co = self.scene.sim.coupler
        return np.asarray(co._abd_data_by_link[self.E[name].base_link][0].transform, float)

    def _frame(self, name):
        """D = T_ipc^-1 T_link at build time (the libuipc body frame is not the URDF link frame)."""
        if name not in self._D:
            o = S.SC["objects"][name]
            T_link = np.eye(4); T_link[:3, :3] = Rot.from_euler("xyz", S.euler(name, o["euler"]), degrees=True).as_matrix()
            T_link[:3, 3] = np.asarray(o["pos"]) + S.shift(name)          # a demo's layout shift (scene.shift)
            self._D[name] = np.linalg.inv(self._T_ipc(name)) @ T_link
        return self._D[name]

    def pose(self, name):
        """URDF link pose of a desk object from libuipc's state (Genesis's own copy of an ipc_only body lags by g dt^2)."""
        T = self._T_ipc(name) @ self._frame(name)
        U, _, Vt = np.linalg.svd(T[:3, :3]); T[:3, :3] = U @ Vt
        return T

    # -- stepping
    def step(self, n=1):
        for _ in range(n):
            self.cart.update()
            for imp in self.imp.values():
                imp.update()
            self.scene.step()
            self.t += self.dt; self.k += 1
            for f in self.on_step:
                f(self)

    def wait(self, dur):
        self.step(max(1, int(round(dur / self.dt))))

    def note(self, msg):
        self.events.append(dict(t=round(self.t, 3), msg=msg)); print(f"[{self.t:7.3f}] {msg}", flush=True)

    # -- motion (yam-arcade stations/jars/robot_demo.Demo)
    def ik(self, side, pos, yaw=0.0, tilt=0.0, quat=None, base_x=None, tol=2e-3, rtol=0.03, seed=None, samples=24, assume=None):
        """collision-free IK (robot_plan.Planner.ik) with the rest of the robot at its current targets (or `assume`)."""
        from yam_cart import down_quat
        self._sync()
        q = self.P.ik(side, pos, down_quat(yaw, tilt) if quat is None else quat, self.q_ref(base_x, assume), seed=seed, tol=tol,
                      rtol=rtol, restarts=samples)
        return q

    def ik_all(self, side, pos, yaw=0.0, tilt=0.0, quat=None, base_x=None, tol=2e-3, rtol=0.03):
        """distinct collision-free arm configurations for one TCP pose, nearest first (robot_plan.Planner.ik_all)."""
        from yam_cart import down_quat
        self._sync()
        return self.P.ik_all(side, pos, down_quat(yaw, tilt) if quat is None else quat, self.q_ref(base_x), tol=tol, rtol=rtol)

    def grasp_yaw(self, side, pos, yaw, tilt=0.0, base_x=None):
        """a parallel gripper is symmetric under a half turn about its approach axis: of yaw and yaw +- pi, the one the
        wrist reaches (joint 6 is limited to +-2.09 rad)."""
        for y in sorted((yaw, yaw - math.pi, yaw + math.pi), key=abs):
            try:
                self.ik(side, pos, yaw=y, tilt=tilt, base_x=base_x); return y
            except RuntimeError:
                pass
        raise RuntimeError(f"no reachable yaw for {side} at {np.round(pos, 4)}")

    def config_free(self, arms, base_x=None):
        """raise RuntimeError if the robot with `arms` ({side: q6}) at these joints (the rest at its targets) collides."""
        self._sync()
        q = self.q_ref(base_x)
        for side, qa in arms.items():
            q[self.cart.arm_dofs[side]] = qa
        if not self.P.free(q, tuple(arms) or ("left", "right")):
            raise RuntimeError(f"pose of {tuple(arms)} collides: {self.P.blame(q, tuple(arms))[:3]}")

    # every arm motion below runs through _run: joint waypoints per arm, checked together (both arms and the base
    # moving at once) before anything moves
    def _full(self, legs, k_frac, base0, base1):
        q = self.q_ref()
        for side, qs in legs.items():
            x_ = k_frac * (len(qs) - 1); i = min(int(x_), len(qs) - 2)
            q[self.cart.arm_dofs[side]] = qs[i] + (x_ - i) * (np.asarray(qs[i + 1]) - qs[i])
        q[self.cart.base_dofs[0]] = base0 + k_frac * (base1 - base0)
        return q

    def check_legs(self, legs, base_x=None):
        """raise if moving the arms through their waypoints together (and the base to base_x) collides anywhere."""
        self._sync()
        b0 = self.base[0]; b1 = b0 if base_x is None else base_x
        n = max(2, int(math.ceil(max([np.abs(np.diff(np.asarray(qs), axis=0)).sum(0).max() for qs in legs.values()] + [abs(b1 - b0) * 10])
                                 / 0.03)))
        sides = tuple(legs) or ("left", "right")
        allow = self.P.viol(self._full(legs, 0.0, b0, b1), sides) or None    # contact the start already has may stay
        for u in np.linspace(0, 1, n + 1)[1:]:
            q = self._full(legs, u, b0, b1)
            if not self.P.free(q, sides, allow):
                raise RuntimeError(f"path of {sides} collides at {u:.2f}: {self.P.blame(q, sides)[:3]}")

    def _run(self, legs, dur, base_x=None, check=True):
        if check:
            self.check_legs(legs, base_x)
        n = max(1, int(round(dur / self.dt))); s = min_jerk(n); b0 = self.base[0]; b1 = b0 if base_x is None else base_x
        for k in range(n):
            for side, qs in legs.items():
                x_ = s[k] * (len(qs) - 1); i = min(int(x_), len(qs) - 2)
                self.cart.set_arm_target(side, qs[i] + (x_ - i) * (np.asarray(qs[i + 1]) - qs[i]))
            if base_x is not None:
                self.cart.set_base_target(b0 + s[k] * (b1 - b0), 0.0, 0.0)
            self.step(1)
        if base_x is not None:
            self.base[0] = b1

    def go(self, arms, dur, base_x=None):
        """free joint-space move of `arms` ({side: q6}) and the base, routed around collisions (robot_plan.route); a
        detour gets the time its length needs at <= 1.2 rad/s."""
        self._sync()
        qa = self.q_ref(); qb = qa.copy()
        for side, q in arms.items():
            qb[self.cart.arm_dofs[side]] = q
        if base_x is not None:
            qb[self.cart.base_dofs[0]] = base_x
        sides = tuple(arms) or ("left", "right")
        path = self.P.route(qa, qb, sides)
        if len(path) > 2:
            self.note(f"go {sides}: detour through {len(path) - 2} via poses")
        legs = {side: [q[self.cart.arm_dofs[side]] for q in path] for side in arms}
        L = max([np.abs(np.diff(np.asarray(qs), axis=0)).sum(0).max() for qs in legs.values()] + [0.0])
        if len(path) > 2:
            dur = max(dur, L / 1.2)
        if not legs:
            legs = {}
        self._run(legs, dur, base_x, check=False)

    def go_paths(self, moves, dur, n_via=8, base_x=None):
        """Cartesian straight-line moves {side: ((pos, yaw, tilt), (pos, yaw, tilt))}, collision-free IK at n_via
        waypoints."""
        from yam_cart import down_quat
        legs = {}
        for side, (a, b) in moves.items():
            legs[side] = self.plan_pose(side, a[0], down_quat(a[1], a[2]), b[0], down_quat(b[1], b[2]), n_via=n_via,
                                        tol=4e-3, rtol=0.06, base_x=base_x)
        self._run(legs, dur, base_x)

    def plan_pose(self, side, p0, q0, p1, q1, n_via=12, tol=6e-3, rtol=0.12, seed=None, base_x=None, assume=None, clear=None):
        """joint waypoints for a straight-line TCP move with the flange orientation slerped from quat q0 to q1 (wxyz);
        collision-free (with the arms in `assume` at those joints instead of their targets, kept `clear` m apart if
        given: an assumed pose is only approximately where that arm will be); raises RuntimeError (before anything
        moves) if a via point is out of reach or collides."""
        self._sync()
        c0 = self.P.clear
        if clear is not None:
            self.P.clear = clear
        try:
            return self.P.chain(side, self.q_ref(base_x, assume), p0, q0, p1, q1, n_via=n_via, seed=seed, tol=tol, rtol=rtol)
        finally:
            self.P.clear = c0

    def plan_to(self, side, targets, quats, base_x=None, tol=4e-3, rtol=0.06):
        """a collision-free joint-space route (not a straight TCP line) for one arm to the first reachable target in
        `targets` with any flange orientation in `quats` (both in preference order). -> (route [q6, ...], target, quat)"""
        self._sync()
        qa = self.q_ref(base_x); errs = []
        for tgt in targets:
            for qw in quats:
                try:
                    q6 = self.P.ik(side, tgt, qw, qa, tol=tol, rtol=rtol)
                    qb = qa.copy(); qb[self.cart.arm_dofs[side]] = q6
                    path = self.P.route(qa, qb, (side,))
                except RuntimeError as e:
                    errs.append(str(e)); continue
                return [q[self.cart.arm_dofs[side]] for q in path], np.asarray(tgt), qw
        raise RuntimeError(f"{side}: no route to any of {len(targets)} targets; e.g. " + " || ".join(e[:100] for e in errs[:3]))

    def follow(self, side, qs, dur):
        """move one arm through joint waypoints (min-jerk in the path parameter)."""
        self.follow_many({side: qs}, dur)

    def follow_many(self, legs, dur):
        """move several arms through their joint waypoints together ({side: [q, ...]}, min-jerk in the path parameter)."""
        self._run(legs, dur)

    def plan_first(self, side, p0, q0, targets, alts=(), n_via=10):
        """joint waypoints from (p0, q0) to the first reachable target in `targets` (ordered by preference), keeping
        q0 or turning to one of `alts` on the way. -> (waypoints, target, quat). Raises if none is reachable."""
        errs = []
        for tgt in targets:
            for qa in (q0, *alts):
                try:
                    return self.plan_pose(side, p0, q0, tgt, qa, n_via=n_via), np.asarray(tgt), qa
                except RuntimeError as e:
                    errs.append(str(e))
        raise RuntimeError(f"{side}: none of {len(targets)} targets reachable from {np.round(p0, 3)}; e.g. " +
                           " || ".join(e[:100] for e in errs[:3]))

    def plan_path(self, side, pts, quat, alts=()):
        """legs of joint waypoints through TCP points `pts`, keeping flange orientation `quat`; a leg out of reach
        is retried turning to each orientation in `alts` (slerped along the leg, kept after). Raises if none works."""
        legs, q_cur, seed = [], quat, None
        for k in range(len(pts) - 1):
            for qa in (q_cur, *alts):
                try:
                    legs.append(self.plan_pose(side, pts[k], q_cur, pts[k + 1], qa, n_via=8, seed=seed)); q_cur = qa; break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(f"{side}: path leg {k} from {np.round(pts[k], 3)} to {np.round(pts[k + 1], 3)} out of reach")
            seed = legs[-1][-1]
        return legs

    def move_pose(self, side, p0, q0, p1, q1, dur, n_via=12, tol=6e-3, rtol=0.12):
        """straight-line TCP move with the flange orientation slerped from quat q0 to q1 (wxyz), IK at n_via points."""
        self._run({side: self.plan_pose(side, p0, q0, p1, q1, n_via, tol, rtol)}, dur)

    def retreat(self, side, moves, yaw, tilt, dur=0.8):
        """after opening, back the hand off along the first reachable displacement in `moves` (world, m)."""
        from yam_cart import down_quat
        p = self.tcp(side)
        for mv in moves:
            try:
                leg = self.plan_pose(side, p, down_quat(yaw, tilt), p + np.asarray(mv), down_quat(yaw, tilt), n_via=6)
            except RuntimeError:
                continue
            self._run({side: leg}, dur)
            return mv
        raise RuntimeError(f"{side} hand found no way back from {np.round(p, 3)}")

    def grip_until(self, side, timeout):
        for _ in range(int(round(timeout / self.dt))):
            self.step(1)
            if self.cart.grasped(side):
                break
        self.wait(0.15)

    def safe_widths(self, widths):
        """each requested gripper opening, narrowed (down to 12 mm) until the opened fingers clear the other arm and the
        cart; opening is not a planned motion, so it is checked here"""
        self._sync()
        out = {}
        for side, w in widths.items():
            q = self.q_ref()
            allow = self.P.viol(q, (side,)) or None
            for wk in (w, 0.75 * w, 0.5 * w, 0.012):
                q[self.cart.finger_dofs[side]] = max(0.0, (wk - 0.001) / 2)
                if self.P.free(q, (side,), allow):
                    break
            if wk < w:
                self.note(f"{side} gripper opens to {wk * 1e3:.0f} mm, not {w * 1e3:.0f}: the other hand is in the way")
            out[side] = wk
        return out

    def open_wait(self, widths, timeout=1.2, speed=0.2):
        widths = self.safe_widths(widths)
        for side, w in widths.items():
            self.cart.set_gripper(side, w, speed=speed)
        for _ in range(int(round(timeout / self.dt))):
            self.step(1)
            if all(self.cart.gripper_width(s) >= w - 0.002 for s, w in widths.items()):
                break
        self.wait(0.05)

    def compliance(self, side, scale):
        """scale one arm's joint PD gains (1 = the cart's own). A stiff arm pulling a part out of a fit held by the other
        hand forces every millimetre of misalignment through the fit and jams it (the drawer effect); a soft arm lets the
        part self-align, as impedance control does on a real robot. Gravity compensation keeps a soft arm from sagging."""
        from yam_cart import ARM_KP, ARM_KV
        dofs = self.cart.arm_dofs[side]
        self.cart.entity.set_dofs_kp(np.asarray(ARM_KP) * scale, dofs)
        self.cart.entity.set_dofs_kv(np.asarray(ARM_KV) * math.sqrt(scale), dofs)

    def tcp(self, side):
        return self.cart.tcp_pose(side)[0]

    def impedance_on(self, side, **kw):
        self.imp[side] = Impedance(self, side, **kw)
        return self.imp[side]

    def impedance_off(self, side):
        """back to joint position control, holding the arm where it is."""
        self.imp.pop(side, None)
        self.cart.set_arm_target(side, self.cart.arm_q(side))

    def home(self, dur=1.2, base_x=None):
        h = self.cart.home_qpos
        self.go({s: h[self.cart.arm_dofs[s]] for s in ("left", "right")}, dur, base_x=base_x)

    def start_record(self, path):
        """per-frame state for re-rendering without re-simulating (nyx_replay.py): robot qpos, every desk object's pose
        (libuipc state) and the shell vertices, at the video frame rate."""
        from genesis.utils.misc import tensor_to_array
        self._rec = dict(path=path, t=[], qpos=[], poses={n: [] for n in S.SC["objects"] if n in self.E},
                         shells={n: [] for n in S.SC["deformables"] if n in self.E})

        def grab(dd):
            if dd.k % dd.every:
                return
            r = dd._rec
            r["t"].append(dd.t); r["qpos"].append(dd.cart.qpos().copy())
            for n in r["poses"]:
                r["poses"][n].append(dd.pose_any(n))
            for n in r["shells"]:
                r["shells"][n].append(tensor_to_array(dd.E[n].get_state().pos)[0].astype(np.float32))
        self.on_step.append(grab)

    def pose_any(self, name):
        """pose of any rigid desk object: libuipc state where libuipc integrates it, else Genesis's."""
        if name in self._D:
            return self.pose(name)
        from genesis.utils.misc import tensor_to_array
        T = np.eye(4); l = self.E[name].base_link
        T[:3, 3] = tensor_to_array(l.get_pos()); w, x, y, z = tensor_to_array(l.get_quat())
        T[:3, :3] = Rot.from_quat([x, y, z, w]).as_matrix()
        return T

    def start_export(self, out_dir, camera=None, kinds=None):
        """Blender-compositor export (yam-arcade export.Recorder format, rendered by render_desk.py): the robot parts,
        every desk object's visual mesh (the textured OBJs nyx_assets.py bakes) and the shells, at the video frame rate.
        Everything is shifted down by the cart's 2.5 cm lift so its wheels stand on the studio floor at z = 0."""
        import re
        import xml.etree.ElementTree as ET
        import nyx_assets
        from export import Recorder
        self._dz = np.array([0.0, 0.0, S.H - 0.75])
        rec = Recorder(out_dir, station="desk", fps=self.fps)
        self.cart.register(rec)
        urdfs = nyx_assets.nyx_urdfs(self._override)
        kinds = dict(jar=dict(kind="glass", color=(0.92, 0.97, 0.96)), stick=dict(kind="wood", color=(0.85, 0.72, 0.50)),
                     **(kinds or {}))
        self._exp = dict(rec=rec, rigid=[], shells={})
        for n in S.SC["objects"]:
            if n not in self.E:
                continue
            root = ET.parse(urdfs[n]["urdf"]).getroot(); d = os.path.dirname(urdfs[n]["urdf"])
            for i, v in enumerate(root.iter("visual")):
                o = v.find("origin"); m = v.find("geometry/mesh")
                Tv = np.eye(4)
                if o is not None:
                    Tv[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
                    Tv[:3, :3] = Rot.from_euler("xyz", [float(x) for x in o.get("rpy", "0 0 0").split()]).as_matrix()
                name = f"desk/{n}" + (f"/{i}" if i else "")
                mat = dict(kinds.get(n, dict(kind="plastic", roughness=0.45)))
                fixed = bool(S.SC["objects"][n].get("fixed", False))
                pose = None
                if fixed:
                    T = self.pose_any(n) @ Tv; x, y, z, w = Rot.from_matrix(T[:3, :3]).as_quat()
                    pose = ((T[:3, 3] - self._dz).tolist(), [w, x, y, z])
                rec.add_rigid(name, os.path.normpath(os.path.join(d, m.get("filename"))), material=mat, static_pose=pose)
                if not fixed:
                    self._exp["rigid"].append((name, n, Tv))
        for n, dd in S.SC["deformables"].items():
            if n in self.E:
                F = np.asarray(trimesh.load(os.path.join(S.A, dd["mesh"]), process=False).faces, np.int32)
                kind = dict(paper="paper", cloth="fabric", pad="fabric")[dd["kind"]]
                rec.add_meshseq(f"desk/{n}", material=dict(kind=kind, color=tuple(dd["rgba"][:3])))
                self._exp["shells"][n] = F
        self._exp["camera"] = camera

        def grab(dd):
            if dd.k % dd.every:
                return
            from genesis.utils.misc import tensor_to_array
            e = dd._exp; rigid = {}
            for name, n, Tv in e["rigid"]:
                T = dd.pose_any(n) @ Tv; x, y, z, w = Rot.from_matrix(T[:3, :3]).as_quat()
                rigid[name] = (T[:3, 3] - dd._dz, np.array([w, x, y, z]))
            robot = {k: (np.asarray(p) - dd._dz, q) for k, (p, q) in dd.cart.poses().items()}
            meshes = {f"desk/{n}": (tensor_to_array(dd.E[n].get_state().pos)[0].astype(np.float32) - dd._dz.astype(np.float32), F)
                      for n, F in e["shells"].items()}
            e["rec"].frame(dd.t, rigid={**robot, **rigid}, meshes=meshes)
        self.on_step.append(grab)

    def close_export(self):
        e = getattr(self, "_exp", None)
        if e is None:
            return
        cam = e["camera"]
        extra = {}
        if cam:   # station-local camera for render_desk.py, shifted like everything else
            extra["camera"] = dict(eye=list(np.asarray(cam["pos"]) - self._dz), target=list(np.asarray(cam["lookat"]) - self._dz),
                                   lens=float(36.0 / (2 * math.tan(math.radians(cam.get("fov", 40)) / 2))))
        e["rec"].close(extra_meta=extra)

    def save_record(self):
        if getattr(self, "_rec", None) is None:
            return
        r = self._rec
        np.savez_compressed(r["path"], t=np.array(r["t"]), qpos=np.array(r["qpos"]),
                            **{f"pose__{n}": np.array(v) for n, v in r["poses"].items()},
                            **{f"shell__{n}": np.array(v) for n, v in r["shells"].items()})

    def start_video(self, path):
        """the cameras record themselves every 1 / (fps dt) steps."""
        for k, c in self.C.items():
            c.start_recording(save_to_filename=path if len(self.C) == 1 else path.replace(".mp4", f"_{k}.mp4"), fps=self.fps)

    def stop_video(self):
        for c in self.C.values():
            c.stop_recording()
