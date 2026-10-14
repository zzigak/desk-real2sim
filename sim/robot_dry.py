"""Dry run of a robot demo routine on the CPU: no physics, the arms track their targets exactly, the desk objects and
sheets stay where a previous run had them at one frame. It runs robot_demos.<demo> through the real Desk planning
methods (robot_plan.py), so unreachable or colliding plans fail here in seconds instead of on a cluster GPU, then
audits the whole arm trajectory with robot_collide (including the impedance phases, tracked by IK).

    python sim/robot_dry.py marker --states out/robot/marker/marker_states.npz [--frame 9] [--lift 0.1 --back 0.1]

Grasps "hold" nothing (the gripper reads `--width`), so anything that depends on what the physics did after a grasp
is only roughly right.
"""
import argparse, math, os, sys, time
import numpy as np

import robot_desk as R
import robot_plan as RP
import scene as S
from robot_collide import Collider


class _State:
    def __init__(self, pos):
        self.pos = pos[None]


class _Shell:
    def __init__(self, V):
        self.V = V

    def get_state(self):
        return _State(self.V)


class _Entity:
    """the few Genesis entity calls the routines make"""
    def set_dofs_force_range(self, *a, **k): pass
    def control_dofs_position_velocity(self, *a, **k): pass
    def get_dofs_control_force(self, dofs): return np.zeros(len(dofs))


class FakeCart:
    def __init__(self, P, q0, width):
        self.P = P
        self.q_target = q0.copy()
        self.base_dofs = np.array([0, 1, 2])
        self.arm_dofs = {s: P.arm[s].copy() for s in ("left", "right")}
        self.finger_dofs = {"left": np.array([9, 10]), "right": np.array([17, 18])}
        self.home_qpos = q0.copy()
        self.entity = _Entity()
        self.hold_width = width
        self.base_pos = P.base_pos

    def qpos(self): return self.q_target.copy()
    def arm_q(self, side): return self.q_target[self.arm_dofs[side]].copy()
    def tcp_pose(self, side): return self.P.fk(self.q_target, side)
    def set_arm_target(self, side, q): self.q_target[self.arm_dofs[side]] = q
    def set_base_target(self, x=0.0, y=0.0, yaw=0.0): self.q_target[self.base_dofs] = (x, y, yaw)
    def set_gripper(self, side, width, force=20.0, speed=0.05):
        self.q_target[self.finger_dofs[side]] = max(0.0, (width - 0.001) / 2)
    def open(self, side, speed=0.05): self.set_gripper(side, 0.096)
    def grasp(self, side, force=10.0, speed=0.05, squeeze_q=-0.02): self.set_gripper(side, self.hold_width)
    def grasped(self, side): return True
    def gripper_width(self, side): return float(self.q_target[self.finger_dofs[side]].sum() + 0.001)
    def finger_force(self, side): return np.zeros(2)
    def update(self): pass


class FakeImp:
    def __init__(self, d, side):
        from yam_cart import mat_from_quat
        self.d, self.side = d, side
        p, q = d.cart.tcp_pose(side)
        self.pos, self.R = p.copy(), mat_from_quat(q)
        self.F, self.M = np.zeros(3), np.zeros(3)

    def track(self):
        """perfect tracking of the target position, orientation held (no collision check: the audit sees it)"""
        P = self.d.P; q = self.d.cart.q_target.copy()
        tgt = np.array(self.pos, float); tgt[2] = max(tgt[2], S.H + 0.001)   # a press stops on the desk, as in the sim
        q, ep, _ = P._solve(self.side, q, tgt, self.R, iters=30)
        self.d.cart.q_target[P.arm[self.side]] = q[P.arm[self.side]]


class DryDesk(R.Desk):
    def __init__(self, states, frame, lift=0.0, back=0.0, width=0.008, keep=(), drop=()):
        import robot_plan as RP
        self.dt, self.fps = R.DT, 30
        self.every = max(1, int(round(1.0 / (self.fps * self.dt))))
        self.t, self.k = 0.0, 0
        self.base = np.zeros(3); self.on_step = []; self.imp = {}; self.log, self.events = [], []
        self.lift, self.back = lift, back
        self.cart_xml = RP.cart_xml(lift)
        dat = np.load(states)
        self._poses = {k[6:]: dat[k][frame] for k in dat.files if k.startswith("pose__") and k[6:] not in drop}
        self.E = {n: None for n in self._poses}
        for k in dat.files:
            if k.startswith("shell__") and k[7:] in keep:
                self.E[k[7:]] = _Shell(dat[k][frame].astype(float))
        base_pos = np.asarray(R.cart_pos(0.0)) - (0.0, back, 0.0)
        self.P = RP.Planner(base_pos, self.cart_xml, objects=list(self._poses))
        q0 = dat["qpos"][0].copy(); q0[:3] = 0.0
        self.cart = FakeCart(self.P, q0, width)
        self.P.home = q0
        self.P.home = self._ready_pose()
        self.cart.home_qpos = self.P.home.copy(); self.cart.q_target[:] = self.P.home
        self._D = {}
        self.qs = []
        self._states, self._sw, self._moves = states, None, []
        # carried objects: a grasp attaches the nearest rigid object within 4 cm of the TCP to that hand's flange
        self._held = {}
        cart = self.cart; g0, s0 = cart.grasp, cart.set_gripper

        def grasp(side, force=10.0, speed=0.05, squeeze_q=-0.02):
            g0(side, force, speed, squeeze_q)
            p, q = cart.tcp_pose(side)
            near = [(np.linalg.norm(T[:3, 3] - p), n) for n, T in self._poses.items() if n != "table"]
            dist, n = min(near) if near else (1.0, None)
            if dist < 0.04:
                self._held[side] = (n, np.linalg.inv(self._flange(side)) @ self._poses[n])

        def set_gripper(side, width, force=20.0, speed=0.05):
            s0(side, width, force, speed)
            if width > 0.02:
                self._held.pop(side, None)
        cart.grasp, cart.set_gripper = grasp, set_gripper
        self.dry = True

    def switch(self, t, frame):
        """at dry-run time t, swap the objects and sheets to the previous run's `frame` (e.g. after a drag)"""
        self._sw = (t, frame)

    def _load(self, frame):
        dat = np.load(self._states)
        self._poses = {k[6:]: dat[k][frame] for k in dat.files if k.startswith("pose__") and k[6:] in self._poses}
        for k in dat.files:
            if k.startswith("shell__") and self.E.get(k[7:]) is not None:
                self.E[k[7:]].V = dat[k][frame].astype(float)

    def _flange(self, side):
        from yam_cart import mat_from_quat
        p, q = self.cart.tcp_pose(side)
        T = np.eye(4); T[:3, :3] = mat_from_quat(q); T[:3, 3] = p
        return T

    def pose(self, name): return self._poses[name].copy()
    def pose_any(self, name): return self._poses[name].copy()

    def step(self, n=1):
        for _ in range(n):
            for imp in self.imp.values():
                if self.k % 3 == 0:
                    imp.track()
            for side, (n, rel) in self._held.items():
                self._poses[n] = self._flange(side) @ rel
            self.t += self.dt; self.k += 1
            if self._sw and self.t >= self._sw[0]:
                self._load(self._sw[1]); self._sw = None
            for mv in list(self._moves):                  # a sheet moved by the physics this dry run doesn't simulate
                if self.t >= mv[0]:
                    self.E[mv[1]].V = self.E[mv[1]].V + mv[2]; self._moves.remove(mv)
            if self.k % self.every == 0:
                self.qs.append(self.cart.q_target.copy())
            for f in self.on_step:
                try:
                    f(self)
                except Exception:
                    pass

    def impedance_on(self, side, **kw):
        self.imp[side] = FakeImp(self, side); return self.imp[side]

    def impedance_off(self, side):
        self.imp.pop(side, None)

    def grip_until(self, side, timeout): self.wait(min(timeout, 0.3))
    def open_wait(self, widths, timeout=1.2, speed=0.2):
        widths = self.safe_widths(widths)
        for s, w in widths.items():
            self.cart.set_gripper(s, w)
        self.wait(0.3)


def audit(d, every=1):
    col = d.P.col
    col.set_objects(d._poses)
    worst = {}
    sheets = {n: e.V for n, e in d.E.items() if isinstance(e, _Shell)}
    for f, q in enumerate(d.qs[::every]):
        for a, b, pen in col.hits(q, tol=2e-3):
            k = tuple(sorted((a, b))); w = worst.setdefault(k, [0, 0.0, f / d.fps])
            w[0] += 1; w[1] = max(w[1], pen)
        for n, V in sheets.items():
            for g, nv in col.sheet_hits(q, V).items():
                w = worst.setdefault((f"sheet {n}", g), [0, 0.0, f / d.fps])
                w[0] += 1; w[1] = max(w[1], nv / 1000.0)
    return worst


if __name__ == "__main__":
    import robot_demos as D
    ap = argparse.ArgumentParser()
    ap.add_argument("demo"); ap.add_argument("--states", required=True); ap.add_argument("--frame", type=int, default=9)
    ap.add_argument("--lift", type=float, default=None); ap.add_argument("--back", type=float, default=None)
    ap.add_argument("--width", type=float, default=0.008)
    ap.add_argument("--move", action="append", default=[], help="T:sheet:dx,dy,dz - shift a sheet at dry time T")
    ap.add_argument("--switch", default="", help="T:F - from dry time T on, objects and sheets as at the run's frame F")
    a = ap.parse_args()
    fn, keep, cam = D.DEMOS[a.demo][:3]
    cfg = D.CART.get(a.demo, {})
    lift = cfg.get("lift", 0.0) if a.lift is None else a.lift
    back = cfg.get("back", 0.0) if a.back is None else a.back
    d = DryDesk(a.states, a.frame, lift=lift, back=back, width=a.width, keep=keep, drop=D.DROP.get(a.demo, ()))
    if a.switch:
        d.switch(float(a.switch.split(":")[0]), int(a.switch.split(":")[1]))
    for mv in a.move:
        t_, n_, v_ = mv.split(":"); d._moves.append((float(t_), n_, np.array([float(x) for x in v_.split(",")])))
    log = []; t0 = time.time(); err = None
    try:
        res = fn(d, log)
    except Exception as e:
        import traceback; traceback.print_exc(); err = e; res = None
    print(f"dry run {a.demo} lift {lift} back {back}: {'FAILED ' + str(err)[:200] if err else 'ok'}; {d.t:.1f} s simulated, "
          f"{time.time() - t0:.0f} s wall, {d.P.checks} collision checks")
    for k, (n, pen, t) in sorted(audit(d).items(), key=lambda kv: -kv[1][0]):
        print(f"  HIT {k[0]:28s} {k[1]:28s} {n:4d} frames, max {1e3 * pen:5.1f} mm, from t {t:.1f}")
