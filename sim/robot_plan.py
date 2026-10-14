"""Collision-free motion planning for the desk's YAM cart, on the CPU (MuJoCo kinematics of the same MJCF Genesis loads;
the link poses agree with Genesis's to 1e-5 m).

Every commanded arm motion is checked against robot_collide.Collider: the arm links may not pass through each other,
the other arm, the cart, the tabletop or the rigid desk objects (the fingertip pads and palms may touch the desk and
its objects: that contact is libuipc's). Pieces:

- `cart_xml(lift)`: the cart MJCF with the I2RT lift-column carriage (carriage, deck and both arms) lowered by `lift`
  metres. The column, base and mast stay put. Arm bases level with the tabletop are hard on a desk: reaching down
  in front of the edge folds the wrist into the deck.
- `Planner.ik`: damped-least-squares IK of one arm, from the seed and then from random restarts; only
  collision-free solutions count, and the one nearest the seed wins.
- `Planner.chain`: joint waypoints for a straight TCP line (slerped flange orientation), each seeded by the last,
  with every segment checked.
- `Planner.route`: a collision-free joint path for a free move: straight if clear, else via raised and tucked
  poses, else RRT-connect with shortcutting.
"""
import math, os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot, Slerp

from robot_collide import Collider, SIDES

TCP_OFFSET = np.array([3.8898e-05, 9.681e-05, 0.14465])
W_ROT = 0.35                     # m per rad: orientation weight in the IK error
J_W = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.4])   # joint-distance weights (proximal joints move more mass)
LIFT_BODIES = ("left_base", "right_base")
LIFT_GEOMS = ("carriage", "carriage_col", "deck", "deck_col")


def cart_xml(lift=0.0):
    """MJCF path of the cart with its lift carriage lowered by `lift` m (0 = the yam-arcade model)."""
    import yam_cart
    src = yam_cart.model_path("yam")
    if abs(lift) < 1e-6:
        return src
    import xml.etree.ElementTree as ET
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "robot_lift", f"lift{round(lift * 1000):03d}")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.normpath(os.path.join(out_dir, "yam_cart_yam.xml"))
    tree = ET.parse(src)                                  # always rebuilt: the mesh dir is an absolute, per-machine path
    for el in tree.getroot().iter():
        if (el.tag == "body" and el.get("name") in LIFT_BODIES) or (el.tag == "geom" and el.get("name") in LIFT_GEOMS):
            p = [float(v) for v in el.get("pos").split()]
            p[2] -= lift
            el.set("pos", " ".join(f"{v:.6g}" for v in p))
    text = ET.tostring(tree.getroot(), encoding="unicode")
    if not (os.path.exists(dst) and open(dst).read() == text):
        tmp = dst + f".tmp{os.getpid()}"
        open(tmp, "w").write(text)
        os.replace(tmp, dst)
    return dst


def _wxyz(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z])


def _xyzw(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w])


class Planner:
    def __init__(self, base_pos, xml, objects=(), clear=0.002, seed=0):
        self.col = Collider(base_pos, xml=xml, objects=objects, margin=max(0.012, 2 * clear))
        self.m, self.d = self.col.m, self.col.d
        self.base_pos = np.asarray(base_pos, float)
        self.clear = clear
        m = self.m
        jid = lambda n: m.joint(n).id
        self.arm = {s: np.array([m.jnt_qposadr[jid(f"{s}_joint{i}")] for i in range(1, 7)]) for s in SIDES}
        self.dof = {s: np.array([m.jnt_dofadr[jid(f"{s}_joint{i}")] for i in range(1, 7)]) for s in SIDES}
        self.lo = {s: m.jnt_range[[jid(f"{s}_joint{i}") for i in range(1, 7)], 0] for s in SIDES}
        self.hi = {s: m.jnt_range[[jid(f"{s}_joint{i}") for i in range(1, 7)], 1] for s in SIDES}
        self.flange = {s: m.body(f"{s}_gripper").id for s in SIDES}
        self.rng = np.random.default_rng(int(os.environ.get("PLAN_SEED", seed)))
        self.home = None                                  # full home qpos, set by the caller
        self.checks = 0

    # -------------------------------------------------------------------------------------------- kinematics
    def fk(self, q, side):
        """TCP world position and flange quaternion (w, x, y, z) for full config q."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        b = self.flange[side]; Rf = self.d.xmat[b].reshape(3, 3)
        return self.d.xpos[b] + Rf @ TCP_OFFSET + self.base_pos, _wxyz(Rot.from_matrix(Rf).as_quat())

    def _solve(self, side, q, pos, Rt, iters=200):
        """DLS IK of one arm from full config q (modified in place); -> (q, pos err, rot err)."""
        m, d = self.m, self.d; a, dof = self.arm[side], self.dof[side]; b = self.flange[side]
        lo, hi = self.lo[side], self.hi[side]
        p_t = np.asarray(pos, float) - self.base_pos
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lam2 = 1e-4; hist = []
        for it in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            Rf = d.xmat[b].reshape(3, 3); p = d.xpos[b] + Rf @ TCP_OFFSET
            ep = p_t - p
            er = Rot.from_matrix(Rt @ Rf.T).as_rotvec()
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
                break
            hist.append(np.linalg.norm(ep) + W_ROT * np.linalg.norm(er))
            if it >= 40 and hist[-1] > 0.02 and hist[-1] > 0.97 * hist[-21]:   # stuck far off (joint limits): give up
                break
            mujoco.mj_jac(m, d, jp, jr, p, b)
            J = np.vstack([jp[:, dof], W_ROT * jr[:, dof]])
            e = np.concatenate([ep, W_ROT * er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            n = np.abs(dq).max()
            if n > 0.25:
                dq *= 0.25 / n
            q[a] = np.clip(q[a] + dq, lo, hi)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        Rf = d.xmat[b].reshape(3, 3); p = d.xpos[b] + Rf @ TCP_OFFSET
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(Rot.from_matrix(Rt @ Rf.T).as_rotvec()))

    def viol(self, q, sides=SIDES):
        """{(geom a, geom b): how far inside the clearance} for pairs closer than `clear` (links of the same arm: closer
        than 0, their hulls already wrap the joint housings)"""
        out = {}
        for a, b, dd in self.col.contacts(q, sides):
            if a.split("_")[0] == b.split("_")[0] and a.split("_")[0] in SIDES:
                c = 0.0                                   # same arm
            elif b.startswith("sheet_"):
                c = 0.0                                   # a sheet: no vertex inside
            elif self.col.transit and (a.startswith("desk_") or b.startswith("desk_")) and \
                    any(h in a or h in b for h in ("tip_left", "tip_right", "gripper_palm")):
                c = 0.02                                  # a hand in transit keeps 2 cm off the desk and its objects
            else:
                c = self.clear
            if dd < c:
                out[(a, b)] = max(out.get((a, b), 0.0), c - dd)
        return out

    def free(self, q, sides=SIDES, allow=None):
        """no pair closer than `clear`, except pairs in `allow` ({pair: depth}, e.g. contact the start pose already
        had) that get no more than 2 mm deeper"""
        self.checks += 1
        v = self.viol(q, sides)
        if allow:
            return all(k in allow and dv <= allow[k] + 2e-3 for k, dv in v.items())
        return not v

    def blame(self, q, sides=SIDES):
        return [(a, b, round(1e3 * dd, 1)) for a, b, dd in self.col.contacts(q, sides) if dd < self.clear]

    def ik(self, side, pos, quat, q_ref, seed=None, tol=2e-3, rtol=0.03, restarts=24, max_jump=None, check=True, allow=None):
        """6 joint angles putting `side`'s TCP at `pos` with flange `quat` (wxyz), the rest of the robot at q_ref
        (full config). Tries the seed (default: q_ref's arm), the home pose and `restarts` random poses; keeps
        converged, collision-free solutions (within max_jump rad of the seed on every joint, if given) and returns
        the one nearest the seed. Raises RuntimeError if there is none."""
        a = self.arm[side]
        Rt = Rot.from_quat(_xyzw(quat)).as_matrix()
        s0 = np.asarray(q_ref)[a].copy() if seed is None else np.asarray(seed, float)
        seeds = [s0]
        if self.home is not None:
            seeds.append(self.home[a].copy())
        seeds += [self.rng.uniform(self.lo[side], self.hi[side]) for _ in range(restarts)]
        best, best_cost, why = None, np.inf, "no IK convergence"
        for k, sd in enumerate(seeds):
            q = np.array(q_ref, float); q[a] = sd
            q, ep, er = self._solve(side, q, pos, Rt)
            if ep > tol or er > rtol:
                continue
            jump = np.abs(q[a] - s0)
            if max_jump is not None and jump.max() > max_jump:
                why = f"only solutions {jump.max():.2f} rad from the seed"; continue
            cost = float(J_W @ jump)
            if cost >= best_cost:
                continue
            if check and not self.free(q, (side,), allow):
                why = f"collides: {self.blame(q, (side,))[:3]}"; continue
            best, best_cost = q[a].copy(), cost
            if k == 0 and cost < 0.3:                     # the seed's own branch, close by: done
                break
        if best is None:
            raise RuntimeError(f"IK {side} at {np.round(pos, 3)}: {why}")
        return best

    def ik_all(self, side, pos, quat, q_ref, tol=2e-3, rtol=0.03, restarts=40, min_sep=0.4):
        """distinct collision-free IK solutions (arm configurations at least min_sep rad apart on some joint), nearest
        to q_ref's arm first"""
        a = self.arm[side]; Rt = Rot.from_quat(_xyzw(quat)).as_matrix(); s0 = np.asarray(q_ref)[a]
        seeds = [s0.copy()] + ([self.home[a].copy()] if self.home is not None else [])
        seeds += [self.rng.uniform(self.lo[side], self.hi[side]) for _ in range(restarts)]
        sols = []
        for sd in seeds:
            q = np.array(q_ref, float); q[a] = sd
            q, ep, er = self._solve(side, q, pos, Rt)
            if ep > tol or er > rtol or any(np.abs(q[a] - o).max() < min_sep for o in sols):
                continue
            if self.free(q, (side,)):
                sols.append(q[a].copy())
        return sorted(sols, key=lambda x: float(J_W @ np.abs(x - s0)))

    # -------------------------------------------------------------------------------------------- paths
    def segment_free(self, qa, qb, sides, step=0.03, allow=None):
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        n = max(1, int(math.ceil(np.abs(qb - qa).max() / step)))
        for u in np.linspace(0, 1, n + 1)[1:]:
            if not self.free(qa + u * (qb - qa), sides, allow):
                return False
        return True

    def path_free(self, qs, sides, allow=None):
        if not self.free(qs[0], sides, allow):
            return False
        return all(self.segment_free(qs[i], qs[i + 1], sides, allow=allow) for i in range(len(qs) - 1))

    def chain(self, side, q_ref, p0, q0, p1, q1, n_via=12, seed=None, tol=6e-3, rtol=0.12, max_jump=0.6):
        """arm joint waypoints [seed, ...] for a straight TCP line p0 -> p1 with the flange slerped q0 -> q1 (wxyz),
        the rest of the robot at q_ref; every waypoint and segment collision-free. Raises RuntimeError."""
        sl = Slerp([0, 1], Rot.from_quat([_xyzw(q0), _xyzw(q1)]))
        ang = (Rot.from_quat(_xyzw(q1)) * Rot.from_quat(_xyzw(q0)).inv()).magnitude()
        n_via = max(n_via, int(math.ceil(ang / 0.12)), int(math.ceil(np.linalg.norm(np.subtract(p1, p0)) / 0.02)))
        a = self.arm[side]
        qs = [np.asarray(q_ref)[a].copy() if seed is None else np.asarray(seed, float)]
        qf = np.array(q_ref, float)
        q_s = qf.copy(); q_s[a] = qs[0]
        allow = self.viol(q_s, (side,)) or None           # leaving a contact the start already has is fine
        for u in np.linspace(0, 1, n_via + 1)[1:]:
            pos = np.asarray(p0) + u * (np.asarray(p1) - np.asarray(p0))
            qw = _wxyz(sl([u]).as_quat()[0])
            qs.append(self.ik(side, pos, qw, q_ref, seed=qs[-1], tol=tol, rtol=rtol, restarts=6, max_jump=max_jump,
                              allow=allow))
            qa_, qb_ = qf.copy(), qf.copy(); qa_[a] = qs[-2]; qb_[a] = qs[-1]
            if not self.segment_free(qa_, qb_, (side,), allow=allow):
                raise RuntimeError(f"chain {side}: segment to {np.round(pos, 3)} collides: {self.blame(qb_, (side,))[:3]}")
        return qs

    def route(self, qa, qb, sides, tries=4000):
        """see _route; free moves are transit: the fingertips and palms may not touch the desk or its objects either
        (from the start pose's own contact they may only move away)"""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        if not self.free(qb, sides):
            raise RuntimeError(f"route: goal collides: {self.blame(qb, sides)[:3]}")
        self.col.transit = True
        try:
            # what the start and the goal themselves have (a hand leaving an object, or arriving at a pre-grasp pose
            # close to one) may stay, not deepen
            allow = self.viol(qa, sides)
            for k, v in self.viol(qb, sides).items():
                allow[k] = max(allow.get(k, 0.0), v)
            return self._route(qa, qb, sides, tries, allow or None)
        finally:
            self.col.transit = False

    def _route(self, qa, qb, sides, tries=4000, allow=None):
        """collision-free joint path [qa, ..., qb] (full configs) moving only `sides`' arms (and the base, linearly,
        if it differs). Straight if clear; else via a tucked pose (home); else RRT-connect + shortcuts."""
        if self.segment_free(qa, qb, sides, allow=allow):
            return [qa, qb]
        if self.home is not None:
            qh = qb.copy()
            for s in sides:
                qh[self.arm[s]] = self.home[self.arm[s]]
            if self.free(qh, sides, allow) and self.segment_free(qa, qh, sides, allow=allow) and \
                    self.segment_free(qh, qb, sides, allow=allow):
                return [qa, qh, qb]
        return self._rrt(qa, qb, sides, tries, allow)

    def _rrt(self, qa, qb, sides, tries, allow=None):
        idx = np.concatenate([self.arm[s] for s in sides])
        lo = np.concatenate([self.lo[s] for s in sides]); hi = np.concatenate([self.hi[s] for s in sides])
        base = qb.copy()

        def full(x):
            q = base.copy(); q[idx] = x; return q

        T = [[qa[idx].copy()], [qb[idx].copy()]]; P = [[-1], [-1]]
        step = 0.25

        def extend(t, x):
            near = int(np.argmin([np.abs(n - x).max() for n in T[t]]))
            v = x - T[t][near]; dist = np.abs(v).max()
            new = x if dist <= step else T[t][near] + v * (step / dist)
            if self.segment_free(full(T[t][near]), full(new), sides, allow=allow):
                T[t].append(new); P[t].append(near); return len(T[t]) - 1
            return None

        for it in range(tries):
            x = self.rng.uniform(lo, hi)
            t = it % 2
            i = extend(t, x)
            if i is None:
                continue
            # connect the other tree towards the new node
            o = 1 - t; tgt = T[t][i]
            while True:
                j = extend(o, tgt)
                if j is None:
                    break
                if np.abs(T[o][j] - tgt).max() < 1e-9:
                    pa = self._trace(T[t], P[t], i); pb = self._trace(T[o], P[o], j)
                    path = pa[::-1] + pb[1:] if t == 0 else pb[::-1] + pa[1:]
                    path = [full(x_) for x_ in path]
                    path[0] = qa.copy(); path[-1] = qb.copy()
                    return self._shortcut(path, sides, allow)
        raise RuntimeError(f"route: no collision-free path for {sides} after {tries} samples")

    @staticmethod
    def _trace(T, P, i):
        out = []
        while i >= 0:
            out.append(T[i]); i = P[i]
        return out

    def _shortcut(self, path, sides, allow=None, n=150):
        path = list(path)
        for _ in range(n):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.segment_free(path[i], path[j], sides, allow=allow):
                path = path[:i + 1] + path[j:]
        return path
