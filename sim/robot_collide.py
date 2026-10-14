"""Collision checks for the YAM cart's arms (MuJoCo, CPU), for planning and for auditing recorded runs.

In the libuipc runs only the fingertip pads and the palms collide; the arm links, the arm bases and the cart pass
through everything. A plan is only physical if the arms never pass through themselves, each other, the cart or the
desk. This model has every arm link (the I2RT convex hulls), the cart's collision boxes, the tabletop slab and every
rigid desk object's URDF collision geometry (mesh -> convex hull), placed in the cart's model frame:
model = world - cart entity position (the cart's base joints then move it inside that frame).

The fingertip pads and palms are left out of the desk checks (libuipc handles their contact); they still count for
self, arm-arm and cart collisions. Same-arm pairs whose hulls overlap at the joint between them in every pose (see
STRUCTURAL) are not collisions.

    col = Collider(base_pos=cart_entity_pos, objects={"table": T, ...})
    col.hits(q19)          # [(geom a, geom b, penetration m), ...] beyond `tol`
    col.clearance(q19)     # smallest signed distance over the checked pairs (m, < 0 = penetrating)
"""
import os, sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh

YA = os.environ.get("YAM_ARCADE", os.path.expanduser("~/projects/yam-arcade"))
for p in (YA, os.path.join(YA, "robots")):
    if p not in sys.path:
        sys.path.insert(0, p)
import scene as S

SIDES = ("left", "right")
TCP_OFFSET = np.array([3.8898e-05, 9.681e-05, 0.14465])
ARM_LINKS = ("base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right")
HAND = ("gripper", "tip_left", "tip_right")      # libuipc-coupled: their desk contact is real physics
MARGIN = 0.05                                    # contacts reported up to 5 cm apart, for clearance readouts
# contype / conaffinity bits: arm geoms collide with arms, the cart and the desk; cart and desk never with each other
ARM_BITS, CART_BITS, DESK_BITS = (1, 1 | 2 | 4), (2, 0), (4, 0)


def _urdf_geoms(name):
    """[(kind, size or mesh path, T_link_geom)] from an object's URDF collision elements."""
    path = os.path.join(S.A, S.SC["objects"][name]["urdf"]) if "urdf" in S.SC["objects"][name] else None
    if path is None or not os.path.exists(path):
        return []
    root = ET.parse(path).getroot(); d = os.path.dirname(path); out = []
    for c in root.iter("collision"):
        T = np.eye(4); o = c.find("origin")
        if o is not None:
            from scipy.spatial.transform import Rotation as R
            T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
            T[:3, :3] = R.from_euler("xyz", [float(x) for x in o.get("rpy", "0 0 0").split()]).as_matrix()
        g = c.find("geometry")[0]
        if g.tag == "box":
            out.append(("box", np.array([float(x) for x in g.get("size").split()]) / 2, T))
        elif g.tag == "cylinder":
            out.append(("cylinder", np.array([float(g.get("radius")), float(g.get("length")) / 2]), T))
        elif g.tag == "sphere":
            out.append(("sphere", np.array([float(g.get("radius"))]), T))
        elif g.tag == "mesh":
            out.append(("mesh", os.path.normpath(os.path.join(d, g.get("filename"))), T))
    return out


def _box_mesh(spec, name, half):
    """an 8-vertex mesh asset for a box of half-extents `half`; returns its name"""
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half, float)[:3]
    h = trimesh.Trimesh(V, process=False).convex_hull
    spec.add_mesh(name=name, uservert=np.asarray(h.vertices, float).ravel(), userface=np.asarray(h.faces, np.int32).ravel())
    return name


def _quat(Rm):
    from scipy.spatial.transform import Rotation as R
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return [w, x, y, z]


class Collider:
    def __init__(self, base_pos, xml=None, objects=(), tol=2e-3, margin=MARGIN):
        import yam_cart
        xml = xml or yam_cart.model_path("yam")
        self.base_pos = np.asarray(base_pos, float)
        self.tol = tol
        spec = mujoco.MjSpec.from_file(xml)
        for g in spec.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue
            body = g.parent.name
            side = next((s for s in SIDES if body.startswith(s + "_")), None)
            g.contype, g.conaffinity = ARM_BITS if side else CART_BITS
            g.margin = margin
            if g.type == mujoco.mjtGeom.mjGEOM_BOX:      # every pair goes through the mesh-mesh routine (see contacts)
                g.meshname = _box_mesh(spec, f"bx_{g.name}", g.size)
                g.type = mujoco.mjtGeom.mjGEOM_MESH
        self.objects = []
        for n in objects:
            geoms = _urdf_geoms(n)
            if not geoms:
                continue
            b = spec.worldbody.add_body(name=f"desk_{n}", mocap=True)
            for i, (kind, size, T) in enumerate(geoms):
                kw = dict(name=f"desk_{n}_{i}", pos=T[:3, 3], quat=_quat(T[:3, :3]), contype=DESK_BITS[0],
                          conaffinity=DESK_BITS[1], margin=margin, group=3)
                if kind == "mesh":
                    m = trimesh.load(size, force="mesh", process=False).convex_hull
                    mname = f"m_{n}_{i}"
                    spec.add_mesh(name=mname, uservert=np.asarray(m.vertices, float).ravel(),
                                  userface=np.asarray(m.faces, np.int32).ravel())
                    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, **kw)
                elif kind == "box":
                    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=_box_mesh(spec, f"bx_{n}_{i}", size), **kw)
                else:
                    t = dict(cylinder=mujoco.mjtGeom.mjGEOM_CYLINDER, sphere=mujoco.mjtGeom.mjGEOM_SPHERE)[kind]
                    sz = np.zeros(3); sz[:len(size)] = size
                    b.add_geom(type=t, size=sz, **kw)
            self.objects.append(n)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.mocap = {n: self.m.body(f"desk_{n}").mocapid[0] for n in self.objects}
        m = self.m
        self.gname = [m.geom(i).name for i in range(m.ngeom)]
        self.gbody = [m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)]
        self.skip = np.array([[self._structural(i, j) for j in range(m.ngeom)] for i in range(m.ngeom)])
        # fingertips and palms against the desk and its objects: left to libuipc while a hand manipulates, but checked
        # in transit (free moves), where touching anything is an accident
        self.handdesk = np.array([[self._handdesk(i, j) for j in range(m.ngeom)] for i in range(m.ngeom)])
        self.transit = False
        self.sheets, self.sheet_held = {}, set()
        self.gside = [self._part(i)[0] for i in range(m.ngeom)]
        self.margin = margin

    def _part(self, i):
        """(side or 'cart' or 'desk', link name)"""
        b = self.gbody[i]
        for s in SIDES:
            if b.startswith(s + "_"):
                return s, b[len(s) + 1:]
        return ("desk", b[5:]) if b.startswith("desk_") else ("cart", b)

    def _handdesk(self, i, j):
        (si, li), (sj, lj) = self._part(i), self._part(j)
        return "desk" in (si, sj) and (lj if si == "desk" else li) in HAND

    def _structural(self, i, j):
        (si, li), (sj, lj) = self._part(i), self._part(j)
        if "desk" in (si, sj):
            return False
        if si == sj and si in SIDES:                     # same arm
            a, b = sorted((ARM_LINKS.index(li), ARM_LINKS.index(lj)))
            return (a, b) in STRUCTURAL
        if "cart" in (si, sj) and (li == "base" or lj == "base"):
            return True                                  # the arm base plates are bolted to the deck
        return False

    def set_objects(self, poses, ignore=()):
        """poses: {name: 4x4 world pose}; objects in `ignore` (e.g. held in a hand) are moved out of the way."""
        for n, T in poses.items():
            if n in self.mocap:
                k = self.mocap[n]
                self.d.mocap_pos[k] = np.asarray(T)[:3, 3] - self.base_pos - (0, 0, 100.0 if n in ignore else 0.0)
                self.d.mocap_quat[k] = _quat(np.asarray(T)[:3, :3])

    def contacts(self, q, sides=None):
        """[(geom a, geom b, signed distance)] for checked pairs closer than the margin; with `sides`, only pairs
        involving one of those arms."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        # every collision box is an 8-vertex convex mesh here: MuJoCo 3.10's box-box narrow phase reports bogus depths
        # for boxes near each other (-46 mm for fingertip pads 23.5 mm apart), and mj_geomDistance flips signs now and
        # then (+12.6 / -12.6 mm on consecutive frames of a still arm); the mesh-mesh routine agrees with both elsewhere
        d1 = {}
        for c in self.d.contact[:self.d.ncon]:
            i, j = int(c.geom1), int(c.geom2)
            if self.skip[i, j] or (self.handdesk[i, j] and not self.transit) or \
                    (sides is not None and self.gside[i] not in sides and self.gside[j] not in sides):
                continue
            d1[(i, j)] = min(d1.get((i, j), np.inf), float(c.dist))
        out = [(self.gname[i], self.gname[j], dc) for (i, j), dc in d1.items()]
        return out + self._sheet_contacts(sides)

    def _sheet_contacts(self, sides):
        """the cloth, paper and pad (self.sheets, world vertices) as obstacles: [(geom, 'sheet_<name>', -depth)] for
        sheet vertices inside an arm link's hull, and in transit inside a fingertip or palm too (except within 6 cm of
        the TCP of a hand in self.sheet_held, which is holding that sheet)"""
        out = []
        if not self.sheets:
            return out
        hulls = self._hulls(hand=self.transit)
        tcp = {}
        for s_ in self.sheet_held:
            b = self.m.body(f"{s_}_gripper").id
            tcp[s_] = self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ TCP_OFFSET
        for name, V in self.sheets.items():
            Vm = V - self.base_pos
            for i, eq in hulls.items():
                side, link = self._part(i)
                if sides is not None and side not in sides:
                    continue
                t = self.d.geom_xpos[i]
                near = np.abs(Vm - t).max(1) < 0.2
                if link in HAND and side in tcp:
                    near &= np.linalg.norm(Vm - tcp[side], axis=1) > 0.06
                if not near.any():
                    continue
                P = (Vm[near] - t) @ self.d.geom_xmat[i].reshape(3, 3)
                sd = (P @ eq[:, :3].T + eq[:, 3]).max(1)        # > 0 outside; inside: -(distance to the nearest face)
                if (sd < 0).any():
                    out.append((self.gname[i], f"sheet_{name}", float(sd.min())))
        return out

    def hits(self, q, tol=None, sides=None):
        """pairs penetrating deeper than `tol` (negative tol = closer than -tol counts as a hit)"""
        tol = self.tol if tol is None else tol
        return [(a, b, -dd) for a, b, dd in self.contacts(q, sides) if dd < -tol]

    def _hulls(self, hand=False):
        """per arm geom (with hand=True also the libuipc-coupled hand parts): halfspace equations of its hull"""
        if getattr(self, "_hull_eq", None) is None:
            from scipy.spatial import ConvexHull
            m = self.m; self._hull_eq = {}
            for i in range(m.ngeom):
                side, link = self._part(i)
                if side not in SIDES or m.geom_contype[i] == 0:
                    continue
                if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                    k = m.geom_dataid[i]
                    V = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[i]
                else:
                    continue
                self._hull_eq[i] = ConvexHull(V).equations
        return {i: e for i, e in self._hull_eq.items() if hand or self._part(i)[1] not in HAND}

    def sheet_hits(self, q, V, tol=0.002):
        """{geom name: number of sheet vertices (world, (N, 3)) deeper than `tol` inside that arm link's hull}"""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        out = {}
        Vm = np.asarray(V, float) - self.base_pos
        for i, eq in self._hulls().items():
            R_ = self.d.geom_xmat[i].reshape(3, 3); t = self.d.geom_xpos[i]
            near = np.abs(Vm - t).max(1) < 0.25
            if not near.any():
                continue
            P = (Vm[near] - t) @ R_
            inside = (P @ eq[:, :3].T + eq[:, 3] < -tol).all(1)
            if inside.any():
                out[self.gname[i]] = int(inside.sum())
        return out

    def clearance(self, q, sides=None):
        return min((dd for *_, dd in self.contacts(q, sides)), default=self.margin)


# same-arm link pairs (indices into ARM_LINKS) whose hulls overlap at their shared joint housing in normal poses;
# calibrated by robot_collide.py --calibrate (parent-child pairs are excluded by MuJoCo already)
STRUCTURAL = {(4, 6)}                            # link4 hull vs the palm box across link5: 97% of random poses


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="same-arm pairs in contact over random poses")
    ap.add_argument("--audit", nargs="*", default=[], help="<states.npz> files: per-frame penetrations")
    ap.add_argument("--shift", default="", help="name=dx,dy,dz layout shifts used by the run")
    a = ap.parse_args()
    from robot_desk import cart_pos
    if a.calibrate:
        STRUCTURAL.clear()
        col = Collider(cart_pos(0.0))
        rng = np.random.default_rng(0)
        lo, hi = col.m.jnt_range[:, 0], col.m.jnt_range[:, 1]
        cnt = {}
        N = 4000
        for _ in range(N):
            q = np.zeros(col.m.nq)
            q[3:] = rng.uniform(lo[3:], hi[3:])
            q[[9, 10, 17, 18]] = rng.uniform(0, 0.0475)
            seen = set()
            for ga, gb, dd in col.contacts(q):
                ia, ib = col.gname.index(ga), col.gname.index(gb)
                (sa, la), (sb, lb) = col._part(ia), col._part(ib)
                if sa == sb and sa in SIDES and dd < -2e-3:
                    seen.add(tuple(sorted((ARM_LINKS.index(la), ARM_LINKS.index(lb)))))
            for k in seen:
                cnt[k] = cnt.get(k, 0) + 1
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(ARM_LINKS[k[0]], ARM_LINKS[k[1]], f"{100 * v / N:.0f}% of random poses")
    for path in a.audit:
        dat = np.load(path)
        names = [k[6:] for k in dat.files if k.startswith("pose__")]
        # the run's cart placement (robot_demos notes "cart: lift L m, back B m"; runs before that note: 0, 0)
        import re, robot_plan as RP
        lift = back = 0.0
        jp = path.replace("_states.npz", ".json")
        if os.path.exists(jp):
            for e in json.load(open(jp)).get("events", []):
                m_ = re.match(r"cart: lift ([-0-9.]+) m, back ([-0-9.]+) m", e["msg"])
                if m_:
                    lift, back = float(m_.group(1)), float(m_.group(2))
        col = Collider(np.asarray(cart_pos(0.0)) - (0.0, back, 0.0), xml=RP.cart_xml(lift), objects=names)
        print(f"   (cart lift {lift}, back {back})")
        q = dat["qpos"]; t = dat["t"]
        worst = {}
        for f in range(len(q)):
            col.set_objects({n: dat[f"pose__{n}"][f] for n in names})
            deep = {}
            for ga, gb, pen in col.hits(q[f]):
                key = tuple(sorted((ga, gb))); deep[key] = max(deep.get(key, 0.0), pen)
            for key, pen in deep.items():
                w = worst.setdefault(key, dict(frames=0, max_mm=0.0, t0=float(t[f]), t1=0.0))
                w["frames"] += 1; w["max_mm"] = max(w["max_mm"], 1e3 * pen); w["t1"] = float(t[f])
        sheets = {}
        for k in dat.files:
            if k.startswith("shell__"):
                Vs = dat[k]
                for f in range(0, len(q), 2):
                    for g, nv in col.sheet_hits(q[f], Vs[f]).items():
                        w = sheets.setdefault((k[7:], g), dict(frames=0, max_verts=0, t0=float(t[f]), t1=0.0))
                        w["frames"] += 1; w["max_verts"] = max(w["max_verts"], nv); w["t1"] = float(t[f])
        print(f"== {path}: {len(q)} frames")
        for k, w in sorted(worst.items(), key=lambda kv: -kv[1]["frames"]):
            print(f"  {k[0]:28s} {k[1]:28s} {w['frames']:4d} frames  max {w['max_mm']:5.1f} mm  t {w['t0']:.1f}-{w['t1']:.1f}")
        for (sh, g), w in sorted(sheets.items(), key=lambda kv: -kv[1]["frames"]):
            print(f"  sheet {sh:16s} in {g:28s} {2 * w['frames']:4d} frames  max {w['max_verts']:4d} vertices  t {w['t0']:.1f}-{w['t1']:.1f}")
