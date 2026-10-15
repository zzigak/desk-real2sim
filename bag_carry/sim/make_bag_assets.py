"""Assets for the bag scenes (sim/run_bag.py), following CoDimRecon (arXiv 2609.36024): a bag is a manifold
thin shell (midsurface + scalar thickness), a bin is a separate rigid solid.

Writes assets/bag/:
  trash_bin.obj    one watertight solid: kitchen bin, outer radius 11 cm, 6 mm wall and floor, 22 cm tall
  trash_bag.obj    open-top shell lining it: radius 9.9 cm (5 mm inside the wall), bottom 4 mm above the bin floor,
                   rim 5 cm above the bin's rim; ~1 cm triangles, closed flat bottom
  paper_bag.obj    open-top shell with a flat rectangular bottom: 16 x 10 cm, 24 cm tall; 1 cm grid

    python3 make_bag_assets.py      (genesis venv)
"""
import json
import os

import numpy as np
import trimesh
from scipy.spatial import Delaunay

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "assets", "bag")
H = 0.01  # target edge length

BIN = dict(r_out=0.11, wall=0.006, height=0.22, floor=0.006)
TRASH_BAG = dict(radius=BIN["r_out"] - BIN["wall"] - 0.005, bottom=BIN["floor"] + 0.004, top=BIN["height"] + 0.05)
PAPER_BAG = dict(w=0.16, d=0.10, h=0.24)


def trash_bin():
    b = BIN
    ri = b["r_out"] - b["wall"]
    profile = np.array([[0.0, 0.0], [b["r_out"], 0.0], [b["r_out"], b["height"]], [ri, b["height"]],
                        [ri, b["floor"]], [0.0, b["floor"]]])
    m = trimesh.creation.revolve(profile, sections=96)
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    if m.volume < 0:
        m.invert()
    assert m.is_watertight and m.volume > 0
    return m


def open_cylinder(radius, z_bottom, z_top, h=H):
    """Closed flat bottom (Delaunay disc) + wall rings sharing the disc's outer ring; open at the top."""
    n = int(round(2 * np.pi * radius / h))
    pts = [[0.0, 0.0]]
    K = int(np.floor(radius / h))
    for k in range(1, K):
        r = k * h
        m = max(6, int(round(2 * np.pi * r / h)))
        a = np.linspace(0, 2 * np.pi, m, endpoint=False) + 0.37 * k
        pts += np.stack([r * np.cos(a), r * np.sin(a)], 1).tolist()
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer0 = len(pts)
    pts += np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1).tolist()
    pts = np.array(pts)
    tri = Delaunay(pts).simplices
    v = [np.concatenate([pts, np.full((len(pts), 1), z_bottom)], 1)]
    faces = [tri]
    rows = int(round((z_top - z_bottom) / h))
    zs = np.linspace(z_bottom, z_top, rows + 1)
    prev = np.arange(outer0, outer0 + n)
    count = len(pts)
    for z in zs[1:]:
        ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n, z)], 1)
        cur = np.arange(count, count + n)
        v.append(ring)
        count += n
        for i in range(n):
            a, b_, c, d = prev[i], prev[(i + 1) % n], cur[(i + 1) % n], cur[i]
            faces.append(np.array([[a, b_, c], [a, c, d]]))
        prev = cur
    m = trimesh.Trimesh(np.concatenate(v), np.concatenate(faces), process=False)
    trimesh.repair.fix_winding(m)
    if m.face_normals[: len(tri), 2].mean() > 0:  # bottom normals point down, i.e. outward
        m.invert()
    return m


def open_box_shell(w, d, hgt, h=H):
    """Flat rectangular bottom (grid) + four walls as rings over its boundary; open at the top."""
    nx, ny = int(round(w / h)), int(round(d / h))
    xs, ys = np.linspace(-w / 2, w / 2, nx + 1), np.linspace(-d / 2, d / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = [np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], -1)]
    idx = np.arange(X.size).reshape(nx + 1, ny + 1)
    f = []
    for i in range(nx):
        for j in range(ny):
            a, b_, c, e = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            f += [[a, c, b_], [a, e, c]]
    # boundary ring, counter-clockwise seen from above
    ring = ([idx[i, 0] for i in range(nx)] + [idx[nx, j] for j in range(ny)] + [idx[i, ny] for i in range(nx, 0, -1)]
            + [idx[0, j] for j in range(ny, 0, -1)])
    base_xy = v[0][ring, :2]
    n = len(ring)
    prev = np.array(ring)
    count = X.size
    rows = int(round(hgt / h))
    for z in np.linspace(0, hgt, rows + 1)[1:]:
        cur = np.arange(count, count + n)
        v.append(np.concatenate([base_xy, np.full((n, 1), z)], 1))
        count += n
        for i in range(n):
            a, b_, c, e = prev[i], prev[(i + 1) % n], cur[(i + 1) % n], cur[i]
            f += [[a, b_, c], [a, c, e]]
        prev = cur
    m = trimesh.Trimesh(np.concatenate(v), np.array(f), process=False)
    trimesh.repair.fix_winding(m)
    if m.face_normals[: 2 * nx * ny, 2].mean() > 0:
        m.invert()
    return m


# --- v2, following CoDimRecon Fig. 1: a loose liner folded over the rim of a tapered waste basket -------------------
BIN2 = dict(r_bot=0.10, r_top=0.125, wall=0.006, height=0.24, floor=0.006)  # outer radii; wall/floor thickness
# 6 mm off the bin everywhere; 3 narrow inward folds (fins of the liner's slack, 4 cm deep, 3 cm wide at the root),
# one facing the robot (-x). Gentle cosine pleats (5 x 4 cm over 72 deg each) were too flat to pinch: the fingers slid
# over them.
LINER = dict(clear=0.006, folds=(np.pi, np.pi + 2 * np.pi / 3, np.pi - 2 * np.pi / 3), fold_depth=0.04, fold_root=0.03,
             cuff=0.05)


def bin_r_in(z):
    b = BIN2
    return (b["r_bot"] - b["wall"]) + (b["r_top"] - b["r_bot"]) * z / b["height"]


def bin_r_out(z):
    b = BIN2
    return b["r_bot"] + (b["r_top"] - b["r_bot"]) * z / b["height"]


def tapered_bin():
    b = BIN2
    H = b["height"]
    profile = np.array([[0.0, 0.0], [bin_r_out(0), 0.0], [bin_r_out(H), H], [bin_r_in(H), H],
                        [bin_r_in(b["floor"]), b["floor"]], [0.0, b["floor"]]])
    m = trimesh.creation.revolve(profile, sections=96)
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    if m.volume < 0:
        m.invert()
    assert m.is_watertight and m.volume > 0
    return m


def liner(h=H):
    """Loose plastic liner: flat bottom 4 mm above the bin floor, walls 6 mm inside the bin with 7 inward pleats (the
    slack a real liner has, and something to pinch), the top folded over the rim on a circular arc 6 mm clear of it,
    and a 5 cm cuff hanging outside the bin, also 6 mm clear. One open edge: the cuff's lower hem."""
    b, L = BIN2, LINER
    H = b["height"]
    # angles: ~1 cm apart around the rim, plus 9 extra per fold so each fin is resolved (~3 mm apart across it)
    w = 0.5 * L["fold_root"] / (bin_r_in(H) - L["clear"])  # fold half-width, rad
    base = np.linspace(0, 2 * np.pi, int(round(2 * np.pi * bin_r_out(H) / h)), endpoint=False)
    extra = np.concatenate([(c + np.linspace(-w, w, 11)) % (2 * np.pi) for c in L["folds"]])
    th = np.unique(np.round(np.concatenate([base[np.min(np.abs(np.angle(np.exp(1j * (base[:, None] - np.array(L["folds"])[None]))))
                                                               , axis=1) > 1.2 * w], extra]), 9))
    n = len(th)
    zb = b["floor"] + 0.004

    def pleat(z):  # fold depth: full below H-3.5cm, fading to 0 by H-1cm (the fold over the rim is round)
        return L["fold_depth"] * np.clip((H - 0.01 - z) / 0.025, 0.0, 1.0)

    d = np.min(np.abs(np.angle(np.exp(1j * (th[:, None] - np.array(L["folds"])[None])))), axis=1)
    wave = np.clip(1.0 - d / w, 0.0, 1.0)  # triangular fin profile

    def ring_wall(z):
        r = bin_r_in(z) - L["clear"] - pleat(z) * wave
        return np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], 1)

    rings = []
    # bottom: polar rings scaling the pleated contour, centre vertex added separately
    contour = bin_r_in(zb) - L["clear"] - pleat(zb) * wave
    K = int(round(contour.mean() / h))
    for k in range(1, K + 1):
        r = contour * k / K
        rings.append(np.stack([r * np.cos(th), r * np.sin(th), np.full(n, zb)], 1))
    # inner wall up to the rim
    for z in np.linspace(zb, H, int(round((H - zb) / h)) + 1)[1:]:
        rings.append(ring_wall(z))
    # arc over the rim: centre on the rim's mid-thickness, radius = half the wall + clearance
    rc = 0.5 * (bin_r_in(H) + bin_r_out(H))
    rho = 0.5 * (bin_r_out(H) - bin_r_in(H)) + L["clear"]
    for a in np.linspace(np.pi, 0.0, max(4, int(round(np.pi * rho / h))) + 1)[1:]:
        r, z = rc + rho * np.cos(a), H + rho * np.sin(a)
        rings.append(np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], 1))
    # cuff hanging outside
    for z in np.linspace(H, H - L["cuff"], int(round(L["cuff"] / h)) + 1)[1:]:
        r = bin_r_out(z) + L["clear"]
        rings.append(np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], 1))
    v = np.concatenate([np.array([[0.0, 0.0, zb]])] + rings)
    f = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    for j in range(len(rings) - 1):
        a0, b0 = 1 + j * n, 1 + (j + 1) * n
        for i in range(n):
            i1 = (i + 1) % n
            f += [[a0 + i, b0 + i, b0 + i1], [a0 + i, b0 + i1, a0 + i1]]
    m = trimesh.Trimesh(v, np.array(f), process=False)
    trimesh.repair.fix_winding(m)
    return m


def handle_bag(w=0.20, d=0.12, hgt=0.22, span=0.08, leg=0.02, rise=0.09, h=H):
    """Paper shopping bag: flat w x d bottom, hgt walls, open top, and on each long wall a die-cut style handle: an
    inverted-U strip in the wall's plane rising `rise` above the rim, legs `leg` wide, inner span `span`. The handle
    legs grow out of the rim, so bag + handles are one manifold sheet."""
    m = open_box_shell(w, d, hgt, h)
    v, f = list(m.vertices), [list(x) for x in m.faces]
    V = np.array(v)
    top = np.where(np.isclose(V[:, 2], hgt))[0]
    xs = np.round(np.arange(-span / 2 - leg, span / 2 + leg + 1e-9, h), 6)
    zs = np.round(np.arange(hgt, hgt + rise + 1e-9, h), 6)
    for side in (-1, 1):
        y = side * d / 2
        idx = {}
        for x in xs:
            for z in zs:
                if z == zs[0]:
                    k = top[np.argmin(np.abs(V[top, 0] - x) + np.abs(V[top, 1] - y))]
                    assert abs(V[k, 0] - x) < 1e-6 and abs(V[k, 1] - y) < 1e-6, "handle leg not on a rim vertex"
                    idx[(x, z)] = int(k)
                else:
                    idx[(x, z)] = len(v)
                    v.append([x, y, z])
        bar = zs[-1] - leg
        for x0, x1 in zip(xs[:-1], xs[1:]):
            for z0, z1 in zip(zs[:-1], zs[1:]):
                in_leg = x1 <= -span / 2 + 1e-9 or x0 >= span / 2 - 1e-9
                if in_leg or z0 >= bar - 1e-9:
                    a, b_, c, e = idx[(x0, z0)], idx[(x1, z0)], idx[(x1, z1)], idx[(x0, z1)]
                    f += [[a, b_, c], [a, c, e]]
        # drop the unused grid points inside the U (they are not referenced by faces)
    m2 = trimesh.Trimesh(np.array(v), np.array(f), process=False)
    m2.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(m2)
    return m2


def rope_handles(w=0.20, d=0.12, hgt=0.22, span=0.08, rise=0.09, tuck=0.02, h=H):
    """Twisted-paper rope handles for the plain open box bag: on each long wall (y = +-d/2) a polyline that starts
    `tuck` below the rim on the wall at x = -span/2, rises along the wall to the rim, arches `rise` above it in the
    wall's plane and comes back down to x = +span/2, `tuck` below the rim. The tucked ends lie on wall vertices of
    open_box_shell(w, d, hgt) (1 cm grid), where they are glued (stitched) to the bag. Returns [points (N, 3)] x 2 and
    the number of tucked vertices per end."""
    out = []
    n_tuck = int(round(tuck / h)) + 1  # wall vertices from rim-tuck up to the rim
    for side in (-1, 1):
        y = side * d / 2
        down = [np.array([-span / 2, y, hgt - tuck + k * h]) for k in range(n_tuck)]
        # half-ellipse arc from (-span/2, rim) over (0, rim+rise) to (+span/2, rim), ~1 cm spacing
        a, b = span / 2, rise
        L = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))) / 2
        m = int(round(L / h))
        arc = [np.array([-a * np.cos(t), y, hgt + b * np.sin(t)]) for t in np.linspace(0, np.pi, m + 1)[1:-1]]
        up = [np.array([span / 2, y, hgt - k * h]) for k in range(n_tuck)]
        out.append(np.array(down + arc + up))
    return out, n_tuck


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    b = trash_bin()
    b.export(os.path.join(OUT, "trash_bin.obj"))
    meta["trash_bin"] = dict(BIN, faces=len(b.faces), volume=float(b.volume))
    t = TRASH_BAG
    bag = open_cylinder(t["radius"], t["bottom"], t["top"])
    bag.export(os.path.join(OUT, "trash_bag.obj"))
    meta["trash_bag"] = dict(t, verts=len(bag.vertices), faces=len(bag.faces), boundary_edges=int(len(bag.edges) - 2 * len(bag.faces) + len(bag.edges_unique) * 0))
    p = PAPER_BAG
    pb = open_box_shell(p["w"], p["d"], p["h"])
    pb.export(os.path.join(OUT, "paper_bag.obj"))
    meta["paper_bag"] = dict(p, verts=len(pb.vertices), faces=len(pb.faces))
    for name, m in (("trash_bag", bag), ("paper_bag", pb)):
        e = np.sort(m.edges, axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        meta[name].update(open_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                          winding_consistent=bool(m.is_winding_consistent), extents=np.round(m.extents, 4).tolist())
    tb = tapered_bin()
    tb.export(os.path.join(OUT, "basket.obj"))
    meta["basket"] = dict(BIN2, faces=len(tb.faces), volume=float(tb.volume), watertight=bool(tb.is_watertight))
    ln = liner()
    ln.export(os.path.join(OUT, "liner.obj"))
    meta["liner"] = dict(LINER, verts=len(ln.vertices), faces=len(ln.faces))
    rb = open_box_shell(0.20, 0.12, 0.22)
    rb.export(os.path.join(OUT, "rope_bag.obj"))
    ropes, n_tuck = rope_handles()
    np.save(os.path.join(OUT, "rope_handles.npy"), np.array(ropes, dtype=object), allow_pickle=True)
    rv = np.array(rb.vertices)
    for r in ropes:
        for pnt in list(r[:n_tuck]) + list(r[-n_tuck:]):
            assert np.linalg.norm(rv - pnt, axis=1).min() < 1e-9, "rope end not on a bag vertex"
    seg = [np.linalg.norm(np.diff(r, axis=0), axis=1) for r in ropes]
    meta["rope_bag"] = dict(w=0.20, d=0.12, h=0.22, verts=len(rb.vertices), faces=len(rb.faces), rope_points=len(ropes[0]),
                            rope_tuck_vertices=n_tuck, rope_segment_mm=[float(min(map(min, seg)) * 1000), float(max(map(max, seg)) * 1000)])
    hb = handle_bag()
    hb.export(os.path.join(OUT, "handle_bag.obj"))
    meta["handle_bag"] = dict(w=0.20, d=0.12, h=0.22, span=0.08, leg=0.02, rise=0.09, verts=len(hb.vertices), faces=len(hb.faces))
    for name, m in (("trash_bag", bag), ("paper_bag", pb), ("liner", ln), ("handle_bag", hb), ("rope_bag", rb)):
        e = np.sort(m.edges, axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        meta[name].update(open_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                          winding_consistent=bool(m.is_winding_consistent), extents=np.round(m.extents, 4).tolist())
    # liner clearance to the basket (must exceed d_hat = 1 mm everywhere)
    clr = trimesh.proximity.ProximityQuery(tb).signed_distance(ln.vertices)
    meta["liner"]["min_clearance_to_basket_mm"] = float(-clr.max() * 1000)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("basket", "liner", "rope_bag")}, indent=1))


if __name__ == "__main__":
    main()
