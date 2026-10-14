"""SAM 3D meshes -> scene assets: z-up, real size, decimated (vertex colours kept), yaw fitted to the VGGT points.
Writes assets/<name>_vis.ply (+ _vis_<part>.ply splits) and assets/sam3d_fit.json."""
import json, os, sys
import numpy as np, trimesh, pymeshlab
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = os.path.join(ROOT, "assets")
L = json.load(open(os.path.join(ROOT, "layout.json")))
SP = json.load(open(os.path.join(A, "specs.json")))
TC_FIX = None
PICK = {  # name: (sam3d file, size rule)
    "mouse": ("mouse_p10_s42", dict(length=0.108)),
    "keyboard": ("keyboard_p10_s42", dict(length=0.356)),
    "bottle": ("bottle_p02_s42", dict(diameter=2 * SP["bottle"]["r"], height=SP["bottle"]["top"])),
    "tea_tin": ("tea_tin_p03_s7", dict(diameter=2 * SP["tea_tin"]["r"],
                                        height=SP["tea_tin"]["lid_seated_z"] + SP["tea_tin"]["lid_h"] + SP["tea_tin"]["disc_h"])),
}

def load_zup(f):
    sc = trimesh.load(f); g = sc.to_geometry() if isinstance(sc, trimesh.Scene) else sc
    V = g.vertices[:, [0, 2, 1]] * np.array([1, -1, 1])           # glTF y-up -> z-up: (x, y, z) -> (x, -z, y)
    return V, np.asarray(g.visual.vertex_colors), g.faces

def decimate(V, Cc, F, target):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F, v_color_matrix=Cc.astype(np.float64) / 255.0))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, preservenormal=True, qualitythr=0.5)
    m = ms.current_mesh()
    return m.vertex_matrix(), (m.vertex_color_matrix() * 255).clip(0, 255).astype(np.uint8), m.face_matrix()

def normalise(V, rule):
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2
    V = V - np.array([c[0], c[1], lo[2]])
    ext = hi - lo
    if "length" in rule:
        s = rule["length"] / max(ext[0], ext[1]); return V * s
    sxy = rule["diameter"] / ((ext[0] + ext[1]) / 2); sz = rule["height"] / ext[2]
    return V * np.array([sxy, sxy, sz])

def fit_pose(V, Cc, P, Pc, xy0, z0, lam=0.03):
    """yaw (deg) and xy offset minimising mean NN distance + lam * grey-level mismatch (both sets normalised)."""
    g = lambda c: c[:, :3].astype(float) @ [0.299, 0.587, 0.114]
    gm, gp = g(Cc), g(Pc)
    gm = (gm - gm.mean()) / (gm.std() + 1e-6)
    global gp_s
    best = (1e9, None)
    for coarse in (True, False):
        yaws = np.arange(0, 360, 4) if coarse else np.arange(best[1][0] - 4, best[1][0] + 4.1, 1)
        offs = [(dx, dy) for dx in np.arange(-0.02, 0.021, 0.01) for dy in np.arange(-0.02, 0.021, 0.01)] if coarse else \
               [(best[1][1] + dx, best[1][2] + dy) for dx in np.arange(-0.006, 0.0061, 0.003) for dy in np.arange(-0.006, 0.0061, 0.003)]
        for yaw in yaws:
            a = np.radians(yaw); R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            W = V @ R.T
            for dx, dy in offs:
                tree = cKDTree(W + np.array([xy0[0] + dx, xy0[1] + dy, z0]))
                d, j = tree.query(P[:, :3])
                cost = d.mean() + lam * np.abs(gm[j] - gp_s).mean()
                if cost < best[0]: best = (cost, (yaw, dx, dy))
    return best

if __name__ == "__main__":
    pts = np.load(os.path.join(ROOT, "work", "obj_points.npz"))
    TC = np.array([(-0.405 + 0.51) / 2, (-0.152 + 0.40) / 2]) * 1.37
    fits = {}
    for name, (src, rule) in PICK.items():
        V, Cc, F = load_zup(os.path.join(ROOT, "sam3d", src + ".glb"))
        V, Cc, F = decimate(V, Cc, F, 60000 if name in ("tea_tin", "bottle") else 40000)
        V = normalise(V, rule)
        mesh = trimesh.Trimesh(V, F, vertex_colors=Cc, process=False)
        # VGGT points in world frame (table-centre origin, z from table top)
        P = pts[name + "_p"].astype(float); P[:, :2] -= TC; Pc = pts[name + "_c"]
        xy0 = np.array(L[name]["xy"]); z0 = 0.0
        if name == "mouse": z0 = L["mousepad"]["thickness"]
        keep = np.linalg.norm(P[:, :2] - xy0, axis=1) < 0.25
        P, Pc = P[keep], Pc[keep]
        sel = np.random.default_rng(1).choice(len(P), min(4000, len(P)), replace=False); P, Pc = P[sel], Pc[sel]
        g = lambda c: c[:, :3].astype(float) @ [0.299, 0.587, 0.114]
        gp = g(Pc); gp_s = (gp - gp.mean()) / (gp.std() + 1e-6)
        cost, (yaw, dx, dy) = fit_pose(V, Cc, P, Pc, xy0, z0)
        fits[name] = dict(src=src, yaw=float(yaw), xy=[float(xy0[0] + dx), float(xy0[1] + dy)], z0=z0, cost=float(cost),
                          extents=mesh.extents.tolist(), faces=len(F))
        print(name, fits[name], flush=True)
        mesh.export(os.path.join(A, f"{name}_vis.ply"))
    json.dump(fits, open(os.path.join(A, "sam3d_fit.json"), "w"), indent=1)
