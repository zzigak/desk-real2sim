"""Point-splat contact sheet of SAM 3D meshes (vertex colours, z-buffer), 3 views each. glTF frame is Y-up."""
import sys, glob, numpy as np, trimesh, cv2
def load(f):
    sc = trimesh.load(f); g = sc.to_geometry() if isinstance(sc, trimesh.Scene) else sc
    return g.vertices, np.asarray(g.visual.vertex_colors)[:, :3]
def view(V, Cc, az, el, res=260):
    V = V - V.mean(0); V = V / np.abs(V).max()
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = V @ (Rx @ Ry).T
    u = ((P[:, 0] * 0.45 + 0.5) * res).astype(int); v = ((-P[:, 1] * 0.45 + 0.5) * res).astype(int)
    img = np.full((res, res, 3), 255, np.uint8); zb = np.full((res, res), -9.0)
    o = np.argsort(P[:, 2]); u, v, z, c = u[o], v[o], P[o, 2], Cc[o]
    ok = (u >= 1) & (u < res - 1) & (v >= 1) & (v < res - 1)
    for du in (0, 1):
        for dv in (0, 1):
            img[v[ok] + dv, u[ok] + du] = c[ok][:, ::-1]
    return img
rows = []
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "sam3d/*.glb")):
    V, C = load(f)
    ims = [view(V, C, az, el) for az, el in ((0, 15), (90, 15), (200, 30), (0, 80))]
    lab = np.full((260, 200, 3), 255, np.uint8)
    cv2.putText(lab, f.split("/")[-1][:-4], (5, 130), 0, 0.5, (0, 0, 0), 1)
    rows.append(np.hstack([lab] + ims))
cv2.imwrite("work/sam3d_sheet.jpg", np.vstack(rows))
