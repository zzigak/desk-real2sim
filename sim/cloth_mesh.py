"""Folded two-layer cloth: a W x L sheet folded once along its length (fold on the -x side), layers `gap` apart.
Built flat; gravity drapes the overhanging part. Returns a trimesh in the cloth's local frame (z=0 = bottom layer)."""
import numpy as np, trimesh

def folded_cloth(width_folded, length, gap=0.003, h=0.0075, fold_segments=6):
    nx = int(round(width_folded / h)); ny = int(round(length / h))
    xs = np.linspace(0, width_folded, nx + 1); ys = np.linspace(0, length, ny + 1)
    # path across the width: bottom layer x: W -> 0, half circle around the fold, top layer x: 0 -> W
    r = gap / 2
    path = [(x, 0.0) for x in xs[::-1]]
    for k in range(1, fold_segments):
        a = -np.pi / 2 - np.pi * k / fold_segments            # from bottom (-90 deg) around the -x side to top
        path.append((r * np.cos(a) + 0.0 - 0.0, r + r * np.sin(a)))
    path += [(x, gap) for x in xs]
    path = np.array(path)
    path[len(xs):len(xs) + fold_segments - 1, 0] -= 0.0     # fold bulges to x<0
    V = np.array([[px, y, pz] for (px, pz) in path for y in ys])
    m = len(ys)
    F = []
    for i in range(len(path) - 1):
        for j in range(m - 1):
            a, b, c, d = i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1
            F += [[a, b, c], [a, c, d]]
    return trimesh.Trimesh(V, np.array(F), process=False)

def sheet(w, l, h):
    nx, ny = max(1, int(round(w / h))), max(1, int(round(l / h)))
    xs = np.linspace(-w / 2, w / 2, nx + 1); ys = np.linspace(-l / 2, l / 2, ny + 1)
    V = np.array([[x, y, 0.0] for x in xs for y in ys]); m = len(ys)
    F = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1
            F += [[a, b, c], [a, c, d]]
    return trimesh.Trimesh(V, np.array(F), process=False)
