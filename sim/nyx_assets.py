"""Nyx-renderable copies of the desk URDFs. Nyx shades textures but ignores vertex colours (and crashes on PLY
visuals), so every vertex-coloured PLY visual is baked into an OBJ + MTL + PNG: each triangle gets its own cell in a
texture atlas (two triangles per square cell, a 1-texel inset against bilinear bleeding), filled with the barycentric
blend of its vertex colours. Positions and smooth vertex normals are written shared, only the texture coordinates are
per corner, so shading stays smooth. Collision meshes are untouched.

nyx_urdfs(override) -> scene.build urdf_override that swaps the visuals (writes assets/urdf_nyx/)."""
import os, re
import numpy as np
import trimesh
from PIL import Image

import scene as S

NU = os.path.join(S.A, "urdf_nyx")
CELL = 8


def bake(ply, out_obj):
    m = trimesh.load(ply, process=False)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    C = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32)
    N = np.asarray(m.vertex_normals)
    nf = len(F); ncell = (nf + 1) // 2; G = int(np.ceil(np.sqrt(ncell))); T = G * CELL
    tex = np.zeros((T, T, 3), np.float32)
    # texel centres of one cell, and the two triangles' corner texels (inset by one texel)
    u, v = np.meshgrid(np.arange(CELL) + 0.5, np.arange(CELL) + 0.5)
    a, b = 1.0, CELL - 1.0
    triA = np.array([[a, a], [b, a], [a, b]]); triB = np.array([[b, b], [a, b], [b, a]])

    def bary(tri):
        (x0, y0), (x1, y1), (x2, y2) = tri
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        w0 = ((y1 - y2) * (u - x2) + (x2 - x1) * (v - y2)) / d
        w1 = ((y2 - y0) * (u - x2) + (x0 - x2) * (v - y2)) / d
        return np.stack([w0, w1, 1 - w0 - w1], -1)
    wA, wB = bary(triA), bary(triB)
    inA = (u + v) <= CELL                                  # the lower-left half belongs to triangle A
    W = np.where(inA[..., None], np.clip(wA, 0, 1), np.clip(wB, 0, 1)); W /= W.sum(-1, keepdims=True)
    fa = np.arange(0, nf, 2); fb = np.arange(1, nf, 2)
    cols = np.zeros((ncell, CELL, CELL, 3), np.float32)
    cA = C[F[fa]]                                          # (ncell, 3 corners, 3)
    cols[:] = np.einsum("hwk,nkc->nhwc", W * inA[..., None], cA)
    if len(fb):
        cB = C[F[fb]]
        cols[:len(fb)] += np.einsum("hwk,nkc->nhwc", W * (~inA)[..., None], cB)
        cols[len(fb):] += np.einsum("hwk,nkc->nhwc", W * (~inA)[..., None], cA[len(fb):])   # odd count: A fills its cell
    gi, gj = np.divmod(np.arange(ncell), G)
    tex.reshape(G, CELL, G, CELL, 3)[gi, :, gj, :, :] = cols
    png = os.path.splitext(out_obj)[0] + ".png"
    Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8)[::-1]).save(png)   # OBJ v runs bottom-up
    # per-corner texture coordinates
    cell = np.repeat(np.arange(ncell), 2)[:nf]; half = np.tile([0, 1], ncell)[:nf]
    corners = np.where(half[:, None, None] == 0, triA[None], triB[None])
    ox, oy = gj[cell] * CELL, gi[cell] * CELL
    uv = np.stack([(ox[:, None] + corners[..., 0]) / T, (oy[:, None] + corners[..., 1]) / T], -1).reshape(-1, 2)
    name = os.path.splitext(os.path.basename(out_obj))[0]
    with open(os.path.splitext(out_obj)[0] + ".mtl", "w") as f:
        f.write(f"newmtl {name}\nKd 1 1 1\nKa 0 0 0\nKs 0.05 0.05 0.05\nNs 20\nmap_Kd {os.path.basename(png)}\n")
    idx = np.arange(nf * 3).reshape(nf, 3) + 1
    with open(out_obj, "w") as f:
        f.write(f"mtllib {name}.mtl\nusemtl {name}\n")
        f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
        f.write("".join(f"vn {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in N))
        f.write("".join(f"vt {x:.6f} {1 - y:.6f}\n" for x, y in uv))
        Fv = F + 1
        f.write("".join(f"f {a}/{ta}/{a} {b}/{tb}/{b} {c}/{tc}/{c}\n" for (a, b, c), (ta, tb, tc) in zip(Fv, idx)))
    return out_obj


def nyx_urdfs(override=None):
    """URDF override that makes every desk object Nyx-renderable (baked OBJ visuals); `override` (robot fit URDFs)
    is respected and passed through for everything it names."""
    os.makedirs(NU, exist_ok=True)
    override = dict(override or {})
    out = {}
    for name, o in S.SC["objects"].items():
        src = override.get(name, {}).get("urdf", os.path.join(S.A, o["urdf"]))
        text = open(src).read(); d = os.path.dirname(src)
        def fix(mo):
            fn = mo.group(1); p = os.path.normpath(os.path.join(d, fn))
            if fn.endswith(".ply"):
                obj = os.path.join(NU, f"{name}__{os.path.splitext(os.path.basename(fn))[0]}.obj")
                if not os.path.exists(obj) or os.path.getmtime(obj) < os.path.getmtime(p):
                    bake(p, obj)
                p = obj
            return f'filename="{os.path.relpath(p, NU)}"'
        text = re.sub(r'filename="([^"]+)"', fix, text)
        dst = os.path.join(NU, f"{name}.urdf"); open(dst, "w").write(text)
        out[name] = {**override.get(name, {}), "urdf": dst}
        out[name].setdefault("rho", o["rho"])
    return out
