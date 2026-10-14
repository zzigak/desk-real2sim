"""Tiny OBJ+MTL writer for textured boxes (per-face material + UV) and helpers for URDF generation."""
import os
import numpy as np

# box faces: (axis, sign) -> corner order CCW seen from outside, in terms of (-1/+1) offsets of the half extents
_FACES = {
    ("z", 1): [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    ("z", -1): [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)],
    ("y", -1): [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
    ("y", 1): [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)],
    ("x", 1): [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
    ("x", -1): [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)],
}

class Obj:
    def __init__(self):
        self.v, self.vt, self.groups = [], [], {}          # groups: material -> list of faces [(vi, ti) x3]
    def box(self, center, size, mats, uvfun=None):
        """mats: dict face key -> material name (key 'default' for the rest). uvfun(face_key, corner_xyz) -> (u, v)."""
        c, h = np.asarray(center, float), np.asarray(size, float) / 2
        for key, corners in _FACES.items():
            mat = mats.get(key, mats.get("default"))
            idx = []
            for s in corners:
                p = c + h * np.array(s)
                self.v.append(p)
                uv = uvfun(key, p) if (uvfun and key in mats) else (0.5, 0.5)
                self.vt.append(uv); idx.append(len(self.v))
            g = self.groups.setdefault(mat, [])
            g.append([(idx[0], idx[0]), (idx[1], idx[1]), (idx[2], idx[2])])
            g.append([(idx[0], idx[0]), (idx[2], idx[2]), (idx[3], idx[3])])
        return self
    def write(self, path, materials):
        """materials: name -> dict(kd=(r,g,b), map=texture path relative to the obj dir or None, d=alpha)."""
        base = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(os.path.dirname(path), base + ".mtl"), "w") as f:
            for name, m in materials.items():
                f.write(f"newmtl {name}\nKa 0 0 0\nKd {m['kd'][0]:.4f} {m['kd'][1]:.4f} {m['kd'][2]:.4f}\nKs 0.05 0.05 0.05\nNs 20\n"
                        f"d {m.get('d', 1.0)}\nillum 2\n")
                if m.get("map"): f.write(f"map_Kd {m['map']}\n")
                f.write("\n")
        with open(path, "w") as f:
            f.write(f"mtllib {base}.mtl\n")
            for p in self.v: f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for t in self.vt: f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
            for mat, faces in self.groups.items():
                f.write(f"usemtl {mat}\n")
                for fc in faces: f.write("f " + " ".join(f"{a}/{b}" for a, b in fc) + "\n")

def inertia_box(m, s):
    x, y, z = s
    return m / 12 * (y * y + z * z), m / 12 * (x * x + z * z), m / 12 * (x * x + y * y)

def inertial_xml(mass, com=(0, 0, 0), I=(1e-4, 1e-4, 1e-4)):
    return (f'<inertial><origin xyz="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"/><mass value="{mass}"/>'
            f'<inertia ixx="{I[0]:.3e}" ixy="0" ixz="0" iyy="{I[1]:.3e}" iyz="0" izz="{I[2]:.3e}"/></inertial>')

def mesh_inertial(mesh, mass):
    """mass properties of a closed trimesh scaled to `mass` (uniform density)."""
    mp = mesh.mass_properties; k = mass / mp["mass"]
    I = mp["inertia"] * k
    return inertial_xml(mass, mp["center_mass"], (I[0, 0], I[1, 1], I[2, 2]))
