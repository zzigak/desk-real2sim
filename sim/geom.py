"""Procedural, watertight collision/visual meshes for the desk scene (metres, z up).

revolve(): closed (r, z) profile -> solid of revolution. Profile vertices with r == 0 become poles.
A profile segment can carry a helical thread: r += sign * h * f(z / p - theta / 2pi + phase) * fade(z),
which keeps the surface a clean height field (no self-intersection), so a lid bore built as the jar neck
offset outward by a clearance mates with it exactly and screws along it under contact.
"""
import functools

import numpy as np
import trimesh


def _thread_f(u):
    return 0.5 - 0.5 * np.cos(2 * np.pi * u)          # rounded (sinusoidal) thread, 0 = root, 1 = crest


@functools.lru_cache(maxsize=None)
def thread_envelope(h, p, clear, n=4096):
    """female bore for a male thread r = r_root + h*f(z/p): the male profile grown by `clear` in every direction of the
    (z, r) plane (upper envelope of discs), as r - r_root over one pitch. A bore offset only radially leaves the flanks
    closer than the radial gap (0.76 mm of normal clearance from 1.4 mm radial at h 2.3, p 4.5), which under IPC
    pre-loads both flanks and binds the thread."""
    U = np.arange(n) / n; s = np.linspace(-clear, clear, 401)
    e = (h * _thread_f(U[:, None] + s[None] / p) + np.sqrt(np.maximum(clear ** 2 - s[None] ** 2, 0))).max(1)
    return lambda u: np.interp(np.mod(u, 1.0), U, e, period=1.0), h + clear


def revolve(profile, n=96, threads=(), subdiv=None):
    """profile: list of (r, z), closed implicitly (last -> first), counter-clockwise in the (r, z) half plane
    seen with r to the right and z up (i.e. outer surface goes upward). threads: list of dicts
    {seg: (i0, i1) profile index range, h, p, phase, z0, z1, ramp, rest}: r += h*(fade*f + (1-fade)*rest),
    rest=0 for a male thread (fades to the root), rest=1 for a female bore (fades fully open) applied to vertices in that range.
    A female thread with a `clear` key uses thread_envelope (uniform clearance) instead of r += h*f.
    subdiv: max segment length (float, or {segment index: length}) so threads are resolved."""
    prof = np.asarray(profile, float)
    pts, tags = [], []
    for i in range(len(prof)):
        a, b = prof[i], prof[(i + 1) % len(prof)]
        L = np.linalg.norm(b - a)
        lim = subdiv.get(i) if isinstance(subdiv, dict) else subdiv
        k = max(1, int(np.ceil(L / lim))) if lim else 1
        for t in np.arange(k) / k:
            pts.append(a + (b - a) * t); tags.append(i)
    pts = np.array(pts); tags = np.array(tags)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V, idx = [], []
    for j, (r, z) in enumerate(pts):
        if r < 1e-9:
            idx.append([len(V)] * n); V.append([0, 0, z]); continue
        rr = np.full(n, r)
        for T in threads:
            if T["seg"][0] <= tags[j] <= T["seg"][1] and T["z0"] - 1e-9 <= z <= T["z1"] + 1e-9:
                fade = np.clip(min(z - T["z0"], T["z1"] - z) / T.get("ramp", 1e-9), 0, 1)
                u = z / T["p"] - th / (2 * np.pi) + T.get("phase", 0.0)
                if "clear" in T:   # female bore with a uniform clearance (thread_envelope); profile r = male root
                    env, top = thread_envelope(T["h"], T["p"], T["clear"])
                    rr = rr + fade * env(u) + (1 - fade) * T.get("rest", 0.0) * top
                else:
                    rr = rr + T["h"] * (fade * _thread_f(u) + (1 - fade) * T.get("rest", 0.0))
        idx.append(list(range(len(V), len(V) + n)))
        V.extend(np.c_[rr * np.cos(th), rr * np.sin(th), np.full(n, z)].tolist())
    F = []
    m = len(pts)
    for j in range(m):
        A, B = idx[j], idx[(j + 1) % m]
        for k in range(n):
            k1 = (k + 1) % n
            a0, a1, b0, b1 = A[k], A[k1], B[k], B[k1]
            if a0 == a1 and b0 == b1: continue
            if a0 == a1: F.append([a0, b0, b1]); continue
            if b0 == b1: F.append([a0, b0, a1]); continue
            F.append([a0, b0, b1]); F.append([a0, b1, a1])
    mesh = trimesh.Trimesh(np.array(V), np.array(F), process=True)
    mesh.merge_vertices(); mesh.update_faces(mesh.nondegenerate_faces()); mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def tube_profile(r_out, r_in, z0, z1, bottom):
    """open-top cup: outer wall, flat bottom of thickness `bottom`."""
    return [(0, z0), (r_out, z0), (r_out, z1), (r_in, z1), (r_in, z0 + bottom), (0, z0 + bottom)]


def check(mesh, name=""):
    ok = mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    print(f"{name:14s} V={len(mesh.vertices):6d} F={len(mesh.faces):6d} watertight={mesh.is_watertight} "
          f"volume={mesh.volume*1e6:9.2f} cm3 ext={np.round(mesh.extents, 4)}" + ("" if ok else "  <-- BAD"))
    return ok
