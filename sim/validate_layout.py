"""Static start-state check: every collision mesh / deformable at its world pose, all pairs, min distance.
IPC refuses to start on overlap (and contact closer than d_hat starts with barrier forces), so this runs before any sim."""
import json, os, sys, itertools
import xml.etree.ElementTree as ET
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); A = f"{ROOT}/assets"
SC = json.load(open(f"{A}/scene.json"))

def urdf_meshes(path, qpos):
    """collision meshes of each link in the URDF root frame, with revolute joints set to qpos."""
    r = ET.parse(path).getroot(); d = os.path.dirname(path)
    links = {}
    for ln in r.findall("link"):
        ms = []
        for c in ln.findall("collision"):
            o = np.array([float(v) for v in c.find("origin").get("xyz").split()]); g = c.find("geometry")
            if g.find("box") is not None:
                m = trimesh.creation.box([float(v) for v in g.find("box").get("size").split()])
            else:
                m = trimesh.load(os.path.join(d, g.find("mesh").get("filename")), process=False)
            m.apply_translation(o); ms.append(m)
        links[ln.get("name")] = ms
    T = {r.find("link").get("name"): np.eye(4)}
    for j in r.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        o = np.array([float(v) for v in j.find("origin").get("xyz").split()]); ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        M = np.eye(4); M[:3, 3] = o
        rpy = [float(v) for v in j.find("origin").get("rpy", "0 0 0").split()]; M[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
        R = np.eye(4); R[:3, :3] = Rot.from_rotvec(ax * qpos.get(j.get("name"), 0.0)).as_matrix()
        T[c] = T[p] @ M @ R
    out = {}
    for n, ms in links.items():
        for i, m in enumerate(ms):
            mm = m.copy(); mm.apply_transform(T[n]); out[f"{n}{i}"] = mm
    return out

def world(pos, euler):
    M = np.eye(4); M[:3, :3] = Rot.from_euler("xyz", euler, degrees=True).as_matrix(); M[:3, 3] = pos; return M

bodies = {}
for k, o in SC["objects"].items():
    for ln, m in urdf_meshes(f"{A}/{o['urdf']}", o.get("init_qpos", {})).items():
        m.apply_transform(world(o["pos"], o["euler"])); bodies[f"{k}/{ln}"] = (m, k)
for k, d in SC["deformables"].items():
    m = trimesh.load(f"{A}/{d['mesh']}", process=False); m.apply_transform(world(d["pos"], d["euler"])); bodies[k] = (m, k)
# contact thickness for shells (mid-surface): thickness + d_hat like libuipc's distance check
THICK = dict(paper_top=1e-4, paper_bottom=1e-4, mousepad=0.003, cloth=0.001)
names = list(bodies); bad = []
samples = {n: (m.sample(60000) if len(m.faces) else m.vertices) for n, (m, _) in bodies.items()}
trees = {n: cKDTree(np.vstack([samples[n], bodies[n][0].vertices])) for n in names}
for a, b in itertools.combinations(names, 2):
    (ma, ka), (mb, kb) = bodies[a], bodies[b]
    if ka == kb and a.split('/')[1][:-1] == b.split('/')[1][:-1]: continue   # same link
    lo = np.maximum(ma.bounds[0], mb.bounds[0]); hi = np.minimum(ma.bounds[1], mb.bounds[1])
    if np.any(lo > hi + 0.003): continue
    da, _ = trees[b].query(np.vstack([samples[a], ma.vertices])); db, _ = trees[a].query(np.vstack([samples[b], mb.vertices]))
    dmin = min(da.min(), db.min())
    pen = 0
    for (m1, n1), pts in (((ma, a), np.vstack([samples[b], mb.vertices])), ((mb, b), np.vstack([samples[a], ma.vertices]))):
        if m1.is_watertight and m1.is_volume:
            sel = np.all((pts >= m1.bounds[0]) & (pts <= m1.bounds[1]), 1)
            if sel.any(): pen += int(m1.contains(pts[sel][:20000]).sum())
    need = THICK.get(ka, 0) / 2 + THICK.get(kb, 0) / 2 + 0.0005
    flag = "PENETRATES" if pen else ("TOO CLOSE" if dmin < need else "ok")
    if flag != "ok" or dmin < 0.004:
        print(f"{a:22s} {b:22s} min_dist {dmin*1000:6.2f} mm  inside_samples {pen:5d}  need>{need*1000:.2f}  {flag}")
    if flag != "ok": bad.append((a, b))
print("BAD pairs:", len(bad))
