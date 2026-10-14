"""Diagnostic: cap upright above the jar on its axis, height released (velocity-limited descent); where does it stop,
and which surfaces are closest there."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, trimesh
import genesis as gs
import scene as S, ipc_hands
from scipy.spatial import cKDTree
SPEC = json.load(open(os.path.join(S.A, "specs.json")))["jar"]
gs.init(backend=gs.gpu, logging_level="warning")
S.SC["objects"]["jar_lid"]["euler"] = [0, 0, 0]
jar = np.array(S.SC["objects"]["jar"]["pos"]); seated = jar + [0, 0, SPEC["lid_seated_z"]]
S.SC["objects"]["jar_lid"]["pos"] = (seated + [0, 0, 0.030]).tolist()           # start 30 mm above seated, upright
scene, E, C = S.build(skip=tuple(k for k in list(S.SC["objects"]) + list(S.SC["deformables"]) if k not in ("table", "jar", "jar_lid")), cameras=())
def traj(t, Tc):
    M = np.eye(4); M[:3, 3] = [seated[0], seated[1], Tc[2, 3] - 0.0003]; return M
h = ipc_hands.Hand("cap", E["jar_lid"].base_link, traj); ipc_hands.install([h])
scene.build()
for i in range(150):
    h.t = i * 0.01; scene.step()
    if i % 15 == 0: print(f"t={i*0.01:.2f} cap z above seated {1000*(h.T[2,3]-seated[2]):.2f} mm  xy err {1000*np.hypot(*(h.T[:2,3]-seated[:2])):.2f} mm", flush=True)
cap = trimesh.load(os.path.join(S.A, "jar_lid_col.obj")); cap.apply_transform(h.T)
jm = trimesh.load(os.path.join(S.A, "jar_col.obj")); jm.apply_translation(S.link_pos(E["jar"])[0])
d, j = cKDTree(jm.vertices).query(cap.vertices); k = np.argsort(d)[:5]
for i in k:
    v = cap.vertices[i] - seated
    print(f"closest cap vertex {1000*d[i]:.2f} mm at r={1000*np.hypot(v[0], v[1]):.2f} mm, z(rel seated)={1000*v[2]:.2f} mm; jar vertex r={1000*np.hypot(*(jm.vertices[j[i]][:2]-seated[:2])):.2f} z(rel jar base)={1000*(jm.vertices[j[i]][2]-jar[2]):.2f}")
