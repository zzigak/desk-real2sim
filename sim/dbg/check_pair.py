import json, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
A = "/home/ziga/projects/desk-real2sim/assets"; SC = json.load(open(f"{A}/scene.json"))
def world(pos, euler):
    M = np.eye(4); M[:3, :3] = Rot.from_euler("xyz", euler, degrees=True).as_matrix(); M[:3, 3] = pos; return M
st = trimesh.load(f"{A}/urdf/stand_col.obj", process=False); o = SC["objects"]["laptop_stand"]; st.apply_transform(world(o["pos"], o["euler"]))
cl = trimesh.load(f"{A}/urdf/cloth.obj", process=False); d = SC["deformables"]["cloth"]; cl.apply_transform(world(d["pos"], d["euler"]))
print("stand watertight", st.is_watertight, "is_volume", st.is_volume, "bodies", len(st.split(only_watertight=False)))
sel = (cl.vertices[:, 0] < -0.30) & (cl.vertices[:, 1] > -0.09)
print("cloth verts near the corner:\n", np.round(cl.vertices[sel][:8], 4))
print("contains:", st.contains(cl.vertices[sel]).sum(), "of", sel.sum())
