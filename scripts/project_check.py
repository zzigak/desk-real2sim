"""Project placed meshes (world frame) back into a photo via the VGGT camera to eyeball poses."""
import json, sys, numpy as np, cv2, trimesh
sys.path.insert(0, "scripts")
from pix2table import cam_table, K, S_METRIC as S
TC = np.array([(-0.405 + 0.51) / 2, (-0.152 + 0.40) / 2]) * S
H = 0.775
def world_to_pix(photo, Pw):
    Pt = Pw.copy(); Pt[:, :2] += TC; Pt[:, 2] -= H; Pt /= S                    # world -> table frame (units)
    Rc, t = cam_table(photo)                                                  # camera in table frame
    Pc = (Pt - t) @ Rc                                                        # table -> camera (OpenCV)
    uv = (K[photo] @ Pc.T).T; return uv[:, :2] / uv[:, 2:3], Pc[:, 2]
def place(V, yaw, xy, z):
    a = np.radians(yaw); R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return V @ R.T + np.array([xy[0], xy[1], H + z])
if __name__ == "__main__":
    photo = sys.argv[1]; fits = json.load(open("assets/sam3d_fit.json"))
    img = cv2.imread(f"photos/{photo}.jpg")
    for name, f in fits.items():
        m = trimesh.load(f"assets/{name}_vis.ply", process=False)
        V = place(m.vertices, f["yaw"], f["xy"], f["z0"]); C = np.asarray(m.visual.vertex_colors)[:, :3]
        uv, z = world_to_pix(photo, V); o = np.argsort(-z)
        for (u, v), c in zip(uv[o].astype(int), C[o]):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]: img[v, u] = (0, 0, 255) if name == "mouse" else c[::-1]
    for name, yaw in (("mouse_pca", -65.3),):
        pass
    cv2.imwrite(f"work/proj_{photo}.jpg", cv2.resize(img, None, fx=0.5, fy=0.5))
