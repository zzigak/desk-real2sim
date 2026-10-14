"""Photo cameras (VGGT) -> world frame (pos, lookat, up, vfov) for rendering the sim from the photo viewpoints."""
import json, sys, numpy as np
sys.path.insert(0, "scripts")
from pix2table import cam_table, K, S_METRIC as S
TC = np.array([(-0.405 + 0.51) / 2, (-0.152 + 0.40) / 2]) * S; H = 0.775
cams = {}
for ph in ("p09", "p10", "p08", "p07", "p01", "p03", "p06"):
    Rc, t = cam_table(ph)                         # columns of Rc: camera x, y, z axes in table frame (OpenCV)
    pos = t * S; pos[:2] -= TC; pos[2] += H
    fwd, down = Rc[:, 2], Rc[:, 1]
    k = K[ph]; vfov = float(np.degrees(2 * np.arctan(k[1, 2] / k[1, 1])))
    cams[ph] = dict(pos=pos.tolist(), lookat=(pos + 0.8 * fwd).tolist(), up=(-down).tolist(), fov=vfov, res=[1288, 966])
    print(ph, np.round(pos, 3), "fwd", np.round(fwd, 2), "vfov", round(vfov, 1))
sc = json.load(open("assets/scene.json")); sc["cameras"] = cams; json.dump(sc, open("assets/scene.json", "w"), indent=1)
