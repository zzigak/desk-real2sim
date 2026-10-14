"""CPU IK scan: where can a YAM hand take the monitor panel by a side edge (fingers across its 6 cm thickness)?"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
from scipy.spatial.transform import Rotation as Rot
import robot_desk as R, scene as S
from yam_cart import add_cart, quat_from_mat
from genesis.utils.misc import tensor_to_array
o = S.SC["objects"]["monitor_panel"]
Rp = Rot.from_euler("xyz", o["euler"], degrees=True).as_matrix(); P = np.asarray(o["pos"])
W = lambda p: Rp @ np.asarray(p) + P
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
for side in ("left", "right"):
    for edge in (-1, 1):                          # -1: left edge (approach +x), +1: right edge (approach -x)
        for z_rel in (-0.215, -0.19, -0.16, -0.12, -0.06):
            hits = []
            for a_deg in (0, 15, 30, 45, 60):         # approach turned from along the width towards coming in from the front
                ax_ = Rp[:, 0] * (-edge); fwd = -Rp[:, 1]
                a = math.radians(a_deg)
                ap = math.cos(a) * ax_ - math.sin(a) * fwd       # pointing inward and backwards (+y)
                ap /= np.linalg.norm(ap)
                E = W((edge * (0.375 - 0.022), -0.053, z_rel))
                for sgn in (1, -1):
                    cl = sgn * np.cross([0, 0, 1.0], ap); cl /= np.linalg.norm(cl)
                    qq = quat_from_mat(np.column_stack([cl, np.cross(ap, cl), ap]))
                    for bx in (-0.3, -0.15, 0.0, 0.15, 0.3, 0.45):
                        init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
                        ok = True
                        for p in (E, E - ap * 0.09):
                            q, err = cart.ik(side, p, quat=qq, init_qpos=init, return_error=True, max_samples=40)
                            d = cart.arm_dofs[side]
                            if not (np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 0.03 and np.all(q >= lo[d] - 1e-3) and np.all(q <= hi[d] + 1e-3)):
                                ok = False; break
                        if ok:
                            hits.append((a_deg, sgn, bx)); break
                    if hits and hits[-1][0] == a_deg:
                        break
            print(f"{side:5s} edge {'L' if edge < 0 else 'R'} z_rel {z_rel:+.3f}: {hits if hits else '-'}", flush=True)
