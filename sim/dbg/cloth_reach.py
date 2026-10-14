"""CPU IK: side pinches of the hanging cloth (approach along +-x, turned towards +y), by cart base x."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart, quat_from_mat
from genesis.utils.misc import tensor_to_array
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
def ok(side, p, qq, bx):
    init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
    q, err = cart.ik(side, p, quat=qq, init_qpos=init, return_error=True, max_samples=50)
    dd = cart.arm_dofs[side]
    return np.linalg.norm(err[:3]) < 4e-3 and np.linalg.norm(err[3:]) < 0.05 and np.all(q >= lo[dd] - 1e-3) and np.all(q <= hi[dd] + 1e-3)
for side, ex, inward in (("left", -0.342, 1.0), ("right", -0.136, -1.0)):
    for y in (-0.385, -0.40, -0.415):
        for bx in (-0.40, -0.30, -0.24, -0.15, -0.05):
            hits = []
            for ang in (0, 20, 40, -20):
                a = math.radians(ang); ap = np.array([inward * math.cos(a), math.sin(a), 0.0])
                for sgn in (1.0, -1.0):
                    cl = sgn * np.cross([0, 0, 1.0], ap); qq = quat_from_mat(np.column_stack([cl, np.cross(ap, cl), ap]))
                    p = np.array([ex, y, 0.715]) + ap * 0.025
                    if ok(side, p, qq, bx) and ok(side, p - ap * 0.07, qq, bx) and ok(side, p - ap * 0.035, qq, bx):
                        hits.append(ang); break
            print(f"{side:5s} y {y:+.3f} base {bx:+.2f}: {hits or '-'}", flush=True)
