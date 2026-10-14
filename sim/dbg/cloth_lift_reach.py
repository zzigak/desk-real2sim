"""CPU IK: where can the left hand lift the cloth to after backing out in front of the desk (cart at base x)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart, quat_from_mat, down_quat
from genesis.utils.misc import tensor_to_array
H = 0.775
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
ap = np.array([1.0, 0, 0]); cl = -np.cross([0, 0, 1.0], ap); side_q = quat_from_mat(np.column_stack([cl, np.cross(ap, cl), ap]))
Q = dict(side=side_q, down=down_quat(math.pi / 2, 0.0), down_t5=down_quat(math.pi / 2, 0.5), down_y0_t5=down_quat(0.0, 0.5), down_t9=down_quat(math.pi / 2, 0.9))
def ok(p, q, bx):
    init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
    qq, err = cart.ik("left", p, quat=q, init_qpos=init, return_error=True, max_samples=50)
    dd = cart.arm_dofs["left"]
    return np.linalg.norm(err[:3]) < 5e-3 and np.linalg.norm(err[3:]) < 0.06 and np.all(qq >= lo[dd] - 1e-3) and np.all(qq <= hi[dd] + 1e-3)
for bx in (-0.14, -0.04, 0.06):
    for z in (0.12, 0.18, 0.24):
        row = []
        for x in (-0.38, -0.30, -0.22):
            for y in (-0.45, -0.36, -0.27):
                hit = [k for k, q in Q.items() if ok((x, y, H + z), q, bx)]
                row.append(f"({x:+.2f},{y:+.2f}):{','.join(h[:6] for h in hit) or '-'}")
        print(f"base {bx:+.2f} z+{z:.2f}  " + "  ".join(row), flush=True)
