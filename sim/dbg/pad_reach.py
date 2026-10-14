"""CPU IK: right hand poses for the pad edge push (top-down) and the front pinch of a right-edge overhang."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart
from genesis.utils.misc import tensor_to_array
H = 0.775
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
def ok(side, p, yaw, tilt, bx):
    init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
    q, err = cart.ik(side, p, yaw=yaw, tilt=tilt, init_qpos=init, return_error=True, max_samples=60)
    dd = cart.arm_dofs[side]
    return np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 0.03 and np.all(q >= lo[dd] - 1e-3) and np.all(q <= hi[dd] + 1e-3)
for bx in (0.25, 0.35, 0.45, 0.55):
    r = []
    for name, side, p, yaws, tilt in (("push start", "right", (0.338, -0.17, H + 0.0005), (math.pi / 2, -math.pi / 2), 0.0),
                                      ("push end", "right", (0.418, -0.17, H + 0.0005), (math.pi / 2, -math.pi / 2), 0.0),
                                      ("pinch pre", "right", (0.655, -0.32, H - 0.002), (math.pi / 2, -math.pi / 2), math.pi / 2),
                                      ("pinch", "right", (0.655, -0.23, H - 0.002), (math.pi / 2, -math.pi / 2), math.pi / 2),
                                      ("lift", "right", (0.62, -0.23, H + 0.25), (math.pi / 2, -math.pi / 2), math.pi / 2)):
        y = next((y for y in yaws if ok(side, p, y, tilt, bx)), None)
        r.append(f"{name}: {'%.2f' % y if y is not None else 'NO'}")
    print(f"base {bx:.2f}  " + " | ".join(r), flush=True)
