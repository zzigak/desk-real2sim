"""CPU IK: right hand top-down over the jar mouth, by cart base x, height and yaw (joint limits checked)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart
from genesis.utils.misc import tensor_to_array
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
jx, jy = 0.5315, -0.2965
for bx in (0.15, 0.25, 0.35, 0.45, 0.55):
    row = []
    for z in (0.93, 0.95, 0.97, 0.995, 1.02):
        best = None
        for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
            q, err = cart.ik("right", (jx, jy, z), yaw=yaw, init_qpos=init, return_error=True, max_samples=60)
            dd = cart.arm_dofs["right"]
            e = np.linalg.norm(err[:3]) * 1000
            if e < 2 and np.linalg.norm(err[3:]) < 0.03 and np.all(q >= lo[dd] - 1e-3) and np.all(q <= hi[dd] + 1e-3):
                best = f"{yaw:+.2f}"; break
        row.append(f"z{z:.3f}:{best or 'NO'}")
    print(f"base {bx:.2f}  " + "  ".join(row), flush=True)
