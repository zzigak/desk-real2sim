import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R, scene as S
from yam_cart import add_cart
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
H = S.H
def ik(side, p, yaw, tilt, bx):
    init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
    q, err = cart.ik(side, p, yaw=yaw, tilt=tilt, init_qpos=init, return_error=True, max_samples=60)
    return q, np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 0.03
for bx in (0.3, 0.45):
    for (x, y, z) in ((0.40, -0.22, 0.18), (0.40, -0.12, 0.22), (0.45, -0.18, 0.25), (0.35, -0.15, 0.15)):
        for tilt in (math.pi / 2, 2.4, math.pi, -math.pi / 2, -2.4):
            for yaw in (0.0, math.pi):
                q, ok = ik("right", (x, y, H + z), yaw, tilt, bx)
                if ok: print(f"base {bx:.2f} right at ({x},{y},{z}) tilt {tilt:+.2f} yaw {yaw:+.2f}: OK q6 {q[5]:+.2f}")
        for yaw in (math.pi / 2, -math.pi / 2):
            q, ok = ik("left", (x, y, H + z + 0.03), yaw, 0.0, bx)
            if ok: print(f"base {bx:.2f}  left top-down at ({x},{y},{z}+0.03) yaw {yaw:+.2f}: OK")
