"""CPU IK check of the bottle demo's stick drop and push poses (right arm, base x 0.268)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R, scene as S
from yam_cart import add_cart
from genesis.utils.misc import tensor_to_array
H = S.H; bx = 0.268; b = np.array([0.568, 0.168])
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
T = {f"drop {xy}": [((xy[0], xy[1], H + 0.181 + dz), (0.0, math.pi)) for dz in (0.0, 0.10)] for xy in ((0.56, -0.08), (0.53, -0.10), (0.58, -0.04))}
a = (b[0] + 0.0535, b[1], H + 0.13)
T["push a/pre/b"] = [(a, (math.pi / 2, -math.pi / 2)), ((a[0], a[1] - 0.10, a[2]), (math.pi / 2, -math.pi / 2)), ((a[0] - 0.15, a[1], a[2]), (math.pi / 2, -math.pi / 2))]
for k, pts in T.items():
    res = []
    for p, yaws in pts:
        ok = False
        for y in yaws:
            init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
            q, err = cart.ik("right", p, yaw=y, tilt=math.pi / 2, init_qpos=init, return_error=True, max_samples=60)
            d = cart.arm_dofs["right"]
            if np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 0.03 and np.all(q >= lo[d] - 1e-3) and np.all(q <= hi[d] + 1e-3):
                ok = f"yaw {y:.2f}"; break
        res.append(ok or f"NO (err {np.round(np.linalg.norm(err[:3]) * 1000, 1)} mm)")
    print(k, res, flush=True)
