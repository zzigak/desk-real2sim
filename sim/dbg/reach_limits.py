"""CPU IK reach with joint-limit check, per YAM variant, for the fit-demo grasp poses."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R, scene as S
from yam_cart import add_cart
from genesis.utils.misc import tensor_to_array
H = S.H
T = {"monitor panel bottom front (R)": ("right", (0.15, 0.27, 0.98), 0.0, math.pi / 2),
     "monitor panel bottom front (L)": ("left", (0.10, 0.27, 0.98), 0.0, math.pi / 2),
     "stick top (R, top-down)": ("right", (0.53, 0.20, 0.99), 0.0, 0.0),
     "stick top (R, side)": ("right", (0.53, 0.19, 0.99), 0.0, math.pi / 2),
     "bottle side (R)": ("right", (0.568, 0.14, H + 0.10), 0.0, math.pi / 2),
     "cloth hanging corner (L)": ("left", (-0.42, -0.39, H - 0.12), 0.0, math.pi / 2),
     "cloth corner on desk (L)": ("left", (-0.30, -0.25, H + 0.02), 0.0, 0.0),
     "paper corner (R)": ("right", (0.24, -0.33, H + 0.01), 0.0, 0.0),
     "pad front edge (R)": ("right", (0.49, -0.24, H + 0.01), 0.0, 0.0),
     "tin lid side (R)": ("right", (0.33, 0.160, H + 0.115), 0.0, math.pi / 2),
     "tin body side (L)": ("left", (0.33, 0.160, H + 0.045), 0.0, math.pi / 2),
     "tin lid top (R)": ("right", (0.33, 0.148, H + 0.14), 0.0, 0.0),
     "jar mouth top (R)": ("right", (0.531, -0.296, H + 0.20), 0.0, 0.0),
     "marker cap top (L)": ("left", (0.07, -0.213, H + 0.012), 1.4, 0.0)}
for variant in sys.argv[1:]:
    scene = gs.Scene(show_viewer=False)
    cart = add_cart(scene, gs, variant, ipc=False, pos=R.cart_pos(0.0))
    scene.build(); cart.after_build()
    lo, hi = [tensor_to_array(x) for x in cart.entity.get_dofs_limit()]
    for k, (side, p, yaw, tilt) in T.items():
        best = None
        for bx in (-0.3, -0.15, 0.0, 0.15, 0.3, 0.45):
            for y in (yaw, yaw + math.pi):
                init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
                q, err = cart.ik(side, p, yaw=y, tilt=tilt, init_qpos=init, return_error=True, max_samples=60)
                d = cart.arm_dofs[side]
                ok_lim = bool(np.all(q >= lo[d] - 1e-3) and np.all(q <= hi[d] + 1e-3))
                ok = np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 0.03
                if ok and ok_lim:
                    best = (bx, round(y, 2)); break
            if best: break
        print(f"{variant:8s} {k:20s}: {'reachable at base %.2f yaw %.2f' % best if best else 'NOT reachable within joint limits'}", flush=True)
