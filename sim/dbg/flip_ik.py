"""CPU IK check for the jar-lid flip: right hand horizontal grasp across the upside-down lid, then roll 180 deg."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R, scene as S
from yam_cart import add_cart
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build()
H = S.H; lid = S.SC["objects"]["jar_lid"]["pos"]
def ik(side, p, yaw, tilt, bx, seed=None):
    init = cart.q_target.copy(); init[cart.base_dofs] = (bx, 0, 0)
    if seed is not None: init[cart.arm_dofs[side]] = seed
    q, err = cart.ik(side, p, yaw=yaw, tilt=tilt, init_qpos=init, return_error=True)
    return q, np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 0.03
for bx in (0.15, 0.3, 0.45):
    for side in ("right", "left"):
        for z in (0.016, 0.10):
            for yaw in (0.0, math.pi, -math.pi / 2, math.pi / 2):
                q, ok = ik(side, (lid[0], lid[1], H + z), yaw, math.pi / 2, bx)
                print(f"base {bx:.2f} {side:5s} z {z:.3f} yaw {yaw:+.2f}: {'OK ' if ok else 'no '} q6 {q[5]:+.2f}")
