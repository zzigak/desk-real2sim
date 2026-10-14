"""TCP positions of a recorded robot run (states npz qpos) by CPU forward kinematics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart
Z = np.load(sys.argv[1]); side = sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 10
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build(home=False)
for i in range(0, len(Z["t"]), step):
    cart.entity.set_qpos(Z["qpos"][i])
    p, q = cart.tcp_pose(side)
    print(f"t {Z['t'][i]:6.2f} tcp {np.round(p, 4)}  z-H {1000 * (p[2] - 0.775):6.1f} mm", flush=True)
