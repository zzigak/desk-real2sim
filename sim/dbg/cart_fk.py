import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build()
for n in ("left_gripper", "left_tip_left", "right_gripper", "right_tip_right"):
    l = cart.entity.get_link(n); print("q=0", n, np.round(np.array(l.get_pos()), 3))
cart.after_build(home=False)
print("cart base", R.cart_pos(0.0))
