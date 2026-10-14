"""Lowest point and x-extent of the right hand's collision geoms (tip pads + palm) over a recorded run, by CPU FK."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R
from yam_cart import add_cart
from genesis.utils.misc import tensor_to_array
Z = np.load(sys.argv[1]); t0, t1 = float(sys.argv[2]), float(sys.argv[3])
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
scene.build(); cart.after_build(home=False)
e = cart.entity
links = [e.get_link(f"right_{n}") for n in ("tip_left", "tip_right", "gripper")]
for i in range(len(Z["t"])):
    if not (t0 <= Z["t"][i] <= t1) or i % 6: continue
    e.set_qpos(Z["qpos"][i])
    p, _ = cart.tcp_pose("right")
    out = []
    for L in links:
        vs = np.concatenate([tensor_to_array(g.get_verts()) for g in L.geoms]) if L.geoms else np.zeros((1, 3))
        out.append(f"{L.name.split('_', 1)[1]}: zmin {1000 * (vs[:, 2].min() - 0.775):6.1f} x {vs[:, 0].min():.3f}..{vs[:, 0].max():.3f}")
    print(f"t {Z['t'][i]:5.2f} tcp z {1000 * (p[2] - 0.775):5.1f} x {p[0]:.3f} | " + " | ".join(out), flush=True)
