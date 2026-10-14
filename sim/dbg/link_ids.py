import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genesis as gs
gs.init(backend=gs.cpu, logging_level="error")
import robot_desk as R, scene as S
from yam_cart import add_cart
holder = {}
def extra(scene):
    holder["cart"] = add_cart(scene, gs, R.VARIANT, ipc=True, pos=R.cart_pos(0.30)); return {}
scene, E, C = S.build(dt=R.DT, skip=tuple(S.SC["deformables"]), cameras=(), extra=extra, coupler_kw=R.CONTACT, urdf_override=R.robot_assets())
for n, e in list(E.items()) + [("cart", holder["cart"].entity)]:
    if hasattr(e, "links"):
        print(n, [(l.idx, l.name) for l in e.links if n != "cart" or l.name in holder["cart"].coup_links()])
