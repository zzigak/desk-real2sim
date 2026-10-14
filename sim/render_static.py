"""Visual check without physics: every object fixed at its start pose, rendered from the photo cameras."""
import argparse, json, os
import numpy as np, genesis as gs
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); A = os.path.join(ROOT, "assets")
SC = json.load(open(f"{A}/scene.json"))
ap = argparse.ArgumentParser(); ap.add_argument("--cams", nargs="*", default=["p10", "p09", "p08", "p07"])
ap.add_argument("--out", default=os.path.join(ROOT, "out", "static")); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=False),
                 vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.45, 0.45, 0.45), background_color=(0.86, 0.85, 0.81),
                                                   lights=[dict(type="directional", dir=(0.3, 0.6, -1.0), color=(1.0, 0.98, 0.94), intensity=3.0)]))
scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.30, 0.30, 0.32)))
for name, o in SC["objects"].items():
    surf = gs.surfaces.Default(color=tuple(o["rgba"])) if "rgba" in o else gs.surfaces.Default()
    scene.add_entity(gs.morphs.URDF(file=f"{A}/{o['urdf']}", pos=o["pos"], euler=o["euler"], fixed=True, convexify=False, decimate=False,
                                    watertighten=0, collision=False), surface=surf)
for name, d in SC["deformables"].items():
    scene.add_entity(gs.morphs.Mesh(file=f"{A}/{d['mesh']}", fixed=True, collision=False, convexify=False, decimate=False, watertighten=0),
                     surface=gs.surfaces.Default(color=tuple(d["rgba"]), double_sided=True) if hasattr(gs.surfaces.Default(), "double_sided") else gs.surfaces.Default(color=tuple(d["rgba"])))
cams = {c: scene.add_camera(res=tuple(SC["cameras"][c]["res"]), pos=SC["cameras"][c]["pos"], lookat=SC["cameras"][c]["lookat"],
                            up=SC["cameras"][c]["up"], fov=SC["cameras"][c]["fov"], GUI=False) for c in a.cams}
scene.build()
import cv2
for c, cam in cams.items():
    rgb = cam.render()[0]
    cv2.imwrite(os.path.join(a.out, f"static_{c}.png"), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)); print("wrote", c)
