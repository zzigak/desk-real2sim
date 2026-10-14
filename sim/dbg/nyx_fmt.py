import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nyx_cam
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import scene as S
f = sys.argv[1]
scene = gs.Scene(show_viewer=False)
o = S.SC["objects"]["tin_lid"]
scene.add_entity(gs.morphs.URDF(file=f, pos=(0, 0, 0.8), fixed=True, convexify=False, decimate=False, watertighten=0, merge_fixed_links=False),
                 material=gs.materials.Rigid(rho=o["rho"]))
cam = nyx_cam.add_nyx(scene, (0.25, -0.25, 0.95), (0, 0, 0.81), spp=32, lights=nyx_cam.desk_lights((0, 0, 0.8), scale=0.4))
scene.build(); scene.step()
nyx_cam.Frames(os.path.join(os.path.dirname(f), os.path.basename(f) + "_render")).capture(cam)
print("OK", f, flush=True)
