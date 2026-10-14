"""One Nyx frame of the static desk from the p09 photo camera, to check materials (textures, vertex colours, glass)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nyx_cam
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import scene as S
import nyx_assets
cc = S.SC["cameras"]["p09"]
holder = {}
def extra(scene):
    holder["cam"] = nyx_cam.add_nyx(scene, cc["pos"], cc["lookat"], res=(960, 720), fov=cc["fov"], spp=int(sys.argv[1]) if len(sys.argv) > 1 else 64,
                                    lights=nyx_cam.desk_lights((0.0, -0.1, 0.8)))
    return {}
scene, E, C = S.build(skip=(), cameras=(), extra=extra, urdf_override=nyx_assets.nyx_urdfs())
scene.build()
t0 = time.time()
scene.step()
fr = nyx_cam.Frames(os.path.join(S.ROOT, "out", "nyx_test"))
fr.capture(holder["cam"]); print("render", time.time() - t0, "s", flush=True)
