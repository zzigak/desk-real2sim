"""Minimal Nyx scenes to find what crashes the renderer: python nyx_min.py <ipc|plain> <parts...>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nyx_cam
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import scene as S
mode, parts = sys.argv[1], sys.argv[2:]
kw = {}
if mode == "ipc":
    kw["coupler_options"] = gs.options.IPCCouplerOptions(contact_d_hat=1e-3)
    kw["rigid_options"] = gs.options.RigidOptions(enable_collision=False)
scene = gs.Scene(show_viewer=False, **kw)
for n in parts:
    o = S.SC["objects"][n]
    mat = gs.materials.Rigid(coup_type="ipc_only", rho=o["rho"]) if mode == "ipc" else gs.materials.Rigid(rho=o["rho"])
    scene.add_entity(gs.morphs.URDF(file=os.path.join(S.A, o["urdf"]), pos=tuple(o["pos"]), euler=tuple(o["euler"]), fixed=True,
                                    convexify=False, decimate=False, watertighten=0, merge_fixed_links=False), material=mat)
cam = nyx_cam.add_nyx(scene, (0.6, -0.9, 1.3), (0.2, -0.1, 0.8), spp=16)
scene.build(); scene.step()
nyx_cam.Frames(os.path.join(S.ROOT, "out", "nyx_min", mode + "_" + "_".join(parts))).capture(cam)
print("OK", mode, parts, flush=True)
