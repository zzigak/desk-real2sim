"""Re-render a recorded robot run (robot_demos.py writes <demo>_states.npz) without re-simulating: a collision-free
Genesis scene with the desk (Nyx-renderable visuals, nyx_assets.py) and the YAM cart is posed frame by frame from the
recorded robot joint angles and libuipc object poses, and drawn by the Nyx path tracer (or the rasterizer, --raster).
Shells (paper, cloth, pad) are drawn as static meshes in their recorded first-frame shape.

  python sim/nyx_replay.py out/robot/marker/marker_states.npz --cam 0.42 -0.42 0.92 0.10 -0.22 0.81 --fov 40
"""
import argparse, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh

import nyx_cam
import genesis as gs
import scene as S
import nyx_assets
import robot_desk as R

ap = argparse.ArgumentParser()
ap.add_argument("states"); ap.add_argument("--cam", type=float, nargs=6, required=True)
ap.add_argument("--fov", type=float, default=40.0); ap.add_argument("--res", type=int, nargs=2, default=(960, 720))
ap.add_argument("--spp", type=int, default=64); ap.add_argument("--raster", action="store_true")
ap.add_argument("--every", type=int, default=1); ap.add_argument("--out", default=None)
a = ap.parse_args()
Z = np.load(a.states)
out = a.out or os.path.splitext(a.states)[0].replace("_states", "") + ("_raster" if a.raster else "_nyx")
gs.init(backend=gs.cpu if a.raster else gs.gpu, logging_level="warning")

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1 / 30, gravity=(0, 0, 0)),
                 rigid_options=gs.options.RigidOptions(enable_collision=False),
                 vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.45, 0.45, 0.45), background_color=(0.86, 0.85, 0.81),
                                                   lights=[dict(type="directional", dir=(0.3, 0.6, -1.0), color=(1.0, 0.98, 0.94), intensity=3.0)]),
                 show_viewer=False)
scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.30, 0.30, 0.32)))
over = nyx_assets.nyx_urdfs(R.robot_assets())
ents = {}
names = [k[len("pose__"):] for k in Z.files if k.startswith("pose__")]
for n in S.SC["objects"]:
    o = S.SC["objects"][n]
    surf = gs.surfaces.Default(color=tuple(o["rgba"])) if "rgba" in o else gs.surfaces.Default()
    ents[n] = scene.add_entity(gs.morphs.URDF(file=over[n]["urdf"], pos=tuple(o["pos"]), euler=tuple(o["euler"]), fixed=n not in names,
                                              convexify=False, decimate=False, watertighten=0, merge_fixed_links=False),
                               material=gs.materials.Rigid(gravity_compensation=1.0), surface=surf)
for k in Z.files:
    if k.startswith("shell__"):
        n = k[len("shell__"):]; d = S.SC["deformables"][n]
        m = trimesh.load(os.path.join(S.A, d["mesh"]), process=False); m.vertices = Z[k][0]
        f = os.path.join(nyx_assets.NU, f"{n}_replay.obj"); m.export(f)
        scene.add_entity(gs.morphs.Mesh(file=f, fixed=True, convexify=False, decimate=False),
                         surface=gs.surfaces.Default(color=tuple(d["rgba"][:3])))
from yam_cart import add_cart
cart = add_cart(scene, gs, R.VARIANT, ipc=False, pos=R.cart_pos(0.0))
c = a.cam
if a.raster:
    cam = scene.add_camera(res=tuple(a.res), pos=tuple(c[:3]), lookat=tuple(c[3:]), fov=a.fov, GUI=False)
else:
    cam = nyx_cam.add_nyx(scene, c[:3], c[3:], res=a.res, fov=a.fov, spp=a.spp)
scene.build()
cart.after_build(home=False)
fr = nyx_cam.Frames(out, fps=30 // a.every)
for i in range(0, len(Z["t"]), a.every):
    q = Z["qpos"][i]
    cart.entity.set_qpos(q); cart.entity.zero_all_dofs_velocity(); cart.entity.control_dofs_position(q)
    for n in names:
        T = Z[f"pose__{n}"][i]; x, y, z, w = Rot.from_matrix(T[:3, :3]).as_quat()
        ents[n].set_pos(T[:3, 3], zero_velocity=True); ents[n].set_quat([w, x, y, z], zero_velocity=True)
    scene.step()
    if a.raster:
        from PIL import Image
        rgb = cam.render()[0]; Image.fromarray(np.asarray(rgb)).save(os.path.join(out, f"frame_{fr.count:05d}.png")); fr.count += 1
    else:
        fr.capture(cam)
    if i % 30 == 0:
        print(f"frame {i}/{len(Z['t'])}", flush=True)
print("wrote", fr.encode(out + ".mp4"))
