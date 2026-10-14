"""Render a desk robot export (robot_desk.Desk.start_export, yam-arcade export.Recorder format) with the yam-arcade
compositor's studio, materials, mesh import, station loader and Cycles setup (render/compose.py), but without its
arcade table: the desk export carries its own tabletop, and this script adds four legs under it.

    blender -b -P sim/render_desk.py -- --camera desk --source desk=<export dir> --res 1280x720 --samples 64 \\
        --out <frames dir> --encode <video.mp4> [--frames a:b]

ARCADE (env) points at the yam-arcade checkout or render mirror that holds render/compose.py and layout.py."""
import os
import sys
import time

ARC = os.environ.get("ARCADE", os.path.expanduser("~/projects/yam-arcade"))
sys.path.insert(0, os.path.join(ARC, "render"))
sys.path.insert(0, ARC)

import bpy  # noqa: E402
import compose as C  # noqa: E402

DESK = dict(size_x=1.25355, size_y=0.75624, top=0.75, thickness=0.025, leg=0.05, inset=0.07)   # after the export's -2.5 cm


def legs(Mw):
    lm = C.leg_material("black")
    sx, sy, h, th, leg, inset = (DESK[k] for k in ("size_x", "size_y", "top", "thickness", "leg", "inset"))
    xs, ys = sx / 2 - inset - leg / 2, sy / 2 - inset - leg / 2
    root = bpy.data.objects.new("desk_legs", None)
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = Mw
    for i, (a, b) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        V, F = C.box_mesh(leg, leg, h - th, (a * xs, b * ys, (h - th) / 2))
        C.bevel(C.new_mesh_obj(f"desk_leg{i}", V, F, lm, parent=root), 0.006, 3)
    for i, (cx, cy, lx, ly) in enumerate(((0, ys, 2 * xs, 0.02), (0, -ys, 2 * xs, 0.02), (xs, 0, 0.02, 2 * ys), (-xs, 0, 0.02, 2 * ys))):
        V, F = C.box_mesh(lx, ly, 0.06, (cx, cy, h - th - 0.03))
        C.bevel(C.new_mesh_obj(f"desk_apron{i}", V, F, lm, parent=root), 0.003, 2)


def main():
    args = C.parse()
    args.out = os.path.abspath(args.out)
    t_start = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)   # as compose.main: no startup cube / light / camera
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    src = dict(kv.partition("=")[::2] for kv in args.source)
    d = os.path.abspath(src["desk"])
    C.setup_render(args)
    C.build_studio(args)
    coll = bpy.data.collections.new("stations")
    sc.collection.children.link(coll)
    Mw = C.world_matrix(0.0, 0.0, 0.0)
    S = C.Station("desk", d, Mw, coll, hide=tuple(args.hide))
    legs(Mw)
    if args.frames:
        parts = [int(v) if v else None for v in args.frames.split(":")]
        a, b = parts[0] or 0, parts[1] if len(parts) > 1 and parts[1] is not None else S.n
        frames = list(range(a, b, parts[2] if len(parts) > 2 and parts[2] else 1))
    elif args.still:
        frames = [int(v) for v in args.still.split(",")]
    else:
        frames = list(range(S.n))
    cam, cam_upd = C.make_camera(args, {"desk": S}, {"desk": Mw}, frames)
    C.log(f"desk scene built in {time.time() - t_start:.1f}s; {len(frames)} frames")
    os.makedirs(args.out, exist_ok=True)
    for f in frames:
        final = os.path.join(args.out, f"{f:05d}.png")
        if args.skip_existing and os.path.exists(final) and os.path.getsize(final) > 0:
            continue
        t0 = time.time()
        S.apply(f)
        cam_upd(f)
        sc.frame_set(f)
        tmp = os.path.join(args.out, f"_tmp_{os.getpid()}_{f:05d}.png")
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        os.replace(tmp, final)
        C.log(f"frame {f} {time.time() - t0:.1f}s")
    if args.encode and not args.still:
        C.encode(args.out, frames, args.fps, args.encode)
    C.log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
