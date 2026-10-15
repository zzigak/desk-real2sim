"""Blender render of a bag run (sim/run_bag.py --export DIR), in the YAM-arcade studio.

Runs inside Blender 5.2. Everything after `--` is ours:

    blender -b -P render/render_bag.py -- --export out/heavy/exp --out out/frames/close_heavy \\
        --res 1920x1080 --samples 96 --track=0.70,-0.42,0.25 --lens 32 --track-lift 0.08 --orbit 20
    (wide shot: drop --track/--lens/--track-lift and use --orbit 14; --frames a:b[:step] or --still f1,f2 for parts)

The studio, tables, lighting, colour management and most material kinds come from the arcade compositor
(render/arcade/compose.py, copied from zzigak/yam-arcade on 2026-10-13). Added here: the robot pedestal, a kraft
paper material that sticks to the simulated bag surface (its texture coordinates are the bag's rest positions),
twisted-paper rope handles, and render meshes for the free objects. Each render mesh fits inside the collision shape
the simulator used (jar and can: their cylinder; orange: its sphere; paper ball: inside its sphere, crumpled
inwards; tea box: its box), so nothing drawn pokes through the bag more than the simulated shapes did.
Every pose and every bag/rope vertex is the simulator's; nothing is moved or smoothed in time.
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "arcade"))
import compose as C  # noqa: E402

PRIMS = ("jar", "full_can", "can", "orange", "tea_box", "ball")
TEX = os.path.join(HERE, "tex")  # render/fetch_textures.sh (Poly Haven, CC0); FreePBR maps are optional extras


def have_tex(*names):
    missing = [n for n in names if not os.path.exists(os.path.join(TEX, n))]
    if missing:
        C.log(f"texture(s) missing in {TEX}: {missing} -> that layer is skipped (render/fetch_textures.sh)")
    return not missing


def tex_node(nt, fname, vec, colour=True, box=False):
    img = C.load_image(os.path.join(TEX, fname))
    if not colour:
        img.colorspace_settings.name = "Non-Color"
    t = nt.n("ShaderNodeTexImage", image=img, **(dict(projection="BOX", projection_blend=0.2) if box else {}))
    nt.link(vec, t.inputs["Vector"])
    return t


def pbr(nt, p, vec, albedo, rough, normal, normal_strength=1.0, box=False, tint=None, rough_gain=1.0):
    """Wire albedo / roughness / OpenGL normal maps into Principled p, all looked up at `vec`."""
    a = tex_node(nt, albedo, vec, True, box)
    col = a.outputs["Color"]
    if tint is not None:
        m = nt.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        sock = lambda coll, ident: next(x for x in coll if x.identifier == ident)
        sock(m.inputs, "Factor_Float").default_value = 1.0
        nt.link(col, sock(m.inputs, "A_Color"))
        sock(m.inputs, "B_Color").default_value = (*tint, 1.0)
        col = sock(m.outputs, "Result_Color")
    nt.link(col, p.inputs["Base Color"])
    r = tex_node(nt, rough, vec, False, box)
    rg = nt.n("ShaderNodeMath", operation="MULTIPLY")
    nt.link(r.outputs["Color"], rg.inputs[0])
    rg.inputs[1].default_value = rough_gain
    nt.link(rg.outputs[0], p.inputs["Roughness"])
    nm = nt.n("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal_strength
    nt.link(tex_node(nt, normal, vec, False, box).outputs["Color"], nm.inputs["Color"])
    nt.link(nm.outputs["Normal"], p.inputs["Normal"])
    return nm


def object_coords(nt, tile_m):
    tc = nt.n("ShaderNodeTexCoord")
    mp = nt.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / tile_m,) * 3
    nt.link(tc.outputs["Object"], mp.inputs["Vector"])
    return mp.outputs["Vector"]


def upgrade_environment(args):
    """Concrete floor fading into the arcade cove's wall, and a studio HDRI for the sky the cove opens onto."""
    if not have_tex("concrete_wall_009_diff_4k.jpg", "concrete_wall_009_rough_4k.exr", "concrete_wall_009_nor_gl_4k.exr"):
        return _hdri(args)
    mat = bpy.data.materials["studio"]
    nt = C.NT.__new__(C.NT)
    nt.t, nt.x = mat.node_tree, 0
    p = next(n for n in nt.t.nodes if n.type == "BSDF_PRINCIPLED")
    mix = next(n for n in nt.t.nodes if n.type == "MIX")
    mr = next(n for n in nt.t.nodes if n.type == "MAP_RANGE")
    vec = object_coords(nt, 3.5)  # one 4k concrete tile = 3.5 m of floor
    a = tex_node(nt, "concrete_wall_009_diff_4k.jpg", vec)
    br = nt.n("ShaderNodeBrightContrast", **{"Bright": 0.0, "Contrast": 0.05})
    nt.link(a.outputs["Color"], br.inputs["Color"])
    nt.link(br.outputs["Color"], next(x for x in mix.inputs if x.identifier == "A_Color"))
    r = tex_node(nt, "concrete_wall_009_rough_4k.exr", vec, colour=False)
    rm = nt.n("ShaderNodeMapRange", **{"To Min": 0.35, "To Max": 0.8})
    nt.link(r.outputs["Color"], rm.inputs["Value"])
    nt.link(rm.outputs["Result"], p.inputs["Roughness"])
    nm = nt.n("ShaderNodeNormalMap")
    inv = nt.n("ShaderNodeMath", operation="MULTIPLY_ADD")  # strength 0.8 on the floor, 0 up the wall
    nt.link(mr.outputs["Result"], inv.inputs[0])
    inv.inputs[1].default_value = -0.8
    inv.inputs[2].default_value = 0.8
    nt.link(inv.outputs[0], nm.inputs["Strength"])
    nt.link(tex_node(nt, "concrete_wall_009_nor_gl_4k.exr", vec, colour=False).outputs["Color"], nm.inputs["Color"])
    nt.link(nm.outputs["Normal"], p.inputs["Normal"])
    _hdri(args)


def _hdri(args):
    if args.hdri and have_tex(args.hdri):
        w = bpy.context.scene.world
        wt = w.node_tree
        bg = next(n for n in wt.nodes if n.type == "BACKGROUND")
        env = wt.nodes.new("ShaderNodeTexEnvironment")
        env.image = C.load_image(os.path.join(TEX, args.hdri))
        wt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = args.hdri_strength


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--still", default=None)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--eye", default="1.62,-0.68,1.28")
    ap.add_argument("--target", default="0.42,0.0,1.05")
    ap.add_argument("--lens", type=float, default=28.0)
    ap.add_argument("--fstop", type=float, default=5.6)
    ap.add_argument("--orbit", type=float, default=0.0, help="camera azimuth sweep around the target over the whole "
                    "export, degrees (eased; 0 = still camera)")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--light", type=float, default=0.10)
    ap.add_argument("--bg", type=float, default=5.0)
    ap.add_argument("--world", type=float, default=0.35)
    ap.add_argument("--wood", default="charcoal", choices=("charcoal", "oak", "bamboo"), help="table top")
    ap.add_argument("--track", default=None, help="x,y,z: camera follows the bag's centre (Gaussian-smoothed over "
                    "--track-sigma s), eye = smoothed centre + this offset; --eye/--target are then ignored")
    ap.add_argument("--track-sigma", type=float, default=0.7)
    ap.add_argument("--track-lift", type=float, default=0.03, help="aim this far above the bag's centre, m")
    ap.add_argument("--hdri", default="poly_haven_studio_4k.exr", help="world texture in render/tex ('' = grey sky)")
    ap.add_argument("--hdri-strength", type=float, default=0.25)
    ap.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--save-blend", default=None)
    return ap.parse_args(argv)


def vec3(s):
    return tuple(float(v) for v in s.split(","))


# ---------------------------------------------------------------------------------------------------- materials
def kraft_material(name, rgb, attr="rest", fibre=1.0, uv=None):
    """Kraft paper: colour mottling and long fibres from 3D noise on the paper's rest coordinates (so the pattern
    travels with the sheet), a faint fibre bump, matte with a little sheen and translucency."""
    mat = bpy.data.materials.new(name)
    nt = C.NT(mat)
    out = nt.n("ShaderNodeOutputMaterial")
    p = nt.n("ShaderNodeBsdfPrincipled")
    nt.link(p.outputs["BSDF"], out.inputs["Surface"])
    co = nt.n("ShaderNodeAttribute", attribute_name=attr, attribute_type="GEOMETRY")
    # long fibres: noise stretched along x
    mp = nt.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (60.0 * fibre, 600.0 * fibre, 600.0 * fibre)
    nt.link(co.outputs["Vector"], mp.inputs["Vector"])
    fib = nt.n("ShaderNodeTexNoise", **{"Scale": 1.0, "Detail": 6.0, "Roughness": 0.7})
    nt.link(mp.outputs["Vector"], fib.inputs["Vector"])
    mot = nt.n("ShaderNodeTexNoise", **{"Scale": 18.0, "Detail": 3.0, "Roughness": 0.5})
    nt.link(co.outputs["Vector"], mot.inputs["Vector"])
    mix = nt.n("ShaderNodeMath", operation="MULTIPLY_ADD")  # 0.75 mottle + 0.25 fibre
    nt.link(mot.outputs["Fac"], mix.inputs[0])
    mix.inputs[1].default_value = 0.75
    m2 = nt.n("ShaderNodeMath", operation="MULTIPLY")
    nt.link(fib.outputs["Fac"], m2.inputs[0])
    m2.inputs[1].default_value = 0.25
    nt.link(m2.outputs[0], mix.inputs[2])
    ramp = nt.n("ShaderNodeValToRGB")
    lin = np.array(C.srgb2lin(rgb))
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[0].color = (*(lin * 0.80), 1)
    ramp.color_ramp.elements[1].color = (*np.clip(lin * 1.12, 0, 1), 1)
    nt.link(mix.outputs[0], ramp.inputs["Fac"])
    nt.link(ramp.outputs["Color"], p.inputs["Base Color"])
    bump = nt.n("ShaderNodeBump", **{"Strength": 0.08, "Distance": 0.0002})
    nt.link(fib.outputs["Fac"], bump.inputs["Height"])
    nt.link(bump.outputs["Normal"], p.inputs["Normal"])
    if uv and have_tex("white-plain-paper_normal-ogl.png"):  # paper surface relief (white-plain-paper normal map, 2k = 25 cm of sheet) on the rest-shape UVs
        uvn = nt.n("ShaderNodeUVMap", uv_map=uv)
        mpu = nt.n("ShaderNodeMapping")
        mpu.inputs["Scale"].default_value = (4.0, 4.0, 1.0)
        nt.link(uvn.outputs["UV"], mpu.inputs["Vector"])
        nm = nt.n("ShaderNodeNormalMap", uv_map=uv)
        nm.inputs["Strength"].default_value = 0.9
        nt.link(tex_node(nt, "white-plain-paper_normal-ogl.png", mpu.outputs["Vector"], colour=False).outputs["Color"],
                nm.inputs["Color"])
        nt.link(nm.outputs["Normal"], bump.inputs["Normal"])
    nt.set(p, "Roughness", 0.78)
    nt.set(p, "Specular IOR Level", 0.22)
    nt.set(p, "Sheen Weight", 0.25)
    nt.set(p, "Sheen Roughness", 0.5)
    nt.set(p, "Subsurface Weight", 0.06)
    nt.set(p, "Subsurface Scale", 0.0008)
    return mat


def rope_material(name, rgb, lay=0.012, strands=3):
    """Twisted paper cord: `strands` helical grooves with lay length `lay` (m), from the tube's "cord" attribute
    (arc length along the cord in m, fraction around it): phase 2 pi N (v + s / lay)."""
    mat = bpy.data.materials.new(name)
    nt = C.NT(mat)
    out = nt.n("ShaderNodeOutputMaterial")
    p = nt.n("ShaderNodeBsdfPrincipled")
    nt.link(p.outputs["BSDF"], out.inputs["Surface"])
    tc = nt.n("ShaderNodeAttribute", attribute_name="cord", attribute_type="GEOMETRY")
    sep = nt.n("ShaderNodeSeparateXYZ")
    nt.link(tc.outputs["Vector"], sep.inputs[0])
    a = nt.n("ShaderNodeMath", operation="MULTIPLY_ADD")  # s / lay + v
    nt.link(sep.outputs["X"], a.inputs[0])
    a.inputs[1].default_value = 1.0 / lay
    nt.link(sep.outputs["Y"], a.inputs[2])
    s = nt.n("ShaderNodeMath", operation="MULTIPLY")
    nt.link(a.outputs[0], s.inputs[0])
    s.inputs[1].default_value = 2 * math.pi * strands
    sn = nt.n("ShaderNodeMath", operation="SINE")
    nt.link(s.outputs[0], sn.inputs[0])
    bump = nt.n("ShaderNodeBump", **{"Strength": 0.6, "Distance": 0.0006})
    nt.link(sn.outputs[0], bump.inputs["Height"])
    nt.link(bump.outputs["Normal"], p.inputs["Normal"])
    ramp = nt.n("ShaderNodeValToRGB")
    lin = np.array(C.srgb2lin(rgb))
    ramp.color_ramp.elements[0].color = (*(lin * 0.7), 1)
    ramp.color_ramp.elements[1].color = (*np.clip(lin * 1.05, 0, 1), 1)
    nt.link(sn.outputs[0], ramp.inputs["Fac"])
    nt.link(ramp.outputs["Color"], p.inputs["Base Color"])
    nt.set(p, "Roughness", 0.8)
    nt.set(p, "Sheen Weight", 0.3)
    return mat


def principled(name, rgb, rough=0.5, metallic=0.0, coat=0.0, coat_rough=0.1, sss=0.0, sss_scale=0.002, spec=0.5,
               transmission=0.0, ior=1.5):
    mat, nt, p = C.simple_material(name, C.srgb2lin(rgb), rough=rough, metallic=metallic, spec=spec)
    nt.set(p, "Coat Weight", coat)
    nt.set(p, "Coat Roughness", coat_rough)
    nt.set(p, "Subsurface Weight", sss)
    nt.set(p, "Subsurface Scale", sss_scale)
    nt.set(p, "Transmission Weight", transmission)
    nt.set(p, "IOR", ior)
    return mat, nt, p


def orange_material():
    mat, nt, p = principled("orange_peel", (1.0, 0.42, 0.0), rough=0.38, coat=0.25, coat_rough=0.3, sss=0.05,
                            sss_scale=0.002)
    nt.set(p, "Subsurface Radius", (1.0, 0.45, 0.1))
    tc = nt.n("ShaderNodeTexCoord")
    vor = nt.n("ShaderNodeTexVoronoi", feature="SMOOTH_F1", **{"Scale": 260.0})
    nt.link(tc.outputs["Object"], vor.inputs["Vector"])
    bump = nt.n("ShaderNodeBump", **{"Strength": 0.35, "Distance": 0.0004})
    nt.link(vor.outputs["Distance"], bump.inputs["Height"])
    nt.link(bump.outputs["Normal"], p.inputs["Normal"])
    return mat


def label_box_material():
    """Tea carton: green print, a cream label band round the middle, a thin gold rule above and below it."""
    mat, nt, p = principled("tea_box", (0.16, 0.40, 0.27), rough=0.38, coat=0.25, coat_rough=0.2, spec=0.4)
    tc = nt.n("ShaderNodeTexCoord")
    sep = nt.n("ShaderNodeSeparateXYZ")
    nt.link(tc.outputs["Object"], sep.inputs[0])
    ab = nt.n("ShaderNodeMath", operation="ABSOLUTE")
    nt.link(sep.outputs["Z"], ab.inputs[0])
    band = nt.n("ShaderNodeMath", operation="LESS_THAN")
    nt.link(ab.outputs[0], band.inputs[0])
    band.inputs[1].default_value = 0.022
    rule = nt.n("ShaderNodeMath", operation="COMPARE")
    nt.link(ab.outputs[0], rule.inputs[0])
    rule.inputs[1].default_value = 0.0245
    rule.inputs[2].default_value = 0.0012
    m1 = nt.n("ShaderNodeMix", data_type="RGBA")
    sock = lambda coll, ident: next(x for x in coll if x.identifier == ident)
    sock(m1.inputs, "A_Color").default_value = C.rgba((0.16, 0.40, 0.27))
    sock(m1.inputs, "B_Color").default_value = C.rgba((0.93, 0.89, 0.78))
    nt.link(band.outputs[0], sock(m1.inputs, "Factor_Float"))
    m2 = nt.n("ShaderNodeMix", data_type="RGBA")
    nt.link(sock(m1.outputs, "Result_Color"), sock(m2.inputs, "A_Color"))
    sock(m2.inputs, "B_Color").default_value = C.rgba((0.78, 0.62, 0.30))
    nt.link(rule.outputs[0], sock(m2.inputs, "Factor_Float"))
    nt.link(sock(m2.outputs, "Result_Color"), p.inputs["Base Color"])
    return mat


# ---------------------------------------------------------------------------------------------------- meshes
def mesh_obj(name, V, F, mat, parent, smooth=True, sharp_deg=35):
    ob = C.new_mesh_obj(name, np.asarray(V, float), np.asarray(F), mat, smooth=smooth, parent=parent)
    ob.matrix_parent_inverse = Matrix.Identity(4)
    if smooth:
        ob.data.set_sharp_from_angle(angle=math.radians(sharp_deg))
    return ob


def jar(pose):
    """Jam jar inside the simulated cylinder (r 40 mm, h 100 mm, centred): refracting glass walls 2.5 mm thick,
    jam to 2.5 cm below the shoulder, a painted metal lid on the neck."""
    R, H = 0.040, 0.100
    zb, zt = -H / 2, H / 2
    wall = 0.0025
    outer = [(0.0, zb), (R - 0.004, zb), (R - 0.001, zb + 0.0015), (R, zb + 0.005), (R, zt - 0.030),
             (R - 0.004, zt - 0.021), (0.034, zt - 0.016), (0.034, zt - 0.012)]
    inner = [(0.034 - wall, zt - 0.012), (0.034 - wall, zt - 0.016), (R - wall - 0.004, zt - 0.021 - 0.001),
             (R - wall, zt - 0.030), (R - wall, zb + 0.007), (R - wall - 0.004, zb + wall), (0.0, zb + wall)]
    V, F = C.lathe(outer + inner, n=96)
    glass = C.make_material(dict(kind="glass", color=(0.93, 0.97, 0.95)), tag="_jar")
    mesh_obj("jar_glass", V, F, glass, pose)
    jam_top = zt - 0.030
    g = 0.0004
    jp = [(0.0, zb + wall + g), (R - wall - 0.004, zb + wall + g), (R - wall - g, zb + 0.007), (R - wall - g, jam_top),
          (0.0, jam_top)]
    V, F = C.lathe(jp, n=96)
    jm, nt, p = principled("jam", (0.45, 0.04, 0.07), rough=0.15, sss=0.6, sss_scale=0.01, transmission=0.15, ior=1.45)
    mesh_obj("jar_jam", V, F, jm, pose)
    lid = [(0.0, zt), (0.036, zt), (0.0375, zt - 0.0015), (0.0375, zt - 0.011), (0.0355, zt - 0.0125),
           (0.0345, zt - 0.0125), (0.0345, zt - 0.002), (0.0, zt - 0.002)]
    V, F = C.lathe(lid, n=96, flip=True)
    lm, nt, p = principled("lid", (0.80, 0.62, 0.22), rough=0.25, metallic=1.0, coat=0.4)
    mesh_obj("jar_lid", V, F, lm, pose)


def can(pose, paint_rgb, name):
    """Drink can inside the simulated cylinder (r 33 mm, h 122 mm): painted body, bare aluminium necks and ends."""
    R, H = 0.033, 0.122
    zb, zt = -H / 2, H / 2
    body = [(R - 0.0002, zb + 0.012), (R, zb + 0.014), (R, zt - 0.014), (R - 0.0002, zt - 0.012)]
    V, F = C.lathe(body, n=96)
    pm, nt, p = principled(f"{name}_paint", paint_rgb, rough=0.22, metallic=0.6, coat=0.8, coat_rough=0.06)
    mesh_obj(f"{name}_body", V, F, pm, pose)
    bottom = [(0.0, zb + 0.004), (0.018, zb + 0.0015), (0.024, zb), (0.026, zb), (0.030, zb + 0.004),
              (R - 0.0002, zb + 0.012)]
    top = [(R - 0.0002, zt - 0.012), (0.0285, zt - 0.003), (0.0275, zt), (0.0265, zt), (0.0262, zt - 0.003),
           (0.0, zt - 0.003)]
    al, nt, p = principled(f"{name}_alu", (0.86, 0.86, 0.87), rough=0.22, metallic=1.0)
    for nm, prof in (("bottom", bottom), ("top", top)):
        V, F = C.lathe(prof, n=96)
        mesh_obj(f"{name}_{nm}", V, F, al, pose)
    # ring pull, flat on the lid
    tab = [(0.0, zt - 0.0026), (0.007, zt - 0.0026), (0.007, zt - 0.0018), (0.0, zt - 0.0018)]
    V, F = C.lathe(tab, n=32)
    t = mesh_obj(f"{name}_tab", V, F, al, pose)
    t.location = (0.008, 0.0, 0.0)
    t.scale = (1.0, 0.6, 1.0)


def orange(pose, R=0.040):
    import bmesh
    me = bpy.data.meshes.new("orange")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=R)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    me.materials.append(orange_material())
    ob = bpy.data.objects.new("orange", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = pose
    ob.matrix_parent_inverse = Matrix.Identity(4)
    # stem scar: a small dark-green disc set into the peel at the top
    V, F = C.lathe([(0.0, R - 0.0004), (0.0035, R - 0.0006), (0.0035, R - 0.0016), (0.0, R - 0.0016)], n=24, flip=True)
    sm, nt, p = principled("stem", (0.30, 0.33, 0.12), rough=0.6)
    mesh_obj("orange_stem", V, F, sm, pose)


def paper_ball(pose, R=0.030, seed=3):
    """Crumpled paper ball: an icosphere pushed inwards (never out of the simulated 30 mm sphere) by a few random
    creases, flat shaded so the facets read as crumpled sheet."""
    import bmesh
    rng = np.random.default_rng(seed)
    me = bpy.data.meshes.new("paper_ball")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    bm.to_mesh(me)
    bm.free()
    V = np.array([v.co for v in me.vertices])
    n = V / np.linalg.norm(V, axis=1, keepdims=True)
    k = rng.normal(size=(9, 3))
    k = 5.0 * k / np.linalg.norm(k, axis=1, keepdims=True)
    ph = rng.uniform(0, 2 * np.pi, 9)
    d = np.abs(np.sin(n @ k.T + ph)).mean(1)
    r = R * (1.0 - 0.16 * (d - d.min()) / (d.max() - d.min()))
    me.vertices.foreach_set("co", (n * r[:, None]).astype(np.float32).ravel())
    me.update()
    rest = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    rest.data.foreach_set("vector", (n * 0.05).astype(np.float32).ravel())
    me.materials.append(kraft_material("paper_ball", (0.95, 0.94, 0.90), fibre=0.5))
    ob = bpy.data.objects.new("paper_ball", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = pose
    ob.matrix_parent_inverse = Matrix.Identity(4)


def tea_box(pose, size=(0.08, 0.05, 0.11)):
    V, F = C.box_mesh(*size)
    ob = mesh_obj("tea_box", V, F, label_box_material(), pose, smooth=False)
    C.bevel(ob, 0.0012, 2)


# ---------------------------------------------------------------------------------------------------- scene
WOODS = {  # name: (albedo, roughness, normal, metres per texture tile)
    "oak": ("bare-wood1_albedo.png", "bare-wood1_roughness.png", "bare-wood1_normal-ogl.png", 0.9),
    "bamboo": ("bamboo-wood-semigloss-albedo.png", "bamboo-wood-semigloss-roughness.png",
               "bamboo-wood-semigloss-normal.png", 0.7),
}


def charcoal_top_material():
    """Lab-bench epoxy resin: near-black charcoal with a faint mineral speckle and soft satin sheen, so the kraft bag
    (and the white robot) stand out against it."""
    mat, nt, p = principled("charcoal_top", (0.085, 0.09, 0.095), rough=0.38, coat=0.25, coat_rough=0.3, spec=0.5)
    tc = nt.n("ShaderNodeTexCoord")
    sp = nt.n("ShaderNodeTexNoise", **{"Scale": 900.0, "Detail": 2.0, "Roughness": 0.5})
    nt.link(tc.outputs["Object"], sp.inputs["Vector"])
    cl = nt.n("ShaderNodeTexNoise", **{"Scale": 6.0, "Detail": 3.0, "Roughness": 0.5})
    nt.link(tc.outputs["Object"], cl.inputs["Vector"])
    ramp = nt.n("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.62
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[0].color = (*C.srgb2lin((0.085, 0.09, 0.095)), 1)
    ramp.color_ramp.elements[1].color = (*C.srgb2lin((0.20, 0.205, 0.21)), 1)
    nt.link(sp.outputs["Fac"], ramp.inputs["Fac"])
    nt.link(ramp.outputs["Color"], p.inputs["Base Color"])
    rr = nt.n("ShaderNodeMapRange", **{"To Min": 0.30, "To Max": 0.48})
    nt.link(cl.outputs["Fac"], rr.inputs["Value"])
    nt.link(rr.outputs["Result"], p.inputs["Roughness"])
    return mat


def wood_top_material(kind):
    """Table top veneer from PBR maps (WOODS), box-projected, thin satin coat."""
    alb, rough, nor, tile = WOODS[kind]
    mat = bpy.data.materials.new(f"{kind}_top")
    nt = C.NT(mat)
    out = nt.n("ShaderNodeOutputMaterial")
    p = nt.n("ShaderNodeBsdfPrincipled")
    nt.link(p.outputs["BSDF"], out.inputs["Surface"])
    vec = object_coords(nt, tile)
    pbr(nt, p, vec, alb, rough, nor, normal_strength=0.6, box=True, rough_gain=0.8)
    nt.set(p, "Coat Weight", 0.3)
    nt.set(p, "Coat Roughness", 0.22)
    return mat


def build_tables(meta, wood):
    top = float(meta["top"])
    wm = charcoal_top_material() if wood == "charcoal" or not have_tex(*WOODS[wood][:3]) else wood_top_material(wood)
    for i, t in enumerate(meta["tables"]):
        sx, sy = t["size"]
        C.L.TABLE = dict(size_x=sx, size_y=sy, height=top, top_thickness=0.03, leg_inset=0.05, leg_size=0.05)
        C.build_table(f"t{i}", C.world_matrix(t["c"][0], t["c"][1], 0.0), dict(top="wood", wood=(0.8, 0.66, 0.48),
                                                                               legs="alu"))
        bpy.data.objects[f"top_t{i}"].data.materials[0] = wm
    p = meta["pedestal"]
    V, F = C.box_mesh(*p["size"], c=p["c"])
    m, nt, pr = principled("pedestal", (0.14, 0.145, 0.155), rough=0.45, coat=0.2)
    ob = C.new_mesh_obj("pedestal", V, F, m)
    C.bevel(ob, 0.004, 3)


class Rope:
    """A simulated rope centreline as a tube: Catmull-Rom through the simulated nodes (3 sub-points per segment,
    spatial only), parallel-transport frames, 12 sides plus a seam column so the "cord" coordinate never wraps."""

    def __init__(self, name, P, radius, mat, sub=3, sides=12):
        self.P, self.r, self.sub, self.sides = P, radius, sub, sides
        n = (P.shape[1] - 1) * sub + 1
        self.n = n
        ring = sides + 1
        F = []
        for i in range(n - 1):
            for j in range(sides):
                a, b = i * ring + j, i * ring + j + 1
                F += [(a, b, b + ring), (a, b + ring, a + ring)]
        V = np.zeros((n * ring, 3))
        self.me = bpy.data.meshes.new(name)
        C.set_mesh(self.me, V, np.array(F), smooth=True)
        self.attr = self.me.attributes.new("cord", "FLOAT_VECTOR", "POINT")
        self.me.materials.append(mat)
        ob = bpy.data.objects.new(name, self.me)
        bpy.context.scene.collection.objects.link(ob)

    def centreline(self, p):
        m = np.empty_like(p)
        m[1:-1] = 0.5 * (p[2:] - p[:-2])
        m[0], m[-1] = p[1] - p[0], p[-1] - p[-2]
        t = np.linspace(0, 1, self.sub + 1)[:-1]
        h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
        out = [h00[:, None] * p[i] + h10[:, None] * m[i] + h01[:, None] * p[i + 1] + h11[:, None] * m[i + 1]
               for i in range(len(p) - 1)]
        return np.concatenate(out + [p[-1:]])

    def apply(self, f):
        c = self.centreline(self.P[min(f, len(self.P) - 1)].astype(np.float64))
        T = np.gradient(c, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N = np.zeros_like(c)
        a = np.cross(T[0], (0, 0, 1.0))
        N[0] = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array((1.0, 0, 0))
        for i in range(1, len(c)):  # parallel transport
            v = N[i - 1] - T[i] * (N[i - 1] @ T[i])
            N[i] = v / np.linalg.norm(v)
        Bn = np.cross(T, N)
        ang = np.linspace(0, 2 * np.pi, self.sides + 1)
        V = c[:, None, :] + self.r * (np.cos(ang)[None, :, None] * N[:, None, :] + np.sin(ang)[None, :, None] * Bn[:, None, :])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        cord = np.stack(np.broadcast_arrays(s[:, None], (ang / (2 * np.pi))[None, :], 0.0 * s[:, None]), -1)
        self.me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        self.attr.data.foreach_set("vector", cord.astype(np.float32).ravel())
        self.me.update()


class Bag:
    """The simulated bag surface, every frame; kraft material on the rest coordinates; creases kept sharp."""

    def __init__(self, d, meta):
        self.dir = os.path.join(d, meta["objects"]["bag"]["dir"])
        files = sorted(f for f in os.listdir(self.dir) if f.endswith(".npz"))
        self.idx = np.array([int(f[:-4]) for f in files])
        z = np.load(os.path.join(self.dir, files[0]))
        rest = np.load(os.path.join(d, "bag_rest.npy"))
        self.me = bpy.data.meshes.new("bag")
        C.set_mesh(self.me, z["V"], z["F"], smooth=True)
        a = self.me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", rest.astype(np.float32).ravel())
        # UVs in metres on the flat rest shape: each triangle projected onto the box face it lies on
        F = np.asarray(z["F"])
        r0 = rest - rest.min(0)
        n = np.abs(np.cross(r0[F[:, 1]] - r0[F[:, 0]], r0[F[:, 2]] - r0[F[:, 0]]))
        ax = n.argmax(1)  # 0: x-wall (y, z), 1: y-wall (x, z), 2: floor (x, y)
        drop = np.array([[1, 2], [0, 2], [0, 1]])[ax]
        P = r0[F]  # (nf, 3, 3)
        uvs = np.stack([np.take_along_axis(P, drop[:, None, 0:1].repeat(3, 1), 2)[..., 0],
                        np.take_along_axis(P, drop[:, None, 1:2].repeat(3, 1), 2)[..., 0]], -1)
        uvl = self.me.uv_layers.new(name="UVMap")
        lv = np.zeros(len(self.me.loops), np.int32)
        self.me.loops.foreach_get("vertex_index", lv)
        assert np.array_equal(lv, F.ravel()), "loop order differs from the face list"
        uvl.data.foreach_set("uv", uvs.astype(np.float32).ravel())
        self.me.materials.append(kraft_material("kraft", (0.66, 0.49, 0.31), uv="UVMap"))
        ob = bpy.data.objects.new("bag", self.me)
        bpy.context.scene.collection.objects.link(ob)
        self.cur = None

    def apply(self, f):
        j = int(self.idx[max(0, np.searchsorted(self.idx, f, side="right") - 1)])
        if j == self.cur:
            return
        V = np.load(os.path.join(self.dir, f"{j:05d}.npz"))["V"]
        self.me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        self.me.update()
        self.me.set_sharp_from_angle(angle=math.radians(40))
        self.cur = j


def main():
    args = parse()
    t0 = time.time()
    d = os.path.abspath(args.export)
    meta = json.load(open(os.path.join(d, "meta.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    C.setup_render(args)
    C.build_studio(args)
    upgrade_environment(args)
    build_tables(meta, args.wood)
    coll = bpy.data.collections.new("sim")
    sc.collection.children.link(coll)
    st = C.Station("sim", d, Matrix.Identity(4), coll, hide=("bag", "rope_*"))
    # free objects drawn as the things they stand for, inside their simulated shapes
    for oname in meta["objects"]:
        if oname not in PRIMS:
            continue
        pose = bpy.data.objects[f"sim:{oname}"]
        for ch in list(pose.children):
            bpy.data.objects.remove(ch)
        if oname == "jar":
            jar(pose)
        elif oname in ("full_can", "can"):
            can(pose, (0.72, 0.06, 0.10) if oname == "full_can" else (0.10, 0.32, 0.62), oname)
        elif oname == "orange":
            orange(pose)
        elif oname == "ball":
            paper_ball(pose)
        elif oname == "tea_box":
            tea_box(pose)
    cord = rope_material("cord", (0.58, 0.42, 0.26))
    ropes = [Rope(k, st.traj[f"{k}/pts"], float(o["radius"]), cord) for k, o in meta["objects"].items() if o["kind"] == "rod"]
    bag = Bag(d, meta)
    movers = [st, bag] + ropes

    cd = bpy.data.cameras.new("cam")
    cd.sensor_width = 36.0
    cd.clip_start, cd.clip_end = 0.03, 100.0
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    eye, tgt = vec3(args.eye), vec3(args.target)
    cd.lens = args.lens

    track = None
    if args.track:
        # bag centre per export frame (mean of the simulated surface), smoothed in time for the camera only
        com = np.array([np.load(os.path.join(bag.dir, f"{j:05d}.npz"))["V"].mean(0) for j in bag.idx])
        com = np.array([com[max(0, np.searchsorted(bag.idx, f, side="right") - 1)] for f in range(st.n)])
        sig = args.track_sigma * st.fps
        k = np.arange(-int(3 * sig), int(3 * sig) + 1)
        w = np.exp(-0.5 * (k / sig) ** 2)
        w /= w.sum()
        pad = np.pad(com, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
        track = np.stack([np.convolve(pad[:, i], w, mode="valid") for i in range(3)], 1)
        track[:, 2] += args.track_lift
        offset = Vector(vec3(args.track))

    def cam_upd(f):
        u = f / max(st.n - 1, 1)
        a = math.radians(args.orbit) * (0.5 - 0.5 * math.cos(math.pi * u) - 0.5)
        t_ = Vector(track[min(f, len(track) - 1)].tolist()) if track is not None else Vector(tgt)
        rel = offset if track is not None else Vector(eye) - Vector(tgt)
        rel = Matrix.Rotation(a, 3, "Z") @ rel
        C.look_at(cam, t_ + rel, t_)
    if args.fstop > 0:
        cd.dof.use_dof = True
        cd.dof.focus_distance = (offset if track is not None else Vector(eye) - Vector(tgt)).length
        cd.dof.aperture_fstop = args.fstop

    n = st.n
    if args.still:
        frames = [int(v) for v in args.still.split(",")]
    elif args.frames:
        parts = [int(v) if v else None for v in args.frames.split(":")]
        frames = list(range(parts[0] or 0, parts[1] if len(parts) > 1 and parts[1] is not None else n,
                            parts[2] if len(parts) > 2 and parts[2] else 1))
    else:
        frames = list(range(n))
    C.log(f"scene built in {time.time() - t0:.1f}s; {n} export frames at {st.fps:.2f} fps; rendering {len(frames)}")
    if args.save_blend:
        for m in movers:
            m.apply(frames[0])
        cam_upd(frames[0])
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    os.makedirs(args.out, exist_ok=True)
    stats_p = os.path.join(args.out, f"render_stats_{os.getpid()}.json")
    stats = dict(export=d, res=args.res, samples=args.samples, eye=args.eye, target=args.target, lens=args.lens,
                 export_fps=st.fps, host=os.uname().nodename, frames={})
    for f in frames:
        final = os.path.join(args.out, f"{f:05d}.png")
        if args.skip_existing and os.path.exists(final) and os.path.getsize(final) > 0:
            continue
        t1 = time.time()
        for m in movers:
            m.apply(f)
        cam_upd(f)
        sc.frame_set(f)
        tmp = os.path.join(args.out, f"_tmp_{os.getpid()}_{f:05d}.png")
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        os.replace(tmp, final)
        stats["frames"][str(f)] = round(time.time() - t1, 2)
        C.log(f"frame {f} {time.time() - t1:.1f}s")
        json.dump(stats, open(stats_p, "w"), indent=1)
    C.log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
