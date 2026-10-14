"""Build every desk object as a URDF (visual: photo-textured box / SAM 3D mesh; collision: procedural, IPC-clean)
plus deformable meshes, and write assets/scene.json with world poses from layout.json + sam3d_fit.json.

World: z up, floor z=0, table top z=H, x right, y toward the wall, origin under the table-top centre."""
import json, os
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
from objwriter import Obj, inertial_xml, mesh_inertial, inertia_box
from cloth_mesh import folded_cloth, sheet

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
A = os.path.join(ROOT, "assets"); U = os.path.join(A, "urdf"); os.makedirs(U, exist_ok=True)
import shutil, glob
for _t in glob.glob(os.path.join(A, "tex", "*.png")): shutil.copy(_t, U)   # trimesh will not follow ../ in map_Kd
L = json.load(open(f"{ROOT}/layout.json")); FIT = json.load(open(f"{A}/sam3d_fit.json")); SP = json.load(open(f"{A}/specs.json"))
H = L["table"]["top_z"]; DH = 0.001                                   # IPC contact distance
SC = {"H": H, "objects": {}, "deformables": {}}
GAP = 0.0015                                                          # rigid-on-rigid start clearance (> 0, settles)

def write_urdf(name, links, joints=()):
    x = [f'<robot name="{name}">']
    for ln in links:
        x.append(f'<link name="{ln["name"]}">{ln["inertial"]}')
        for v in ln.get("visual", []):
            o = v.get("origin", (0, 0, 0)); r = v.get("rpy", (0, 0, 0))
            x.append(f'<visual><origin xyz="{o[0]:.5f} {o[1]:.5f} {o[2]:.5f}" rpy="{r[0]:.5f} {r[1]:.5f} {r[2]:.5f}"/>'
                     f'<geometry><mesh filename="{v["mesh"]}"/></geometry></visual>')
        for c in ln.get("collision", []):
            o = c.get("origin", (0, 0, 0))
            g = f'<box size="{c["box"][0]:.5f} {c["box"][1]:.5f} {c["box"][2]:.5f}"/>' if "box" in c else f'<mesh filename="{c["mesh"]}"/>'
            x.append(f'<collision><origin xyz="{o[0]:.5f} {o[1]:.5f} {o[2]:.5f}" rpy="0 0 0"/><geometry>{g}</geometry></collision>')
        x.append("</link>")
    for j in joints:
        o = j["origin"]; q0 = j.get("q0", 0.0); ax = np.array(j["axis"], float)
        rpy = Rot.from_rotvec(ax * q0).as_euler("xyz")                   # bake the photographed angle: q=0 is the scene as shot
        j = dict(j, lo=j["lo"] - q0, hi=j["hi"] - q0)
        x.append(f'<joint name="{j["name"]}" type="{j["type"]}"><parent link="{j["parent"]}"/><child link="{j["child"]}"/>'
                 f'<origin xyz="{o[0]:.5f} {o[1]:.5f} {o[2]:.5f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/><axis xyz="{j["axis"][0]} {j["axis"][1]} {j["axis"][2]}"/>'
                 f'<limit lower="{j["lo"]:.4f}" upper="{j["hi"]:.4f}" effort="{j.get("effort", 50)}" velocity="{j.get("vel", 5)}"/>'
                 f'<dynamics damping="{j.get("damping", 0.0)}" friction="{j.get("friction", 0.0)}"/></joint>')
    x.append("</robot>")
    open(f"{U}/{name}.urdf", "w").write("\n".join(x))

def box_volume(boxes): return sum(np.prod(b["box"]) for b in boxes)
def obj(name, pos, euler_deg, mass, col_volume, friction, kind="rigid", **kw):
    SC["objects"][name] = dict(urdf=f"urdf/{name}.urdf", pos=[float(v) for v in pos], euler=[float(v) for v in euler_deg],
                               mass=mass, rho=float(mass / col_volume), friction=friction, kind=kind, **kw)

MAT = dict(alu=dict(kd=(0.42, 0.43, 0.45)), silver=dict(kd=(0.75, 0.76, 0.78)), black=dict(kd=(0.03, 0.03, 0.035)),
           bamboo=dict(kd=(1, 1, 1), map="bamboo.png"), bamboo_side=dict(kd=(0.78, 0.58, 0.33)),
           frame=dict(kd=(0.62, 0.63, 0.65)))

# ======================= table (fixed) =======================
TW, TD, TT = L["table"]["size"]
o = Obj()
o.box((0, 0, H - TT / 2), (TW, TD, TT), {("z", 1): "bamboo", "default": "bamboo_side"},
      uvfun=lambda k, p: ((p[0] + TW / 2) / 0.20, (p[1] + TD / 2) / 0.14))
for sx in (-1, 1):                                                       # standing-desk legs + feet, visual only
    o.box((sx * (TW / 2 - 0.09), 0.0, (H - TT) / 2), (0.07, 0.05, H - TT - 0.002), {"default": "frame"})
    o.box((sx * (TW / 2 - 0.09), 0.0, 0.015), (0.07, 0.65, 0.03), {"default": "frame"})
o.box((0, 0.05, H - TT - 0.04), (TW - 0.25, 0.04, 0.06), {"default": "frame"})
o.write(f"{U}/table_vis.obj", {k: MAT[k] for k in ("bamboo", "bamboo_side", "frame")})
tcol = [dict(box=(TW, TD, TT), origin=(0, 0, H - TT / 2))]
write_urdf("table", [dict(name="table", inertial=inertial_xml(40.0, (0, 0, H - TT / 2), inertia_box(40, (TW, TD, TT))),
                          visual=[dict(mesh="table_vis.obj")], collision=tcol)])
obj("table", (0, 0, 0), (0, 0, 0), 40.0, box_volume(tcol), 0.5, fixed=True)

# ======================= laptop (free base, hinged lid) + stand =======================
LW, LD, TB, TL_, LG = 0.3557, 0.2481, 0.0108, 0.0055, 0.0015   # LG: closed gap; lid swing clears the deck edge by LG (> d_hat)
lp = L["laptop"]; psi, th = np.radians(lp["yaw"]), np.radians(lp["pitch"])
Rl = Rot.from_euler("xyz", [th, 0, psi])                                 # extrinsic: Rz(psi) Rx(theta)
n1 = Rl.apply([0, 0, 1])
deck_c = np.array([lp["base_center"][0], lp["base_center"][1], H + lp["base_center"][2]])
lap_origin = deck_c - TB * n1
o = Obj(); o.box((0, 0, TB / 2), (LW, LD, TB), {("z", 1): "deck", "default": "alu"},
                 uvfun=lambda k, p: ((p[0] + LW / 2) / LW, (p[1] + LD / 2) / LD))
o.write(f"{U}/laptop_base_vis.obj", {"deck": dict(kd=(1, 1, 1), map="laptop_deck.png"), "alu": MAT["alu"]})
hinge = (0, LD / 2, TB + LG + TL_ / 2)
o = Obj(); o.box((0, -LD / 2, 0), (LW, LD, TL_), {("z", -1): "screen", "default": "alu"},     # lid ends AT the hinge axis
                 uvfun=lambda k, p: ((p[0] + LW / 2) / LW, -p[1] / LD))
o.write(f"{U}/laptop_lid_vis.obj", {"screen": dict(kd=(1, 1, 1), map="laptop_lid_inner.png"), "alu": MAT["alu"]})
bcol = [dict(box=(LW, LD, TB), origin=(0, 0, TB / 2))]; lcol = [dict(box=(LW, LD, TL_), origin=(0, -LD / 2, 0))]
write_urdf("laptop", [dict(name="base", inertial=inertial_xml(1.6, (0, 0, TB / 2), inertia_box(1.6, (LW, LD, TB))),
                           visual=[dict(mesh="laptop_base_vis.obj")], collision=bcol),
                      dict(name="lid", inertial=inertial_xml(0.55, (0, -LD / 2, 0), inertia_box(0.55, (LW, LD, TL_))),
                           visual=[dict(mesh="laptop_lid_vis.obj")], collision=lcol)],
           [dict(name="hinge", type="revolute", parent="base", child="lid", origin=hinge, axis=(-1, 0, 0),
                 lo=0.0, hi=np.radians(135), friction=0.8, damping=0.02, q0=np.radians(lp["lid_open_deg"]))])
obj("laptop", lap_origin, np.degrees([th, 0, psi]), 2.15, box_volume(bcol) + box_volume(lcol), 0.8, kind="articulated",
    q0={"hinge": float(np.radians(lp["lid_open_deg"]))})
SC["objects"]["laptop"]["rho"] = float(2.15 / (box_volume(bcol) + box_volume(lcol)))

# stand, in the yaw-aligned frame S under the laptop origin (z = 0 at the table top)
S_org = np.array([lap_origin[0], lap_origin[1], H]); RS = Rot.from_euler("z", psi)
def L2S(p):                                                              # laptop-frame point -> stand frame
    return Rot.from_euler("x", th).apply(p) + np.array([0, 0, lap_origin[2] - H])
tray_lo, tray_hi = (-0.12, -0.135, -0.0055), (0.12, 0.095, -0.0015 - DH * 0.5)
def prism(lo, hi, f):                                                    # box given in laptop frame -> hull in S
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return trimesh.convex.convex_hull(np.array([f(p) for p in c]))
parts = [prism(tray_lo, tray_hi, L2S)]
for sx in (-1, 1):                                                       # front lips catching the laptop's front edge
    parts.append(prism((sx * 0.08 - 0.013, -LD / 2 - 0.0075, -0.001), (sx * 0.08 + 0.013, -LD / 2 - 0.0015, 0.008), L2S))
plate = trimesh.creation.box((0.20, 0.17, 0.005)); plate.apply_translation((0, -0.01, GAP + 0.0025)); parts.append(plate)
for sx in (-1, 1):                                                       # side arms: plate top -> just under the tray
    ys = np.linspace(-0.09, 0.07, 2); pts = []
    for y in ys:
        ztop = L2S((0, y / np.cos(th), tray_lo[2]))[2] - 0.0015
        for x in (sx * 0.09 - 0.003, sx * 0.09 + 0.003): pts += [(x, y, GAP + 0.005 + 0.0015), (x, y, ztop)]
    parts.append(trimesh.convex.convex_hull(np.array(pts)))
stand = trimesh.util.concatenate(parts); stand.export(f"{U}/stand_col.obj")
sv = Obj()  # visual = same pieces, silver
stand_vis = stand.copy(); stand_vis.visual.vertex_colors = np.tile([190, 192, 196, 255], (len(stand_vis.vertices), 1)); stand_vis.export(f"{U}/stand_vis.ply")
write_urdf("laptop_stand", [dict(name="stand", inertial=mesh_inertial(stand, 0.9) if stand.is_volume else inertial_xml(0.9),
                                 visual=[dict(mesh="stand_vis.ply")], collision=[dict(mesh="stand_col.obj")])])
obj("laptop_stand", S_org, (0, 0, np.degrees(psi)), 0.9, sum(p.volume for p in parts), 0.8, mesh_collision=True)

# ======================= monitor (fixed stand, tilt hinge) =======================
mo = L["monitor"]; PW, PH, PT = mo["panel"]          # layout stores [width, height, thickness]
pc = np.array([mo["panel_center"][0], mo["panel_center"][1] + PT / 2, H + mo["panel_center"][2]])   # box centre (front face measured)
M_org = np.array([pc[0], 0.28, H]); hinge_w = np.array([pc[0], 0.325, H + 0.35])
o = Obj()
o.box((0, 0, GAP + 0.006), (0.28, 0.19, 0.012), {"default": "silver"})
o.box((0, 0.0725, GAP + 0.0135 + 0.17), (0.075, 0.035, 0.34), {"default": "silver"})
o.write(f"{U}/monitor_stand_vis.obj", {"silver": MAT["silver"]})
scol = [dict(box=(0.28, 0.19, 0.012), origin=(0, 0, GAP + 0.006)), dict(box=(0.075, 0.035, 0.34), origin=(0, 0.0725, GAP + 0.0135 + 0.17))]
prel = pc - hinge_w
o = Obj()
o.box(prel, (PW, PT, PH), {("y", -1): "front", "default": "back"}, uvfun=lambda k, p: ((p[0] - prel[0] + PW / 2) / PW, (p[2] - prel[2] + PH / 2) / PH))
o.box((0, -0.0108, 0), (0.10, 0.0215, 0.10), {"default": "back"})      # VESA bracket
o.write(f"{U}/monitor_panel_vis.obj", {"front": dict(kd=(1, 1, 1), map="monitor_front.png"), "back": dict(kd=(0.16, 0.16, 0.17))})
pcol = [dict(box=(PW, PT, PH), origin=tuple(prel)), dict(box=(0.10, 0.0215, 0.10), origin=(0, -0.0108, 0))]
write_urdf("monitor", [dict(name="stand", inertial=inertial_xml(4.5, (0, 0.02, 0.05), (0.03, 0.03, 0.03)),
                            visual=[dict(mesh="monitor_stand_vis.obj")], collision=scol),
                       dict(name="panel", inertial=inertial_xml(6.5, tuple(prel), inertia_box(6.5, (PW, PT, PH))),
                            visual=[dict(mesh="monitor_panel_vis.obj")], collision=pcol)],
           [dict(name="tilt", type="revolute", parent="stand", child="panel", origin=tuple(hinge_w - M_org), axis=(-1, 0, 0),
                 lo=np.radians(-5), hi=np.radians(21), friction=6.0, damping=0.5, q0=np.radians(mo["tilt_deg"]))])
obj("monitor", M_org, (0, 0, mo["yaw"]), 11.0, box_volume(scol) + box_volume(pcol), 0.6, kind="articulated", fixed=True,
    q0={"tilt": float(np.radians(mo["tilt_deg"]))})

# ======================= SAM 3D objects (visual) with procedural / hull collision =======================
def sam_obj(name, vis, col_mesh, mass, fit, z0, friction, extra_vis=()):
    m = trimesh.load(f"{A}/{vis}", process=False); m.export(f"{U}/{name}_vis.ply")
    col_mesh.export(f"{U}/{name}_col.obj")
    write_urdf(name, [dict(name=name, inertial=mesh_inertial(col_mesh, mass), visual=[dict(mesh=f"{name}_vis.ply")] + list(extra_vis),
                           collision=[dict(mesh=f"{name}_col.obj")])])
    obj(name, (fit["xy"][0], fit["xy"][1], H + z0), (0, 0, fit["yaw"]), mass, col_mesh.volume, friction, mesh_collision=True)

ms = trimesh.load(f"{A}/mouse_vis.ply", process=False)
hull = ms.convex_hull.simplify_quadric_decimation(face_count=300) if hasattr(ms.convex_hull, "simplify_quadric_decimation") else ms.convex_hull
hull = trimesh.convex.convex_hull(hull.vertices)
mfit = dict(FIT["mouse"]); mfit["xy"] = [mfit["xy"][0], mfit["xy"][1] + 0.012]
PAD_T = 0.003; pad_mid = GAP + PAD_T / 2 + DH
sam_obj("mouse", "mouse_vis.ply", hull, 0.125, mfit, pad_mid + PAD_T / 2 + DH + 0.002, 0.5)
kb = trimesh.load(f"{A}/keyboard_vis.ply", process=False); kx, ky, kz = kb.extents
kcol = trimesh.creation.box((kx, ky, min(kz, 0.040))); kcol.apply_translation((0, 0, min(kz, 0.040) / 2))
kb_c = (kb.bounds[0] + kb.bounds[1]) / 2; kcol.apply_translation((kb_c[0], kb_c[1], 0))
sam_obj("keyboard", "keyboard_vis.ply", kcol, 0.95, FIT["keyboard"], GAP, 0.8)
sam_obj("bottle", "bottle_vis.ply", trimesh.load(f"{A}/bottle_col.obj"), 0.33, FIT["bottle"], GAP, 0.5)
# tea tin: split the SAM 3D visual at the lid seam; collision = procedural body / slip-fit lid
T = SP["tea_tin"]; tv = trimesh.load(f"{A}/tea_tin_vis.ply", process=False)
seam = T["lid_seated_z"] + 0.0005
fz = tv.vertices[tv.faces].mean(1)[:, 2]
body_v = tv.submesh([np.nonzero(fz < seam)[0]], append=True); lid_v = tv.submesh([np.nonzero(fz >= seam)[0]], append=True)
lid_v.apply_translation((0, 0, -T["lid_seated_z"]))
body_v.export(f"{A}/tin_body_vis.ply"); lid_v.export(f"{A}/tin_lid_vis.ply")
tf = FIT["tea_tin"]
sam_obj("tin", "tin_body_vis.ply", trimesh.load(f"{A}/tin_col.obj"), 0.12, tf, GAP, 0.35)
sam_obj("tin_lid", "tin_lid_vis.ply", trimesh.load(f"{A}/tin_lid_col.obj"), 0.035, tf, GAP + T["lid_seated_z"], 0.35)
SC["objects"]["tin_lid"]["mates"] = "tin"

# ======================= procedural-only parts =======================
def proc_obj(name, col, mass, pos, euler, friction, rgba, stripe=None, **kw):
    m = trimesh.load(f"{A}/{col}_col.obj"); m.export(f"{U}/{name}_col.obj")
    v = m.copy(); c = np.tile((np.array(rgba) * 255).astype(np.uint8), (len(v.vertices), 1))
    if stripe:   # a pale wedge (stripe = width in degrees) so a spinning round part visibly turns on camera
        ang = np.degrees(np.arctan2(v.vertices[:, 1], v.vertices[:, 0])); rad = np.linalg.norm(v.vertices[:, :2], axis=1)
        c[(np.abs(ang) < stripe / 2) & (rad > 0.6 * rad.max()), :3] = (215, 212, 200)   # outer band only: the hub vertex sits at angle 0
    v.visual.vertex_colors = c; v.export(f"{U}/{name}_vis.ply")
    write_urdf(name, [dict(name=name, inertial=mesh_inertial(m, mass), visual=[dict(mesh=f"{name}_vis.ply")],
                           collision=[dict(mesh=f"{name}_col.obj")])])
    paint = {} if stripe else dict(rgba=list(rgba))   # an rgba becomes a flat surface colour in scene.py and hides the stripe
    obj(name, pos, euler, mass, m.volume, friction, mesh_collision=True, **paint, **kw)
J = SP["jar"]; jx, jy = L["jar"]["xy"]; jy -= 0.014   # 1.4 cm forward: glass centre was least certain; clears the pad edge
proc_obj("jar", "jar", 0.38, (jx, jy, H + GAP), (0, 0, 0), 0.3, (0.80, 0.90, 0.88, 0.35), glass=True)
lx, ly = L["jar_lid"]["xy"]
proc_obj("jar_lid", "jar_lid", 0.02, (lx, ly, H + GAP + J["lid"]["height"]), (180, 0, 0), 0.25, (0.10, 0.10, 0.11, 1.0), stripe=24, mates="jar")
# bottle stick: leaning from the inner bottom against the rim, toward the back-left
B, S_ = SP["bottle"], SP["stick"]; bc = np.array(FIT["bottle"]["xy"]); d = np.array([-0.6, 0.8])
bot = np.r_[bc - d * (B["r_in"] - S_["r"] - 0.002), H + GAP + B["bottom"] + S_["r"] + 0.002]
rim = np.r_[bc + d * (B["r_in"] - S_["r"] - 0.0015), H + GAP + B["top"]]
u = (rim - bot) / np.linalg.norm(rim - bot); cen = bot + u * S_["length"] / 2
eul = Rot.align_vectors([u], [[0, 0, 1]])[0].as_euler("xyz", degrees=True)
proc_obj("stick", "stick", 0.002, cen, eul, 0.5, (0.85, 0.72, 0.50, 1.0), in_container="bottle")
# paper + marker
PAPER_T = 1e-4; p_bot = GAP + 0.0001 + DH; p_top = p_bot + PAPER_T + DH + 0.0005
for k, z in (("paper_bottom", p_bot), ("paper_top", p_top)):
    pm = sheet(0.2794, 0.2159, 0.007); pm.export(f"{U}/{k}.obj")
    SC["deformables"][k] = dict(mesh=f"urdf/{k}.obj", pos=[*L[k]["xy"], H + z], euler=[0, 0, L[k]["yaw"]], kind="paper",
                                rgba=[0.97, 0.97, 0.96, 1.0])
M = SP["marker"]; mk = L["marker"]; ma = np.radians(mk["yaw"]); dirm = np.array([np.cos(ma), np.sin(ma)])
total = M["cap_seated_z"] + M["cap_len"]
axis_z = H + p_top + PAPER_T / 2 + DH + 0.0008 + M["cap_r"]
butt = np.array(mk["xy"]) - dirm * total / 2
proc_obj("marker", "marker", 0.012, (butt[0], butt[1], axis_z), (0, 90, mk["yaw"]), 0.4, (0.95, 0.95, 0.95, 1.0))
capp = butt + dirm * M["cap_seated_z"]
proc_obj("marker_cap", "marker_cap", 0.004, (capp[0], capp[1], axis_z), (0, 90, mk["yaw"]), 0.4, (0.05, 0.05, 0.06, 1.0), mates="marker")
# mouse pad (thick cloth) and the folded microfibre cloth
pad = sheet(0.27, 0.32, 0.01); pad.export(f"{U}/mousepad.obj")
SC["deformables"]["mousepad"] = dict(mesh="urdf/mousepad.obj", pos=[*L["mousepad"]["xy"], H + pad_mid], euler=[0, 0, L["mousepad"]["yaw"]],
                                     kind="pad", rgba=[0.07, 0.08, 0.12, 1.0])
cl = L["cloth"]; CW, CLEN = 0.20, 0.45
cm = folded_cloth(CW, CLEN, gap=0.010, h=0.0075, fold_segments=6); cm.export(f"{U}/cloth.obj")
cy = np.radians(cl["yaw"]); back = np.array(cl["back_mid_xy"])
org = back + Rot.from_euler("z", cy).apply([-CW / 2, -CLEN, 0])[:2]
SC["deformables"]["cloth"] = dict(mesh="urdf/cloth.obj", pos=[*org, H + GAP + 0.001 + DH], euler=[0, 0, cl["yaw"]], kind="cloth",
                                  rgba=[0.74, 0.90, 0.62, 1.0], overhang=float(CLEN - cl["on_table_depth"]))
for k, d in SC["deformables"].items():
    m = trimesh.load(f"{A}/{d['mesh']}", process=False)
    M = np.eye(4); M[:3, :3] = Rot.from_euler("xyz", d["euler"], degrees=True).as_matrix(); M[:3, 3] = d["pos"]
    m.apply_transform(M); m.export(f"{A}/{d['mesh']}")
    d["placed_pos"], d["placed_euler"] = d["pos"], d["euler"]; d["pos"], d["euler"] = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
# ---- articulations -> separate single-link bodies + an IPC-native revolute joint (scene.json "joints")
def split(name, base_link, child_link, joint_xyz, axis_local, q0, lo, hi, child_fixed=False):
    """Write <name>_<link>.urdf for both links; child placed at its q=q0 pose. Joint pivot/axis in world."""
    o = SC["objects"].pop(name)
    T = np.eye(4); T[:3, :3] = Rot.from_euler("xyz", o["euler"], degrees=True).as_matrix(); T[:3, 3] = o["pos"]
    J = np.eye(4); J[:3, :3] = Rot.from_rotvec(np.array(axis_local, float) * q0).as_matrix(); J[:3, 3] = joint_xyz
    Tc = T @ J
    for ln, Tw, fixed in ((base_link, T, o.get("fixed", False)), (child_link, Tc, child_fixed)):
        write_urdf(f"{name}_{ln['name']}", [ln])
        SC["objects"][f"{name}_{ln['name']}"] = dict(urdf=f"urdf/{name}_{ln['name']}.urdf", pos=Tw[:3, 3].tolist(),
            euler=Rot.from_matrix(Tw[:3, :3]).as_euler("xyz", degrees=True).tolist(), mass=ln["mass"], rho=ln["rho"],
            friction=o["friction"], fixed=bool(fixed), part_of=name)
    SC.setdefault("joints", {})[name] = dict(a=f"{name}_{base_link['name']}", b=f"{name}_{child_link['name']}",
        pivot=Tc[:3, 3].tolist(), axis=(T[:3, :3] @ np.array(axis_local, float)).tolist(), q0=float(q0), lo=float(lo), hi=float(hi))

def link_rho(cols, mass): return float(mass / box_volume(cols))
split("laptop",
      dict(name="base", mass=1.6, rho=link_rho(bcol, 1.6), inertial=inertial_xml(1.6, (0, 0, TB / 2), inertia_box(1.6, (LW, LD, TB))),
           visual=[dict(mesh="laptop_base_vis.obj")], collision=bcol),
      dict(name="lid", mass=0.55, rho=link_rho(lcol, 0.55), inertial=inertial_xml(0.55, (0, -LD / 2, 0), inertia_box(0.55, (LW, LD, TL_))),
           visual=[dict(mesh="laptop_lid_vis.obj")], collision=lcol),
      hinge, (-1, 0, 0), np.radians(lp["lid_open_deg"]), 0.0, np.radians(135))
split("monitor",
      dict(name="stand", mass=4.5, rho=link_rho(scol, 4.5), inertial=inertial_xml(4.5, (0, 0.02, 0.05), (0.03, 0.03, 0.03)),
           visual=[dict(mesh="monitor_stand_vis.obj")], collision=scol),
      dict(name="panel", mass=6.5, rho=link_rho(pcol, 6.5), inertial=inertial_xml(6.5, tuple(prel), inertia_box(6.5, (PW, PT, PH))),
           visual=[dict(mesh="monitor_panel_vis.obj")], collision=pcol),
      tuple(hinge_w - M_org), (-1, 0, 0), np.radians(mo["tilt_deg"]), np.radians(-5), np.radians(21))
# ---- laptop space bar: a separate key on the deck, sliding along the deck normal (IPC-native prismatic joint + drive
# as the return spring). Placed from the deck texture (u 0.30-0.565, v 0.50-0.57); bottom KG above the deck so it can
# travel ~1 mm before the contact barrier stops it.
KW, KD, KT, KG = 0.092, 0.014, 0.0012, 0.0021
kc_local = np.array([-LW / 2 + 0.4325 * LW, LD / 2 - 0.535 * LD, TB + KG + KT / 2])
ob = SC["objects"]["laptop_base"]; Tb = np.eye(4); Tb[:3, :3] = Rot.from_euler("xyz", ob["euler"], degrees=True).as_matrix(); Tb[:3, 3] = ob["pos"]
kcol = [dict(box=(KW, KD, KT), origin=(0, 0, 0))]
o = Obj(); o.box((0, 0, 0), (KW, KD, KT), {"default": "keycap"}); o.write(f"{U}/laptop_key_vis.obj", {"keycap": dict(kd=(0.05, 0.05, 0.055))})
write_urdf("laptop_key", [dict(name="key", inertial=inertial_xml(0.003, (0, 0, 0), inertia_box(0.003, (KW, KD, KT))),
                               visual=[dict(mesh="laptop_key_vis.obj")], collision=kcol)])
kw = Tb[:3, :3] @ kc_local + Tb[:3, 3]
SC["objects"]["laptop_key"] = dict(urdf="urdf/laptop_key.urdf", pos=kw.tolist(), euler=ob["euler"], mass=0.003,
                                   rho=float(0.003 / box_volume(kcol)), friction=0.6, part_of="laptop")
SC["joints"]["laptop_key"] = dict(a="laptop_base", b="laptop_key", pivot=kw.tolist(), axis=(Tb[:3, :3] @ [0, 0, 1.0]).tolist(),
                                  q0=0.0, lo=-0.0015, hi=0.0005, kind="prismatic", drive=0.02, slip=1.0)
json.dump(SC, open(f"{A}/scene.json", "w"), indent=1)
for k, v in SC["objects"].items(): print(f"{k:13s} pos={np.round(v['pos'], 3)} euler={np.round(v['euler'], 1)} mass={v['mass']} rho={v['rho']:.0f}")
for k, v in SC["deformables"].items(): print(f"{k:13s} pos={np.round(v['pos'], 3)} yaw={v['euler'][2]:.1f}")
