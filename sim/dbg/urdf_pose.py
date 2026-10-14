import sys, json, numpy as np, genesis as gs
A = "/home/ziga/projects/desk-real2sim/assets"; SC = json.load(open(f"{A}/scene.json"))
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=False))
ents = {}
for name in sys.argv[1:]:
    o = SC["objects"][name]
    ents[name] = scene.add_entity(gs.morphs.URDF(file=f"{A}/{o['urdf']}", pos=o["pos"], euler=o["euler"], fixed=bool(o.get("fixed", False)),
                                                 convexify=False, decimate=False, watertighten=0, merge_fixed_links=False))
scene.build()
for n, e in ents.items():
    print(n, "dofs", np.round(np.array(e.get_dofs_position()), 4))
    for l in e.links:
        aabb = np.array(l.get_AABB()) if hasattr(l, "get_AABB") else None
        print(f"  link {l.name:8s} pos {np.round(np.array(l.get_pos()), 4)} quat {np.round(np.array(l.get_quat()), 4)}  aabb {None if aabb is None else np.round(aabb, 3).tolist()}")
