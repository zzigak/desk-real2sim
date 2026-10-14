import sys, glob, json, numpy as np, trimesh
for f in sorted(glob.glob("sam3d/*.glb")):
    sc = trimesh.load(f)
    g = sc.dump(concatenate=True) if isinstance(sc, trimesh.Scene) else sc
    ext = g.bounding_box.extents
    vis = type(g.visual).__name__
    pose = json.load(open(f.replace(".glb", "_pose.json")))
    print(f"{f.split('/')[-1]:22s} V={len(g.vertices):7d} F={len(g.faces):7d} watertight={g.is_watertight} ext={np.round(ext,3)} visual={vis} scale={np.round(np.array(pose.get('scale', [0])).ravel()[:3],3)}")
