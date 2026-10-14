"""Why does the marker cap hold harder in the desk scene than in the two-body rig? Stiff virtual hands (eta 100) hold the
barrel and draw the cap off along its axis at 25 mm/s from the desk pose, robot contact settings and robot fit geometry,
in scenes with more or fewer of the other desk objects.  python dbg/fit_bisect.py <full|nojoints|table|pair> <out>"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import scene as S, robot_desk as R, ipc_hands

mode, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
objs = list(S.SC["objects"])
keep = dict(full=objs, nojoints=[o for o in objs if not o.startswith(("laptop", "monitor"))],
            table=["table", "marker", "marker_cap"], pair=["marker", "marker_cap"], far=["marker", "marker_cap"],
            near=["marker", "marker_cap"], far_free=["marker", "marker_cap"], table_free=["table", "marker", "marker_cap"])[mode]
skip = tuple(o for o in objs if o not in keep) + tuple(S.SC["deformables"])
over = R.robot_assets()
dt = R.DT
extra_kw = json.loads(os.environ.get("COUPLER_JSON", "{}"))   # e.g. {"linear_system_tolerance": 1e-6}
scene, E, C = S.build(dt=dt, skip=skip, cameras=(), coupler_kw={**{k: v for k, v in R.CONTACT.items() if not k.startswith("constraint")}, **extra_kw},
                      urdf_override=over)
if mode in ("far", "near", "far_free"):   # a small box: 1 m away (bigger scene bounds only) or 5 cm under the marker
    p = (1.1, 0.8, 0.70) if mode != "near" else (0.10, -0.22, 0.735)
    scene.add_entity(gs.morphs.Box(pos=p, size=(0.10, 0.10, 0.02), fixed=mode != "far_free"), material=gs.materials.Rigid(coup_type="ipc_only"))
if mode != "full" and mode != "nojoints":
    pass


def T0(n):
    o = S.SC["objects"][n]; M = np.eye(4); M[:3, :3] = Rot.from_euler("xyz", o["euler"], degrees=True).as_matrix(); M[:3, 3] = o["pos"]
    return M


Tb, Tc = T0("marker"), T0("marker_cap"); ax = Tb[:3, 2]
mb = S.mass_props("marker", over["marker"]["urdf"].replace(".urdf", "_col.obj"), over["marker"]["rho"])
mc = S.mass_props("marker_cap", over["marker_cap"]["urdf"].replace(".urdf", "_col.obj"), over["marker_cap"]["rho"])
hb = ipc_hands.Hand("barrel", E["marker"].base_link, lambda t, T: Tb, Tb, mass=mb[0], I_link=mb[1], dt=dt)
def pull(t, T):
    M = Tc.copy(); M[:3, 3] += ax * 0.025 * max(t - 0.3, 0.0); return M
hc = ipc_hands.Hand("cap", E["marker_cap"].base_link, pull, Tc, mass=mc[0], I_link=mc[1], dt=dt)
ipc_hands.install([hb, hc])
scene.build()
co = scene.sim.coupler
for h in (hb, hc):
    (obj,) = co._ipc_objects.find(f"rigid_link_{h.link.idx}_0")
    g_anim = obj.geometries()
    g_slot = co._abd_slots_by_link[h.link][0].geometry()
    import uipc
    v_slot = np.asarray(g_slot.positions().view()).reshape(-1, 3)
    print(f"CHECK {h.name}: link idx {h.link.idx} ({h.link.name}), slot verts {len(v_slot)} centroid {np.round(v_slot.mean(0), 4)}, "
          f"abd_data transform {np.round(np.asarray(co._abd_data_by_link[h.link][0].transform)[:3, 3], 4)}, object id {obj.id()}", flush=True)
log = []
for i in range(int(1.0 / dt)):
    t = i * dt; hb.t = hc.t = t
    scene.step()
    rel = float((hc.T[:3, 3] - hb.T[:3, 3]) @ ax) - float((Tc[:3, 3] - Tb[:3, 3]) @ ax)
    log.append(dict(t=t, F=float(hc.F @ ax), off_mm=1000 * rel))
from genesis.utils.misc import tensor_to_array
print("CHECK genesis cap pos moved (mm):", np.round(1000 * (tensor_to_array(E["marker_cap"].base_link.get_pos()) - Tc[:3, 3]), 2),
      "barrel:", np.round(1000 * (tensor_to_array(E["marker"].base_link.get_pos()) - Tb[:3, 3]), 2), flush=True)
pk = max(log, key=lambda r: r["F"])
print(f"RESULT {mode} {extra_kw}: peak {pk['F']:.2f} N at {pk['off_mm']:.2f} mm off; final off {log[-1]['off_mm']:.1f} mm", flush=True)
json.dump(log, open(os.path.join(out, f"{mode}.json"), "w"))
