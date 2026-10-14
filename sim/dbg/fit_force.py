"""Cap pull-off without any SoftTransformConstraint: the barrel is a fixed body and the cap is pulled along its axis by
a libuipc AffineBodyExternalBodyForce ramping 0 -> 20 N over 2 s. Breakaway = force when the cap has moved 1 mm.
python dbg/fit_force.py <pair|table|full> <out>"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import uipc
from uipc.constitution import AffineBodyExternalBodyForce
if os.environ.get("UIPC_INFO"):
    uipc.Logger.set_level(uipc.Logger.Level.Info)
import scene as S, robot_desk as R

mode, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
objs = list(S.SC["objects"])
keep = dict(full=objs, table=["table", "marker", "marker_cap"], pair=["marker", "marker_cap"])[mode]
skip = tuple(o for o in objs if o not in keep) + tuple(S.SC["deformables"])
S.SC["objects"]["marker"]["fixed"] = True
dt = R.DT
scene, E, C = S.build(dt=dt, skip=skip, cameras=(), coupler_kw={k: v for k, v in R.CONTACT.items() if not k.startswith("constraint")},
                      urdf_override=R.robot_assets())
o = S.SC["objects"]["marker"]; ax = Rot.from_euler("xyz", o["euler"], degrees=True).as_matrix()[:, 2]
st = dict(t=0.0, x0=None, x=None)
from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
orig = IPCCoupler._finalize_ipc


def _fin(self):
    ext = AffineBodyExternalBodyForce()
    self._ipc_constitution_tabular.insert(ext)
    link = E["marker_cap"].base_link
    ext.apply_to(self._abd_slots_by_link[link][0].geometry(), np.zeros(12))
    (obj,) = self._ipc_objects.find(f"rigid_link_{link.idx}_0")

    def anim(info):
        geo = info.geo_slots()[0].geometry()
        T = np.array(uipc.view(geo.transforms())[0]); st["x"] = T[:3, 3].copy()
        if st["x0"] is None: st["x0"] = st["x"].copy()
        F = 10.0 * max(st["t"] - 0.3, 0.0)
        uipc.view(geo.instances().find(uipc.builtin.is_constrained))[0] = 1
        uipc.view(geo.instances().find("external_force"))[0] = np.r_[ax * F, np.zeros(9)].reshape(12, 1)
    self._ipc_animator.insert(obj, anim)
    orig(self)


IPCCoupler._finalize_ipc = _fin
scene.build()
if os.environ.get("UIPC_INFO"):
    uipc.Logger.set_level(uipc.Logger.Level.Info)
log = []
for i in range(int(float(os.environ.get("T_END", 2.5)) / dt)):
    st["t"] = i * dt
    scene.step()
    if st["x"] is not None:
        log.append(dict(t=st["t"], F=10.0 * max(st["t"] - 0.3, 0.0), off_mm=1000 * float((st["x"] - st["x0"]) @ ax)))
br = next((r for r in log if r["off_mm"] > 1.0), None)
print(f"RESULT {mode}: breakaway {br['F'] if br else None} N (1 mm), final off {log[-1]['off_mm']:.1f} mm at F {log[-1]['F']:.1f} N", flush=True)
json.dump(log, open(os.path.join(out, f"force_{mode}.json"), "w"))
