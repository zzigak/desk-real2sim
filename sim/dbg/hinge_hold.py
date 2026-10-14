"""Monitor tilt hinge: ramp a downward force on the panel's COM (53 mm in front of the hinge, so torque = F * 0.053
N m on top of the panel's own 3.4 N m) and log the hinge angle; the slip torque is where the angle runs away.
DESK_DRIVE_monitor / DESK_SLIP_monitor set the friction drive.  python dbg/hinge_hold.py <out dir>"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
import uipc
from uipc.constitution import AffineBodyExternalBodyForce
import scene as S, robot_desk as R
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
dt = R.DT
FULL = bool(int(os.environ.get("FULL", "0")))           # 1: the whole robot desk scene (cart + every object), arms idle
if not FULL:
    keep = ["table", "monitor_stand", "monitor_panel"]
    skip = tuple(o for o in S.SC["objects"] if o not in keep) + tuple(S.SC["deformables"])
    scene, E, C = S.build(dt=dt, skip=skip, cameras=(), coupler_kw={k: v for k, v in R.CONTACT.items() if not k.startswith("constraint")},
                          urdf_override=R.robot_assets())
st = dict(t=0.0)
F_RATE, T0 = float(os.environ.get("F_RATE", 40.0)), 0.5           # N per second, after a 0.5 s settle
from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
orig = IPCCoupler._finalize_ipc


def _fin(self):
    ext = AffineBodyExternalBodyForce()
    self._ipc_constitution_tabular.insert(ext)
    link = next(l for l in self._abd_slots_by_link if "monitor_panel" in str(getattr(l.entity.morph, "file", "")))
    ext.apply_to(self._abd_slots_by_link[link][0].geometry(), np.zeros(12))
    (obj,) = self._ipc_objects.find(f"rigid_link_{link.idx}_0")

    g0 = self._abd_slots_by_link[link][0].geometry()
    Xl = np.asarray(g0.positions().view()).reshape(-1, 3); T_ = np.array(uipc.view(g0.transforms())[0])
    print(f"CHECK panel IPC local verts min {np.round(Xl.min(0), 4)} max {np.round(Xl.max(0), 4)}; transform t {np.round(T_[:3, 3], 4)}", flush=True)
    st["T"] = T_
    POINT = os.environ.get("POINT")                   # "y,z" in the panel link frame: a horizontal +y point force there

    def anim(info):
        geo = info.geo_slots()[0].geometry()
        F = F_RATE * max(st["t"] - T0, 0.0)
        uipc.view(geo.instances().find(uipc.builtin.is_constrained))[0] = 1
        if POINT:
            T = np.array(uipc.view(geo.transforms())[0])
            yz = [float(c) for c in POINT.split(",")]
            pw = np.array([0.145, 0.325 + yz[0], 1.125 + yz[1]])          # world point on the panel (start pose)
            xb = np.linalg.solve(T[:3, :3], pw - T[:3, 3])                 # its local (rest) coordinates
            f = np.array([0.0, F, 0.0])
            g = np.r_[f, f[0] * xb, f[1] * xb, f[2] * xb]
        else:
            g = np.r_[0.0, 0.0, -F, np.zeros(9)]
        uipc.view(geo.instances().find("external_force"))[0] = g.reshape(12, 1)
    self._ipc_animator.insert(obj, anim)
    orig(self)


IPCCoupler._finalize_ipc = _fin
if FULL:
    d = R.Desk(keep=(), cart_x=0.0, cams={}); E = d.E

    class _Sc:
        def step(self): d.step(1)
    scene = _Sc()
else:
    scene.build()
h = E["_hinges"]["monitor"]; q0 = S.SC["joints"]["monitor"]["q0"]
log = []
for i in range(int(float(os.environ.get("T_END", 4.0)) / dt)):
    st["t"] = i * dt
    scene.step()
    F = F_RATE * max(st["t"] - T0, 0.0)
    tq = 3.4 + (F * -float(os.environ["POINT"].split(",")[1]) if os.environ.get("POINT") else 0.053 * F)
    log.append(dict(t=st["t"], F=F, torque=tq, tilt_deg=float(np.degrees(h.angle + q0)), aim=float(h.aim)))
for r in log[::15]:
    print(f"t {r['t']:.2f} F {r['F']:6.1f} N  torque {r['torque']:5.2f} N m  tilt {r['tilt_deg']:+.3f} deg  aim {r['aim']:+.5f}", flush=True)
slip = next((r for r in log if r["tilt_deg"] < log[int(T0 / dt)]["tilt_deg"] - 0.5), None)
print(f"RESULT drive {h.drive_strength:g} slip {h.slip:g}: runs away (0.5 deg) at {slip['torque'] if slip else None} N m", flush=True)
json.dump(log, open(os.path.join(out, "hinge_hold.json"), "w"))
