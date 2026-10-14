"""Robot fingertip on the mouse pad: press 10 N, then 25 N, then drag +x 6 cm at 25 N. Logs TCP height, the
impedance force the hand applies, the pad's mean x and the pad midsurface height under the fingertip.
python dbg/pad_press.py <out dir>"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
from genesis.utils.misc import tensor_to_array
import scene as S, robot_desk as R
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
d = R.Desk(keep=("mousepad",), cart_x=0.0, cams={})
V = lambda: tensor_to_array(d.E["mousepad"].get_state().pos)[0]
log = []
def rec(dd):
    v = V(); p = d.tcp("right"); i = np.argmin(np.linalg.norm(v[:, :2] - p[:2], axis=1))
    log.append(dict(t=round(dd.t, 4), tcp_z_mm=1000 * (p[2] - S.H), tcp_x=float(p[0]), F=R.hand_force("right").tolist(),
                    pad_mean_x=float(v[:, 0].mean()), pad_mid_z_mm=1000 * float(v[i, 2] - S.H), pad_min_z_mm=1000 * float(v[:, 2].min() - S.H)))
d.on_step.append(rec)
d.wait(0.5)
v = V(); start = np.array([float(v[:, 0].mean()) + 0.03, float(v[:, 1].mean()) - 0.03, S.H + 0.006])
d.go({}, 1.0, base_x=0.40)
d.cart.set_gripper("right", 0.0)
q = d.ik("right", start + [0, 0, 0.05]); d.go({"right": q}, 1.4)
d.go_paths({"right": ((start + [0, 0, 0.05], 0.0, 0.0), (start + [0, 0, 0.01], 0.0, 0.0))}, 0.8)
kz = 800.0
imp = d.impedance_on("right", k=(2500.0, 2500.0, kz), kr=(3.0, 3.0, 3.0), f_max=60.0, tau_max=2.0)
marks = {}
for press in (10.0, 25.0):
    imp.pos = start - [0, 0, press / kz]; d.wait(1.0); marks[f"press{press:.0f}"] = log[-1]
p0 = imp.pos.copy(); n = int(round(2.0 / d.dt))
for s_ in R.min_jerk(n):
    imp.pos = p0 + [0.06 * s_, 0, 0]; d.step(1)
d.wait(0.5); marks["drag_end"] = log[-1]
for k, r in marks.items():
    print(f"RESULT {k}: tcp z {r['tcp_z_mm']:.2f} mm, pad mid z under tip {r['pad_mid_z_mm']:.2f} mm, F {np.round(r['F'], 1)}, "
          f"pad mean x {r['pad_mean_x']:.4f}, tcp x {r['tcp_x']:.4f}", flush=True)
json.dump(log, open(os.path.join(out, "pad_press.json"), "w"))
