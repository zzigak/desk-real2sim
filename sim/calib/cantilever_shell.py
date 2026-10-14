"""Clamped cantilever for the desk shells (paper, mouse pad): a strip clamped between a ledge and a plate, a free
overhang L droops under gravity; the tip drop gives the effective bending rigidity through the heavy elastica.

    python calib/cantilever_shell.py --kind paper --bend 4.0 --out paper_4.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genesis as gs
import cloth_fix, elastica
from cloth_mesh import sheet
from scene import FABRIC, ipc_fabric

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True, choices=["paper", "pad", "cloth"]); ap.add_argument("--bend", type=float, required=True)
ap.add_argument("--L", type=float, default=0.10); ap.add_argument("--time", type=float, default=3.0); ap.add_argument("--out", required=True)
a = ap.parse_args()
f = ipc_fabric(a.kind, bending_stiffness=a.bend)
f.pop("bending_yield_strain", None); f.pop("bending_hardening_modulus", None)            # calibrate the elastic part
phys = FABRIC[a.kind]
h = {"paper": 0.007, "pad": 0.010, "cloth": 0.0075}[a.kind]
W, top, t = 0.03, 0.10, f["thickness"]                                     # t = contact radius = half the real thickness
gs.init(backend=gs.gpu, logging_level="warning")
cloth_fix.patch({t: (phys["E"] * phys["thickness"], phys["nu"])})
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
                 coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=1e-3, contact_resistance=1e7, newton_semi_implicit_enable=False),
                 show_viewer=False)
gap = t + 1e-3 + 3e-4                                                                      # contact radius + d_hat + margin
scene.add_entity(gs.morphs.Box(pos=(-0.15, 0, top / 2), size=(0.30, 0.10, top), fixed=True), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.8))
scene.add_entity(gs.morphs.Box(pos=(-0.15, 0, top + 2 * gap + 0.005), size=(0.30, 0.10, 0.01), fixed=True), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.8))
total = 0.10 + a.L
m = sheet(total, W, h); m.apply_translation((total / 2 - 0.10, 0, top + gap))
path = f"/tmp/cant_{a.kind}_{os.getpid()}.obj"; m.export(path)
strip = scene.add_entity(gs.morphs.Mesh(file=path), material=gs.materials.FEM.Cloth(**f))
scene.build()
p0 = strip.get_state().pos[0].cpu().numpy()
tip_i = np.nonzero(p0[:, 0] > a.L - h / 2)[0]; base_i = np.nonzero(p0[:, 0] < -0.02)[0]       # by index: the tip moves back as it droops
hist = []
for i in range(int(a.time / 0.01)):
    scene.step()
    if i % 10 == 9:
        p = strip.get_state().pos[0].cpu().numpy()
        hist.append(float(p[base_i, 2].mean() - p[tip_i, 2].mean()))
drop = float(np.mean(hist[-5:]))
w = phys["rho"] * phys["thickness"] * 9.81
json.dump(dict(kind=a.kind, bend=a.bend, L=a.L, drop=drop, hist=hist), open(a.out, "w"))   # raw first
from scipy.optimize import brentq
try:   # bracket on bending length 2 cm .. 1 m (paper / pad are far stiffer than the towel this solver was written for)
    c = float(np.exp(brentq(lambda lc: elastica.tip_drop(w * np.exp(lc) ** 3, w, a.L)[0] - drop, np.log(0.02), np.log(1.0))))
except Exception as e:
    print("elastica inversion failed:", e); c = float("nan")
B = w * c ** 3
json.dump(dict(kind=a.kind, bend=a.bend, L=a.L, drop=drop, B=B, bending_length=c, hist=hist), open(a.out, "w"))
print(f"{a.kind} bend={a.bend:g}: tip drop {drop*100:.2f} cm over L={a.L*100:.0f} cm -> B={B:.3e} N m, bending length {c*100:.2f} cm "
      f"(settled: last 0.5 s {min(hist[-5:])*100:.2f}-{max(hist[-5:])*100:.2f} cm)")
