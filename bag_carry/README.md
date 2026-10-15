# bag_carry

A Franka carries a loaded brown paper bag by one rope handle from one table to another, in Genesis 1.3.1 + libuipc.
The bag and the trash-bin liner follow CoDimRecon (arXiv 2609.36024, App. B.4 and Tab. 2): manifold thin shells
with paper or plastic-film materials, containers as separate rigid solids, nothing pinned, the robot the only
scripted thing. Built in October 2026 next to the desk scene in this repo; it shares no files with it.

**Videos:** the [bag-carry-v0.1 release](https://github.com/zzigak/desk-real2sim/releases/tag/bag-carry-v0.1).
- `bag_close_light.mp4`, `bag_close_heavy.mp4`: Blender Cycles, the camera follows the bag.
- `bag_wide_light.mp4`, `bag_wide_heavy.mp4`: the same runs, whole robot and both tables.

**What this folder holds:** the simulation (`sim/`), the generated assets (`assets/`), the Blender renderer
(`render/`), the sc cluster tools (`tools/`) and the per-run metrics (`out/*/paper.json`, `out/press/*.json`).
Paths assume `~/simquarry/genesis-world` ([simquarry/genesis-world](https://github.com/simquarry/genesis-world),
branch `ipc_rod_and_mpm`) and Blender 5.2; the Franka meshes come from that Genesis install.

## Scene

- Two tables (Poly Haven `wooden_table_02`, top 0.75 m; collision is a box per top), a Franka on a pedestal between
  them (Genesis' Franka with the palm collision replaced by a box, `assets/panda_hand_box.xml`).
- A kraft paper bag 20 x 12 x 22 cm with a flat bottom (`assets/bag/rope_bag.obj`, 1 cm grid, 1681 vertices) and two
  twisted-paper rope handles (`rope_handles.npy`, FEM rods, r 2.5 mm, 6 g/m). Each rope's last three vertices at
  both ends lie on bag-wall vertices and are glued there by libuipc `SoftVertexStitch` (kappa 1e5), like handles
  glued inside a real bag; rope-bag contact is off.
- `--load light` drops an empty can (15 g), a crumpled paper ball (5 g), a tea box (60 g) and a rubber duck (30 g,
  Poly Haven `rubber_duck_toy`) into the bag. `--load heavy` drops a jam jar (0.45 kg), a full can (0.38 kg), an
  orange (0.20 kg), a tea box (0.15 kg) and the duck: 1.2 kg. They fall in during a 5 s settle, staggered, within 4 cm
  of the centre line so they clear the handles.
- The robot pinches the front handle's apex (fingers closing across the handle plane, 20 N cap), lifts to 0.50 m above
  the table, carries the bag to table B, sets it down, stands it up and lets go (`carry_bag` in `sim/run_bag.py`).

## Materials and solver vs. the paper

| CoDimRecon (App. B.4, Tab. 2) | here |
|---|---|
| paper: "isotropic St. Venant-Kirchhoff membrane with hinge bending: density 761.9 kg/m3 (areal density 80 g/m2), Young's modulus 2.25 GPa for membrane and bending, Poisson's ratio 0.15, thickness 0.105 mm, contact activation distance 0.5 mm, and friction 0.4" | same numbers; membrane = libuipc `NeoHookeanShell` (libuipc has no StVK shell; the two agree at small strain); hinge stiffness 1.8 on this mesh, calibrated by cantilever to E h^3/12(1-nu^2) = 2.2e-4 N m (bending length 6.6 cm) |
| "plastic variant uses yield curvature 25 m^-1 with no hardening" | `StrainPlasticDiscreteShellBending`, yield 0.25 rad dihedral change (25 /m over 1 cm hinges) |
| plastic bag: "density 920 kg/m3, Young's modulus 60 MPa for membrane and bending, Poisson's ratio 0.45, thickness 0.03 mm ... friction 0.45" | same (trash task) |
| "up to 100 Newton iterations (velocity tolerance 2 x 10^-3 m/s), 4,000 PCG iterations (tolerance rate 10^-6), and 24 line-search iterations"; resistance 1e8; dt 0.01 s; "We add no artificial damping" | same; libuipc's defaults (0.05 m/s, 8 line searches) let a released sheet hover for 0.45 s |
| "independent finger PD controllers are limited to 20 N"; "the gripper holds it through contact alone" | 20 N on the bag, contact only |
| Franka "modeled with affine body dynamics" inside the IPC solve | every link a libuipc affine body tied to Genesis' articulation by its two-way soft constraint (`external_articulation` crashed: Genesis forbids `set_qpos` there) |

Friction smoothing: libuipc's default `eps_velocity` 0.01 m/s lets resting contents creep ~5 mm/s down the pile
inside the bag (the bag never came to rest); the carries use `--eps-v 1e-3`.

## Checks (`sim/check_bag.py`)

From the paper's behavioural checks (Sec. 3.4, App. B.4) plus robot checks: every bag edge within [0.9, 1.1] of rest
length every frame; drift < 1 mm and nodal speed < 2 mm/s over the last 0.5 s of settling; the bag rises >= 120 mm;
no robot link below the table top; hand within 10 deg of vertical; median grip within 20% of its cap; no frozen
frames; at the end the bag is on table B, standing (bottom-centre to rim-centre axis within 20 deg of vertical), and
every object's centre is inside the convex hull of the bag's final surface.

## Results

| load | verdict | stretch (min-max) | settle speed | rise | grip | end tilt | hand tilt | wall time |
|---|---|---|---|---|---|---|---|---|
| light | FAIL: not at rest when the robot starts (drift 0.12 mm) | 0.998-1.002 | 3.9 mm/s | 169 mm | 20.0 N | 1.6 deg | 0.13 deg | 31 min |
| heavy | PASS | 0.996-1.004 | 0.2 mm/s | 151 mm | 20.0 N | 2.1 deg | 0.21 deg | 24 min |

Both bags end standing on table B with every object inside. Light fails one check by a hair: a single 0.12 mm pop of
the bag's top rim corner (a plastic crease yielding) 0.5 s before the robot starts, 3.9 mm/s for one frame; the rest
of that window is at 0.4-0.5 mm/s. `out/<load>/paper.json` holds the per-frame log (bag stretch, bbox, rest metrics,
finger forces and gaps, object positions), the plan, `bag_end_verts` and the set-down numbers. These are the runs in
the videos (made with this code before it moved into this folder; only import paths changed).

The bag carried by one handle hangs 20-50 deg over (its centre of mass is 6 cm from the handle's wall and the rim
folds under the load). The Franka's 8 cm opening cannot take both handles (12 cm apart). So the set-down is a
feedback loop: straight down at 2 cm/s to first contact, then the hand walks back to the pose it grasped from (above
the bottom edge of the handle's wall, at the grasp height), re-measuring the bag every 0.1 s, and opens only when the
bag is within 6 deg of upright; then it centres on the rope, slides 7 cm along it out of the loop, and rises.

Things that went wrong on the way, each visible in a run and fixed in the robot motion or the check, not hidden:
- Lowering to a fixed height kept pushing after a heavier, sagging bag had landed; the slack rope rode up over a
  fingertip pad and the rising open hand hooked it and lifted the bag again.
- Letting go at first contact + 3 cm left the bag 22 deg over on one bottom edge; the contents slumped to the low wall
  and both bags toppled.
- The first version of the end check ("bag touches table B", "object inside the bag's bounding box") passed those
  toppled bags; the tilt and hull tests replaced it.

**Press test** (`sim/press_bag.py`, CoDimRecon's "replaying the same trajectory with plasticity disabled"): two
paddles squeeze the empty bag's ends by 4.5 cm each, hold, back off. Plastic hinges (paper text): width 20.0 cm at rest, 12.5 cm squeezed, 15.2 cm after release (48 mm kept); elastic control (`--elastic`): 12.9 cm squeezed, 17.3 cm after release (27 mm kept: an end wall stays snapped inward, which a thin sheet does without any plasticity). Edge length within 0.981-1.016 of rest in both (`out/press/`).

**Trash bin** (`--task trash`, work in progress): a 3-fold plastic liner over a tapered rigid basket, three objects
dropped in; the robot pinches the fold facing it 4 cm below the rim and pulls the bag fully out (44 cm rise) with the
objects inside, but run 7 still fails three checks: film stretch 0.88-1.13 near the pinch on the 1 cm mesh, settle
creep 13 mm/s, hand tilt 15 deg. Not rendered.

## Render (`render/render_bag.py`, Blender 5.2)

`sim/run_bag.py --export exp` writes every logged frame (33.3 fps) in the zzigak/yam-arcade export format
(`sim/recorder.py`): one OBJ per robot link and colour in the link frame plus its pose, the free objects' meshes and
poses, the bag surface (`meshseq/bag/*.npz`), the rope centrelines, `bag_rest.npy`. `render/render_bag.py` builds the
arcade studio with its compositor (`render/arcade/`, copied from zzigak/yam-arcade) and adds: a concrete floor and a
studio HDRI (`render/fetch_textures.sh`, Poly Haven CC0), charcoal table tops (so the kraft bag stands out), kraft
paper whose fibre pattern lives on the bag's rest-shape UVs, twisted cord tubes through the rope centrelines, and
render meshes for the objects that fit inside their simulated shapes (jam jar, cans, orange, crumpled paper ball, tea
carton). `--track x,y,z` makes the camera follow the bag's centre, Gaussian-smoothed over 0.7 s. Every pose and vertex
is the simulator's.

```bash
cd ~/simquarry/genesis-world
.venv/bin/python <repo>/bag_carry/sim/run_bag.py --task paper --load heavy --finger-armature 2.0 --eps-v 1e-3 \
    --export exp --out <repo>/bag_carry/out/heavy                     # ~25 min on an L40S / RTX PRO 6000
python3 <repo>/bag_carry/sim/check_bag.py <repo>/bag_carry/out/heavy/paper.json
cd <repo>/bag_carry && render/fetch_textures.sh
blender -b -P render/render_bag.py -- --export out/heavy/exp --out out/frames/close_heavy --res 1920x1080 \
    --samples 96 --track=0.70,-0.42,0.25 --lens 32 --track-lift 0.08 --orbit 20
```

On the sc cluster: `tools/run_on_sc.sh sim/run_bag.py ...` and `tools/render_on_sc.sh <export> <tag> 1920x1080 96 8
[render args]` (interleaved frame chunks; ~8 s per 1080p frame on an A5000). libuipc's CUDA backend is sm_89 only, so
the simulation needs an Ada/Hopper/Blackwell GPU; Blender renders on anything with OptiX.

`sim/make_bag_assets.py` regenerates `assets/bag/` (also older, unused shells: an open paper bag without handles and
a die-cut-handle bag that creased and slipped).
