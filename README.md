# desk-real2sim

Ziga's desk (10 phone photos, 2026-10-13) rebuilt as a Genesis 1.3.1 + libuipc scene: rigid and articulated
objects, real threads and slip fits, and cloth/paper/mouse-pad shells.

**Videos:** the [v0.1 release](https://github.com/zzigak/desk-real2sim/releases/tag/v0.1) has every clip.
- `hands_<demo>.mp4`: eleven interactions done by soft-constraint "hands" (`sim/demos.py`).
- `robot_<demo>.mp4`: six YAM-cart robot demos, rendered in Blender Cycles.
- `robot_jar_sim_preview.mp4`: the robot jar run (lid flipped, landed and screwed on), as an in-sim render only.

**What this repo holds:**
- the pipeline scripts (`scripts/`), the simulation (`sim/`) and the cluster tools (`tools/`, `cluster/`);
- the finished environment (`assets/`: meshes, photo textures, URDFs, `scene.json`, `specs.json`; `layout.json`);
- the per-run metrics (`out/**/*.json`).

**Not included:** the input photos and the pipeline's intermediate outputs (`photos/`, `recon/`, `masks/`, `sam3d/`
raw outputs, `work/`), so the scene runs as-is, but the photo-to-scene steps below can't be re-run from this repo
alone. Robot fit variants and generated robot MJCFs are rebuilt at run time.

**Sibling repos:**
- [zzigak/yam-arcade](https://github.com/zzigak/yam-arcade) provides the robot cart (`robots/yam_cart.py`), the
  contact settings and the Blender compositor (`render/compose.py`). Point `ARCADE` at a checkout of it.
- [zzigak/cloth-genesis](https://github.com/zzigak/cloth-genesis) holds the cloth calibration that the paths
  `~/projects/franka-deformables/...` refer to.

Paths assume `~/simquarry/genesis-world` ([simquarry/genesis-world](https://github.com/simquarry/genesis-world),
branch `ipc_rod_and_mpm`) and `~/projects/yam-arcade`.

## Pipeline

| step | script | output |
| --- | --- | --- |
| cameras, depth, points | `vggt-omega photos/p09.jpg ... -o recon/` (wide view first) | `recon/` |
| table plane, gravity | `scripts/align.py` | `work/align.json`, `work/topdown.png` |
| 2D masks (SAM 2.1, box prompts) | `scripts/segment.py work/prompts*.json` | `masks/<photo>_<obj>.png` |
| masks -> 3D, plane fits | `scripts/lift.py`, `scripts/planes.py`, `scripts/cyl_fit.py` | `work/lift.json`, `work/*_plane*.npz` |
| metric layout | `scripts/layout.py` | `layout.json` |
| SAM 3D meshes | `scripts/run_sam3d.py` (two seeds each) | `sam3d/` |
| SAM 3D -> assets | `sim/process_sam3d.py` (z-up, real size, decimate, yaw fit vs VGGT colours) | `assets/*_vis.ply`, `assets/sam3d_fit.json` |
| photo textures | `scripts/build_textures.py` | `assets/tex/` |
| threaded / slip-fit parts | `sim/build_assets.py` (`sim/geom.py`) | `assets/*_col.obj`, `assets/specs.json` |
| URDFs, world poses | `sim/build_scene_assets.py` then `scripts/export_cameras.py` | `assets/urdf/`, `assets/scene.json` |
| start-state check | `sim/validate_layout.py` | prints pairs closer than IPC allows |
| simulation | `sim/scene.py` (builder), `sim/settle.py`, `sim/demos.py` | `out/` |

## Measured scale and layout

- VGGT-Omega is non-metric. Point-cloud scale 1.37 m/unit, set by four agreeing lengths: Filco TKL keyboard
  356 x 138 mm and SteelSeries QcK pad 320 x 270 mm (1.366 - 1.379).
- Casting pixels onto the table plane reads ~4% long (Letter paper and the pad both), a focal/depth inconsistency
  inside VGGT. Positions from casting are kept, lengths from casting are divided by 1.04.
- Desk 1.25 x 0.76 m, top 0.775 m above the floor. Monitor = 32" Dell (active 0.698 x 0.392 m), tilt range -5..21 deg.
  Laptop = MacBook Pro 16: deck pitched 12.7 deg by the stand, lid open 113 deg, turned 29 deg toward the centre.

## Physics model

Genesis' rigid solver integrates the free bodies (its own collision OFF); libuipc does every contact, every joint and
every hand. All desk objects are `ipc_only`.

- **Fits** are calibrated with `sim/pull_test.py` (two-body rig, same contact settings). Any gap below IPC's 1 mm
  contact distance is pre-loaded by the barrier, which plays the part of the elastic interference in a real fit, so a
  fit's holding force is set by how far under 1 mm its gap is and over what area.
  - **Marker cap**: click ring. A 3 mm ring on the barrel meets a 3 mm rib in the cap mouth at 0.70 mm; everywhere
    else the gap is 1.15 mm or more. Pull-off 5.3 N (0.65 mm gave 7.7 N, 0.72 mm 4.5 N, 0.75 mm 3.3 N); once the rib
    clears the ring the cap is free.
  - **Tea tin lid**: slip fit, 0.9 mm radial gap over the 2 cm skirt. 2.7 N to start, 2.4 N to slide (lid weighs
    0.34 N), 3.3 N as the rim clears. At 0.7 mm it took 100-190 N.
  - **Jar lid**: single-start helical thread, h 3.0 mm, pitch 5.5 mm. The lid bore is the jar thread grown by a uniform
    1.2 mm (`geom.thread_envelope`): a bore offset only radially (old design, 1.4 mm) left the flanks 0.76 mm apart,
    inside d_hat, and needed 25-50 N m to turn. Now 0.04 N m running, 5.48 mm/turn; it meets the rim after about 1.1
    turns from one pitch up and the 1.5 N m wrist stalls at 1.3 turns. The neck has room for one turn of engagement at
    this pitch; a lid started two pitches up lands on the run-out and jams.
- **Laptop hinge, monitor tilt, space bar**: libuipc `AffineBodyRevoluteJoint` / `AffineBodyPrismaticJoint` plus a
  `Driving*` joint, added in an `IPCCoupler._finalize_ipc` hook (`sim/ipc_joints.py`). A friction hinge is the drive
  with stick-slip on its aim angle; the key is the prismatic drive at strength 0.02 (a soft spring) with a 1 mm stop
  under it. The photographed angles are baked into the joint origins (q = 0 is the scene as shot).
- **Hands** (`sim/ipc_hands.py`): `SoftTransformConstraint` on a rigid body. libuipc's affine-body transform is
  COM-centred, so each hand stores D = T_ipc^-1 T_link at the first frame and aims T_target D^-1. The spring is
  k = eta * m / dt^2 (eta 100: stiff enough to drag a part through any fit). The marker, tin and jar demos use
  person-like hands instead: stiffness K 1000 N/m (2000 bracing), K_rot 5 N m/rad for the jar wrist, and a strength
  cap (20 N, 1-1.5 N m) applied by keeping the aim within f_max / K of the part. Every hand with mass properties
  logs the force and torque it applied (`hand.F`, `hand.tau`).
- **Grasps on shells** (`sim/ipc_grasps.py`): `SoftPositionConstraint` on the vertices within a radius of the pick
  point, captured when the grasp starts and released by clearing `is_constrained`.
- **Cloth**: folded 0.40 x 0.45 m microfibre (two layers, 5 mm fold), bending 0.66 from
  `~/projects/franka-deformables/calib`. **Paper**: two Letter sheets, 0.1 mm shells, E 1 GPa, bending fitted to an
  80 gsm cantilever (`sim/calib/`), crease plasticity with yield 0.1 rad. **Mouse pad**: 3 mm shell, bending fitted to
  B = 3e-3 N m. Shell thickness is a contact radius in libuipc, so `scene.ipc_fabric` passes t/2 with doubled rho, E.
- SAM 3D meshes are visual only; collisions are procedural or convex hulls (`convexify=False`, `watertighten=0`).

## Demos (`sim/demos.py <name>`, outputs in `out/demos_v3/`)

`paper_fold`, `cloth_fold`, `cloth_drop`, `marker`, `pad_pick`, `bottle_tip`, `jar`, `laptop`, `monitor`, `key`,
`tin`. Each writes `<name>.mp4` and a `<name>.json` of per-step metrics (cap travel, turns, key travel, ...).

## Robot demos (`sim/robot_demos.py <name>`, outputs in `out/robot/<name>/`)

The same desk with the yam-arcade YAM cart (`VARIANT="yam"`, two YAM arms, 96 mm grippers) at the front edge
(`sim/robot_desk.py`). Arm joints are position-controlled from IK paths; fits are pulled with a Cartesian impedance
(`Desk.impedance_on`, joint torques J^T F with |F| capped) so two grippers on one fitted pair don't jam it. In libuipc
only the gripper pads and palms collide, so every arm motion is planned collision-free instead:

- **Planner** (`sim/robot_plan.py`, CPU, MuJoCo on the same MJCF; link poses agree with Genesis to 1e-5 m): damped
  least-squares IK with random restarts that keeps only collision-free solutions (nearest to the seed wins),
  `chain` for straight TCP lines (adaptive via count, every segment checked), `route` for free joint moves (straight,
  via the ready pose, or RRT-connect + shortcuts). `Desk.ik/go/go_paths/plan_pose/follow*` all go through it, and
  `Desk._run` checks both arms and the base moving together before anything moves. Contact a start pose already has
  may persist but not deepen.
- **Collision model** (`sim/robot_collide.py`): the I2RT link hulls, the cart's boxes, the tabletop and every rigid
  object's URDF collision geometry at its current pose (objects within 6 cm of a closing hand are treated as held).
  Same-arm pairs need no penetration, everything else 2 mm clearance. link4 vs palm overlaps across link5 in 97% of
  random poses and is skipped (`--calibrate`). Every collision box is turned into an 8-vertex convex mesh: MuJoCo
  3.10's box-box narrow phase reports bogus depths for boxes near each other (-46 mm for fingertip pads 23.5 mm
  apart) and `mj_geomDistance` flips signs now and then (+12.6 / -12.6 mm on consecutive frames of a still arm),
  while the mesh-mesh routine is consistent. An arm under Cartesian impedance is planned against where it is, not its
  stale joint target; gripper openings are checked too (`Desk.safe_widths`), and a hand that will be somewhere later
  can be planned against with `assume=` (the paper's fold arc is planned against the hold hand's pressed pose). `python sim/robot_collide.py --audit <states.npz>...` reports every
  penetration and every sheet vertex inside an arm link per recorded run.
- **Dry runs** (`sim/robot_dry.py <demo> --states <old run>_states.npz [--switch T:F]`): the routine on the CPU with
  perfect tracking and frozen objects, plus the audit; a planning bug costs seconds instead of a GPU job.
- **Cart placement** (`robot_demos.CART`, env `CART_LIFT` / `CART_BACK`): `back` parks the cart further from the edge
  and `lift` lowers the Flow Base lift carriage (deck + both arms; `robot_plan.cart_xml` writes the MJCF under
  `assets/robot_lift/`, rebuilt every run because its mesh dir is absolute). paper_fold: back 0.30, lift 0.15;
  cloth_fold: back 0.30, lift 0.20 (reaching to desk height in front of the edge from a deck level with the desk
  folds the wrist into the deck).

The pre-planner runs (2026-10-14 morning) audited badly: the marker's wrists overlapped by 8 cm, the cloth arm folded
through itself, the deck and the tabletop, the tin and bottle runs put the right upper arm through the jar, and the
cloth drape passed through the left arm's base plate (flush with the desk edge).

`marker`, `tin`, `jar`, `monitor`, `bottle`, `cloth_fold`, `cloth_drop`, `paper_fold`, `pad_pick`. Each writes
`<name>.mp4` (Genesis rasterizer), `<name>.json` (per-step metrics + notes), `<name>_states.npz` and `export/` in the
yam-arcade `export.Recorder` format. `--bare` drops the shells.

- **Contact settings** are the yam-arcade jars station's: d_hat 2e-4, contact_resistance 1e9, eps_velocity 1e-3,
  newton_tolerance 1e-3, dt 1/150, kappa 1e4 MPa on the fitted parts (`install_kappa`), the centroid coupling patch.
- **Fits** for the robot (`build_assets.ROBOT_FITS`, override with env `ROBOT_FITS_JSON`) were calibrated in the
  robot scene itself, because libuipc's hold on a preloaded fit (gap inside d_hat) depends on unrelated scene
  content: cap + barrel alone 0.8-4 N, add a table that doesn't touch them and it sticks at >22 N. The marker band
  is a cliff: 0.200 mm gap ~25 N, 0.205 mm 3.8-4.3 N, 0.210 mm 0.9 N. Jar: 87.8 mm lid (fits the 96 mm gripper),
  thread h 2.2 p 4.5 clear 0.4 mm. Tin: lid gap 0.5 mm, band 0.20 mm.
- **Hinges** (`sim/ipc_joints.py`): the friction drive holds kappa * slip, kappa = drive * (m_a + m_b) / dt^2,
  but it re-aims at most `slip` per step, so a stiff drive with a tiny slip is a speed limit, not friction. Pick a
  soft kappa (~1500 N m/rad) and slip = hold / kappa. Env `DESK_DRIVE_<joint>` / `DESK_SLIP_<joint>` override.
  The monitor panel's weight is 3.4 N m forward (COM 53 mm in front of the hinge); its bottom edge meets the stand
  neck at -8 deg.
- **Reach**: the YAM can't roll a lid 180 deg or point its gripper up; it can't reach the monitor's top bezel. The jar
  body (98 mm) is wider than the gripper, so it stays on the desk by friction and the hand's push.
- **Rendering**: `tools/render_on_sc.sh robot/<name> 1280x720 48` (Blender 5.2.2 Cycles with the yam-arcade
  compositor, `sim/render_desk.py`; any GPU, svl TitanRTX nodes work), `CHUNKS=N` to split, `--encode-only` to
  fetch and encode here (sc has no ffmpeg). Nyx (`sim/nyx_replay.py`, `sim/nyx_assets.py`) needs baked OBJ+PNG
  visuals: PLY visuals segfault it and it ignores vertex colours.
- **Running on sc**: `tools/run_on_sc.sh sim/robot_demos.py <name> --out out/robot/<run>` (Ada/Hopper/Blackwell only
  for libuipc; first run on a node compiles for ~5 min), `tools/fetch.sh robot/<run>`. When the queue is busy,
  `tools/run_pack_on_sc.sh "jar robot/r1" "monitor robot/r2 ENV=VAL"` runs several demos side by side in one 1-GPU job.
  On sc each run writes its own copy of the robot fit assets (`assets/urdf_robot/run_<host>_<pid>`): shared files
  regenerated by simultaneous runs on the network filesystem were read empty/truncated even with atomic renames.
- **Per-demo layout and hinge changes** live in `robot_demos.HINGE_ENV` (applied before the scene is built):
  monitor hinge drive/slip; `pad_pick` shifts the pad and mouse +6.4 cm in x (`DESK_SHIFT_<name>`, `DESK_EULER_<name>`
  in `scene.shift/euler`; `Desk._frame` includes them so pose readout and exports stay right).

### Robot demo outcomes (2026-10-14, collision-free planner)

Every run below has a clean `robot_collide.py --audit` (no arm link through anything, no link vertex-deep in a sheet)
and `sim/dbg/bystanders.py` shows no object the robot did not mean to touch moving (except as noted).

| Demo | Final run | Cart | Result | Render camera (`RARGS`, export frame) |
|---|---|---|---|---|
| marker | `c1_marker` | x 0.105 | cap off at 4.7 N, 53 mm; hands lean 30 deg outwards | demo camera |
| tin | `c1_tin` | x 0.15 | lid off at 6.6 N peak, rose 58 mm, set down beside the tin; lid grip pitched 34 deg | demo camera |
| bottle | `c2_bottle` | x 0.10 | stick out (jams in the neck ~1 s, ~130 N coupling force, as in the old run) and dropped; bottle tipped, rolls off the back edge | `--eye=1.00,-0.30,1.02 --target=0.47,0.12,0.79 --lens 36` |
| pad_pick | `c5_pad_pick` | back 0.15, lift 0.10 | front pinch of the overhang, pad lifted 22 cm, mouse slides off; the swinging pad nudges the jar 2 cm | `--eye=1.35,0.10,1.12 --target=0.50,-0.12,0.84 --lens 28` |
| cloth_fold | `s2_cloth_fold` | back 0.30, lift 0.20 | softer cloth for this demo (`HINGE_ENV` `DESK_FABRIC_cloth=E=2e5,bending_stiffness=0.08`: bending length ~1.2 cm instead of 2.4, E t 200 N/m); left side edge pinched under the slab, laid up; 100% of the cloth on the desk | `--eye=-0.85,-0.22,0.95 --target=-0.24,-0.38,0.73 --lens 42` |
| paper_fold | `c8_paper_fold` | back 0.30, lift 0.15 | marker removed (`DROP`); two-hand drag (the sheet slips: 2-3 cm overhang instead of 4.5), both front corners pinched, edge swung over the middle line; left holds, right presses the crease. 82% of the front half on the back half, 47% within 15 mm of a clean fold, crease 26 mm high. Holding is joint control (an impedance-held hand jittered 2-4 mm) | `--eye=0.12,0.18,1.10 --target=0.12,-0.30,0.75 --lens 32` |
| monitor | - | | **no tilt** (unchanged; see the earlier runs: a fingertip push does not slip the friction hinge) | - |
| jar | `j1_jar` | x 0.35 -> 0.45 | two-hand flip without handover (`_flip_lid`: left pivots the lid ~65 deg on its edge, right takes it and pushes it over, again near the top if it stands on its rim), staging next to the jar, diametral grasp, landing 0.7 mm off axis, 4 impedance strokes (`JAR_STROKE` default 600 N/m z, 4 N down, 15 N m/rad tilt, 3 N m/rad twist, 0.6 N m cap; softer tilt let the lid cock 10-25 deg about the pad line and jam): 2.35 turns, seated (-0.6 mm), 0.02 mm off axis | demo camera |

Failed attempts on the way, kept in `out/robot/`: `c1_pad_pick` (side pinch from beyond the desk only reaches with a
folded arm), `c2/c3_pad_pick` (bite 1 cm into the pad edge, closed on nothing; a finger-width reading can't tell),
`c4_pad_pick` (the hand knocked the jar off the desk in a free move: hands are now checked in transit),
`c1..c5_paper_fold` (fold arc vs the pressing hand; the soft rotational hold impedance let it tip), `c6_paper_fold`
(the hold 6 cm from the crease let the corner slide over the crease line instead of folding: "past the crease" is not
a fold, the metric is now the distance to the mirrored position), `c2/c3_cloth_fold` (after laying the corner, a
half-open hand counted as holding the cloth, dragged the pile over the front edge and it slid off).

Lessons from getting there (all in `sim/dbg/`): `monitor_reach.py`, `cloth_reach.py`, `cloth_lift_reach.py`,
`jar_reach.py`, `pad_reach.py`, `bottle_reach.py` (CPU IK scans with joint limits, run them before a GPU job);
`pad_press.py` (the pad sticks to the desk: 20 N drag under 22 N press, also at 1 mm thickness);
`hinge_hold.py` (friction-hinge slip torque, isolated or `FULL=1`, force at the COM or `POINT=y,z`);
`tcp_trace.py` / `pad_box_trace.py` (FK of recorded runs). Traps: the fingertip pad boxes end at the TCP, but the
soft Genesis-IPC coupling lets the Genesis fingers read ~1.7 mm on a 3 mm pad (closed-on-nothing reads 0.4-0.8 mm);
a joint-space move to a pose under the tabletop cuts through the slab (go out in front, drop, slide in); top-down
over the jar mouth reaches only to a TCP height of ~0.97 m; Blender logs print "Error opening HIP dynamic library"
on CUDA nodes, so don't grep for "Error" to detect a failed render; negative `--eye` values need `--eye=`.

## Gotchas found building it

- Genesis rotates FEM meshes about their vertex centroid but rigid meshes about their origin: deformable meshes are
  exported already in world coordinates.
- Genesis' default `watertighten=5` re-wraps a non-convex collision mesh; it must be 0 for threads.
- IPC refuses any cloth edge shorter than the contact thickness: the fold arc is 6 segments on a 5 mm radius.
- On a nearly full GPU libuipc fails with a bogus `TripletMatrixView: row index out of submatrix range`; check
  `nvidia-smi` before debugging.
- Genesis `external_articulation` NaNs on the fixed-base monitor and `two_way` bodies resting on contact drift, hence
  the IPC-native joints.
- `np.linalg.svd(points)` on 120k points allocates 115 GB (full U); always `full_matrices=False`.
- A demo that "works" can still be wrong: read the force the hand used from its spring stretch before trusting a fit.
  Capping a stiff (eta 100) hand by re-aiming it at "part + f_max / k" makes it a damper of k dt (the part barely
  moves); switching to libuipc's external body force at the cap gives a light part a full step of force, so it
  overshoots and the force flips every step. A soft spring with a capped aim works.
- libuipc smooths friction below `contact/eps_velocity` (10 mm/s): a part under a steady load below its pull-off
  force still creeps slowly.
