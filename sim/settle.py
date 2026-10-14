"""Whole desk, everything passive: let it settle, log per-object drift, render from the photo viewpoints."""
import argparse, json, os, time
import numpy as np
import genesis as gs
import scene as S

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=float, default=1.5); ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--out", default=os.path.join(S.ROOT, "out", "settle")); ap.add_argument("--skip", nargs="*", default=[]); ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--cams", nargs="*", default=["p10", "p09"]); ap.add_argument("--res", type=int, nargs=2, default=None)
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
gs.init(backend=gs.gpu, logging_level="warning")
skip = list(args.skip)
if args.only is not None:
    skip += [k for k in list(S.SC["objects"]) + list(S.SC["deformables"]) if k not in args.only and k != "table"]
scene, E, C = S.build(dt=args.dt, skip=skip, cameras=args.cams, res=args.res)
t0 = time.time(); scene.build(); print(f"build {time.time()-t0:.1f}s", flush=True)
rig = [k for k in E if k in S.SC["objects"]]
p0 = {k: S.link_pos(E[k])[0] for k in rig}
for k, c in C.items(): c.start_recording(save_to_filename=os.path.join(args.out, f"settle_{k}.mp4"), fps=50)
n = int(args.T / args.dt); t0 = time.time(); log = []
for i in range(n):
    scene.step()
    if i % 10 == 0 or i == n - 1:
        row = {"t": round((i + 1) * args.dt, 3), "ms_per_step": round((time.time() - t0) / (i + 1) * 1000)}
        for k in rig: row[k] = round(float(np.linalg.norm(S.link_pos(E[k])[0] - p0[k])) * 1000, 2)   # drift, mm
        for hn, h in E.get("_hinges", {}).items(): row[f"{hn}_deg"] = round(float(np.degrees(h.angle)), 3)
        for k in ("cloth", "paper_top", "mousepad"):
            if k in E:
                v = E[k].get_state().pos[0].cpu().numpy(); row[k + "_zmin"] = round(float(v[:, 2].min()), 4)
        log.append(row); print(row, flush=True)
for c in C.values():
    c.stop_recording()
json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
