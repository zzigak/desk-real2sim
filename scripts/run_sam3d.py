"""Batch SAM 3D Objects: one model load, many (photo, mask) jobs. Usage: run_sam3d.py OUTDIR name:photo:mask ..."""
import os, sys, time, json
sys.path.insert(0, os.environ["SAM3D_ROOT"])
from sam3d_infer import load_model, reconstruct
out = sys.argv[1]; jobs = [j.split(":") for j in sys.argv[2:]]
os.makedirs(out, exist_ok=True)
t = time.time(); model = load_model(); print(f"loaded {time.time()-t:.0f}s", flush=True)
for name, photo, mask in jobs:
    for seed in (42, 7):
        pre = f"{out}/{name}_s{seed}"
        if os.path.exists(pre + ".glb"): continue
        t = time.time()
        try:
            reconstruct(model, photo, mask, pre, seed=seed)
            print(f"{name} seed {seed}: {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"{name} seed {seed}: FAILED {e!r}", flush=True)
