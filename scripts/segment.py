"""SAM 2.1 box/point-prompted masks. prompts.json: {photo: {obj: {"box":[x0,y0,x1,y1], "pos":[[x,y]], "neg":[[x,y]]}}}
in original pixel coords. Writes masks/<photo>_<obj>.png and work/seg_<photo>.jpg overlays."""
import json, os, sys
import numpy as np, torch, cv2
from PIL import Image
from transformers import Sam2Model, Sam2Processor

prompts = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
dev = "cuda"
proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(dev).eval()
os.makedirs("masks", exist_ok=True)
rng = np.random.default_rng(3)
for photo, objs in prompts.items():
    if only and photo not in only: continue
    img = Image.open(f"photos/{photo}.jpg").convert("RGB")
    over = np.array(img).copy()
    for name, p in objs.items():
        kw = {}
        if "box" in p: kw["input_boxes"] = [[p["box"]]]
        pts = [(q, 1) for q in p.get("pos", [])] + [(q, 0) for q in p.get("neg", [])]
        if pts:
            kw["input_points"] = [[[q for q, _ in pts]]]
            kw["input_labels"] = [[[l for _, l in pts]]]
        inp = proc(images=img, return_tensors="pt", **kw).to(dev)
        with torch.no_grad():
            out = model(**inp, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][0]  # (3,H,W)
        scores = out.iou_scores[0, 0].cpu().numpy()
        k = int(p.get("pick", np.argmax(scores)))
        m = masks[k].numpy().astype(bool)
        # keep largest connected component(s) unless asked otherwise
        if not p.get("multi", False):
            n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
            if n > 2:
                big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); m = lab == big
        cv2.imwrite(f"masks/{photo}_{name}.png", m.astype(np.uint8) * 255)
        col = rng.integers(60, 255, 3)
        over[m] = (0.45 * over[m] + 0.55 * col).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs):
            cv2.putText(over, name, (int(xs.mean()) - 40, int(ys.mean())), 0, 1.6, (255, 255, 255), 4)
        print(photo, name, "scores", np.round(scores, 3), "pick", k, "area", m.sum())
    small = cv2.resize(over.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"work/seg_{photo}.jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
