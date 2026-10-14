"""frames.py video.mp4 out.jpg [n=8] [w=480]: evenly spaced frames, labelled with time, as a 2-row sheet."""
import sys, cv2, numpy as np
v, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 8; w = int(sys.argv[4]) if len(sys.argv) > 4 else 480
cap = cv2.VideoCapture(v); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 50; fr = []
for t in np.linspace(0, N - 1, n).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = cap.read()
    if not ok: continue
    f = cv2.resize(f, (w, int(f.shape[0] * w / f.shape[1]))); cv2.putText(f, f"{t/fps:.2f}s", (8, 28), 0, 0.8, (255, 255, 255), 2); fr.append(f)
half = (len(fr) + 1) // 2; rows = [np.hstack(fr[:half]), np.hstack(fr[half:] + [np.zeros_like(fr[0])] * (half - len(fr[half:])))]
cv2.imwrite(out, np.vstack(rows)); print(N, "frames")
