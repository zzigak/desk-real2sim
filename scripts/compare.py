"""Photo vs. sim render from the same (VGGT) camera: side by side + 50/50 blend. compare.py photo render.png out.jpg"""
import sys, cv2, numpy as np
ph, rd, out = sys.argv[1:4]
P = cv2.imread(f"photos/{ph}.jpg"); R = cv2.imread(rd)
P = cv2.resize(P, (R.shape[1], R.shape[0]), interpolation=cv2.INTER_AREA)
blend = cv2.addWeighted(P, 0.5, R, 0.5, 0)
for im, t in ((P, "photo"), (R, "simulation"), (blend, "50/50")):
    cv2.putText(im, t, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3); cv2.putText(im, t, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 1)
cv2.imwrite(out, np.hstack([P, R, blend]), [cv2.IMWRITE_JPEG_QUALITY, 88])
