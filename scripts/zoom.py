"""zoom.py photo x0 y0 x1 y1 out [step]: crop of the original photo with a labelled pixel grid."""
import sys, cv2
f, x0, y0, x1, y1, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
step = int(sys.argv[7]) if len(sys.argv) > 7 else 20
im = cv2.imread(f)[y0:y1, x0:x1]; s = min(1300 / (x1 - x0), 950 / (y1 - y0))
im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
for x in range((x0 // step + 1) * step, x1, step):
    big = x % (5 * step) == 0
    cv2.line(im, (int((x - x0) * s), 0), (int((x - x0) * s), im.shape[0]), (0, 0, 255) if big else (0, 220, 255), 1)
    if big: cv2.putText(im, str(x), (int((x - x0) * s) + 2, 12), 0, 0.4, (0, 0, 255), 1)
for y in range((y0 // step + 1) * step, y1, step):
    big = y % (5 * step) == 0
    cv2.line(im, (0, int((y - y0) * s)), (im.shape[1], int((y - y0) * s)), (0, 0, 255) if big else (0, 220, 255), 1)
    if big: cv2.putText(im, str(y), (2, int((y - y0) * s) - 2), 0, 0.4, (0, 0, 255), 1)
cv2.imwrite(out, im)
