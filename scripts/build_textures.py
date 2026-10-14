"""Rectify photo regions into textures (assets/tex/*.png)."""
import os, sys, numpy as np, cv2
sys.path.insert(0, "scripts")
from project_check import world_to_pix
OUT = "assets/tex"; os.makedirs(OUT, exist_ok=True)
def rectify(photo, quad, wh, out):
    img = cv2.imread(f"photos/{photo}.jpg"); w, h = wh
    Hm = cv2.getPerspectiveTransform(np.float32(quad), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    cv2.imwrite(f"{OUT}/{out}", cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_CUBIC)); print(out, w, h)
s = 1.2667
d2o = lambda q: [(280 + x / s, 660 + y / s) for x, y in q]
rectify("p10", d2o([(5, 72), (757, 13), (820, 468), (95, 550)]), (1024, 714), "laptop_lid_inner.png")        # TL TR BR BL
rectify("p10", d2o([(95, 552), (822, 472), (965, 733), (127, 945)]), (1024, 714), "laptop_deck.png")         # back-L back-R front-R front-L
rectify("p10", [(937.3, 321.0), (2105.3, 290.1), (2061.0, 939.5), (985.8, 936.4)], (1600, 899), "monitor_screen.png")
# monitor front face: black bezel around the screen photo (panel 0.750 x 0.459, active 0.698 x 0.392, centred 6 mm high)
scr = cv2.imread(f"{OUT}/monitor_screen.png"); pw, ph = 1720, 1053; ppm = pw / 0.750
front = np.full((ph, pw, 3), 22, np.uint8)
aw, ah = int(0.698 * ppm), int(0.392 * ppm); x0 = (pw - aw) // 2; y0 = (ph - ah) // 2 - int(0.006 * ppm)
front[y0:y0 + ah, x0:x0 + aw] = cv2.resize(scr, (aw, ah), interpolation=cv2.INTER_AREA)
cv2.putText(front, "DELL", (pw // 2 - 40, y0 + ah + int(0.028 * ppm)), cv2.FONT_HERSHEY_DUPLEX, 1.1, (190, 190, 190), 2)
cv2.imwrite(f"{OUT}/monitor_front.png", front)
# bamboo tile: inverse-warp a clean desk patch (front-left corner, world frame) from p09
x0w, x1w, y0w, y1w, res = -0.60, -0.40, -0.36, -0.22, 0.0004
gx, gy = np.meshgrid(np.arange(x0w, x1w, res), np.arange(y1w, y0w, -res))
P = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 0.775)]
uv, _ = world_to_pix("p09", P)
img = cv2.imread("photos/p09.jpg")
tile = cv2.remap(img, uv[:, 0].reshape(gx.shape).astype(np.float32), uv[:, 1].reshape(gx.shape).astype(np.float32), cv2.INTER_LINEAR)
cv2.imwrite(f"{OUT}/bamboo.png", tile); print("bamboo", tile.shape, "covers", x1w - x0w, "x", y1w - y0w, "m")
