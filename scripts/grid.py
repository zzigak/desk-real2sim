import sys, cv2
for f in sys.argv[1:]:
    im = cv2.imread(f); H, W = im.shape[:2]; s = 1288 / W
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for x in range(0, W, 100):
        c = (0, 0, 255) if x % 500 == 0 else (0, 200, 255)
        cv2.line(im, (int(x*s), 0), (int(x*s), im.shape[0]), c, 1)
        if x % 200 == 0: cv2.putText(im, str(x), (int(x*s)+2, 14), 0, 0.45, (0, 0, 255), 1)
    for y in range(0, H, 100):
        c = (0, 0, 255) if y % 500 == 0 else (0, 200, 255)
        cv2.line(im, (0, int(y*s)), (im.shape[1], int(y*s)), c, 1)
        if y % 200 == 0: cv2.putText(im, str(y), (2, int(y*s)-2), 0, 0.45, (0, 0, 255), 1)
    cv2.imwrite(f.replace("photos/", "work/grid_"), im)
