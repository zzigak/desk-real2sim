import sys, numpy as np
sys.path.insert(0, "scripts"); from pix2table import cam_table, K, S_METRIC
def cast(photo, uv, n, c):
    """rays through pixels uv -> intersection with plane (n, c) given in METRIC table frame."""
    uv = np.atleast_2d(np.asarray(uv, float)); Rc, t = cam_table(photo); t = t * S_METRIC
    rays = (np.linalg.inv(K[photo]) @ np.c_[uv, np.ones(len(uv))].T).T @ Rc.T
    lam = ((c - t) @ n) / (rays @ n)
    return t + lam[:, None] * rays
if __name__ == "__main__":
    m = np.load("work/monitor_plane.npz"); n, c = m["n"], m["c"]
    s = 0.9701
    for name, q in {"outer": [(38, 62), (1248, 25), (1200, 728), (93, 726)],
                    "active": [(75, 98), (1208, 68), (1165, 698), (122, 695)]}.items():
        P = cast("p10", [(860 + x / s, 220 + y / s) for x, y in q], n, c)
        e = [np.linalg.norm(P[(k + 1) % 4] - P[k]) for k in range(4)]
        print(name, "edges", np.round(e, 3), "aspect", round((e[0] + e[2]) / (e[1] + e[3]), 3),
              "diag(in)", round(np.hypot((e[0] + e[2]) / 2, (e[1] + e[3]) / 2) / 0.0254, 1),
              "z top/bottom", np.round([P[:2, 2].mean(), P[2:, 2].mean()], 3), "center", np.round(P.mean(0), 3))
