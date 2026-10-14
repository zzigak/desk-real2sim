"""Robot-hand versions of the fit demos: the wheeled bimanual YAM (robot_desk.py) does the work with friction grasps.
Only joint PD targets (IK + min-jerk joint paths), the base PD target and gripper commands are used; nothing on a desk
object is ever set. Per-step logs go to <out>/<demo>.json, the video to <out>/<demo>.mp4.

  python sim/robot_demos.py marker --out out/robot
"""
import argparse, json, math, os, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import genesis as gs
from genesis.utils.misc import tensor_to_array

import scene as S
import robot_desk as R

H = S.H


def axis_of(T):
    return T[:3, 2].copy()


def hand_quat(approach, close):
    """flange quaternion (w, x, y, z) for a gripper pointing along `approach` with its fingers closing along `close`
    (perpendicular to it)."""
    from yam_cart import quat_from_mat
    a = np.asarray(approach, float); a = a / np.linalg.norm(a)
    c = np.asarray(close, float); c = c - a * (c @ a); c /= np.linalg.norm(c)
    return quat_from_mat(np.column_stack([c, np.cross(a, c), a]))


# ------------------------------------------------------------------------------------------------ marker
def marker(d, log):
    """the right hand holds the barrel, the left hand grips the cap and pulls it straight off along the axis with both
    arms in Cartesian impedance. The cap is held only by its click ring (build_assets.ROBOT_FITS, calibrated in this
    scene to about 12 N)."""
    d.wait(0.3)                                        # settle first: grasp heights come from the resting pose
    Tm, Tc = d.pose("marker"), d.pose("marker_cap")
    ax = axis_of(Tm)                                   # barrel -> cap (cap mouth sits at cap_seated_z along it)
    az = Tm[2, 3]                                      # axis height
    cap_c = Tc[:3, 3] + ax * 0.0275                    # middle of the 55 mm cap
    bar_c = Tm[:3, 3] + ax * 0.045                     # barrel, clear of the cap mouth
    z_g = az - 0.006                                   # pad ends 6 mm below the axis: the pads span the equator
    bx = 0.5 * (cap_c[0] + bar_c[0])
    # fingers close across the axis. Both hands top-down on a 9 cm marker put the two elbows through each other, so
    # each hand leans outwards along the axis (the cap hand's wrist towards the cap end, the barrel hand's towards
    # the barrel end): the arms stay apart
    cl = np.array([-ax[1], ax[0], 0.0]); cl /= np.linalg.norm(cl); zz = np.array([0.0, 0.0, 1.0])
    pL = np.array([cap_c[0], cap_c[1], z_g]); pR = np.array([bar_c[0], bar_c[1], z_g])
    grip = None
    for th in np.radians([30, 40, 20, 50]):
        aL = math.sin(th) * (-ax) - math.cos(th) * zz; aR = math.sin(th) * ax - math.cos(th) * zz
        for sL, sR in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
            qL_, qR_ = hand_quat(aL, sL * cl), hand_quat(aR, sR * cl)
            try:
                jL = d.ik("left", pL - 0.06 * aL, quat=qL_, base_x=bx); d.ik("left", pL, quat=qL_, base_x=bx, seed=jL)
                jR = d.ik("right", pR - 0.06 * aR, quat=qR_, base_x=bx); d.ik("right", pR, quat=qR_, base_x=bx, seed=jR)
            except RuntimeError:
                continue
            grip = (th, aL, aR, qL_, qR_); break
        if grip:
            break
    if grip is None:
        raise RuntimeError("marker: no collision-free two-hand grasp")
    th, aL, aR, qL_, qR_ = grip
    st = dict(c0=None)

    def rec(dd):
        Tm_, Tc_ = dd.pose("marker"), dd.pose("marker_cap")
        rel = float((Tc_[:3, 3] - Tm_[:3, 3]) @ axis_of(Tm_))
        st["c0"] = rel if st["c0"] is None else st["c0"]
        if dd.k % 5 == 0:
            pads = {k: R.FORCES.get(f"left_{k}", np.zeros(3)) for k in ("tip_left", "tip_right", "gripper")}
            tcpL, tcpR = dd.tcp("left"), dd.tcp("right")
            log.append(dict(t=round(dd.t, 4), cap_off_mm=1000 * (rel - st["c0"]), pull_N=float(R.hand_force("left") @ ax),
                            barrel_N=float(R.hand_force("right") @ ax), grip_cmd_left=float(np.mean(d.cart.finger_force("left"))),
                            pad_l_N=float(-pads["tip_left"] @ cl), pad_r_N=float(-pads["tip_right"] @ cl), palm_N=float(np.linalg.norm(pads["gripper"])),
                            cap_slip_mm=1000 * float((Tc_[:3, 3] - tcpL) @ ax), barrel_slip_mm=1000 * float((Tm_[:3, 3] - tcpR) @ ax),
                            cap_below_tcp_mm=1000 * float(tcpL[2] - (Tc_[2, 3])), width_left_mm=1000 * d.cart.gripper_width("left"),
                            cap_lat_um=1e6 * float(np.linalg.norm((np.linalg.inv(Tm_) @ Tc_)[:2, 3])),
                            cap_tilt_mdeg=1e3 * float(np.degrees(np.arccos(np.clip(Tc_[:3, 2] @ Tm_[:3, 2], -1, 1)))),
                            marker_z_mm=1000 * (Tm_[2, 3] - az)))
    d.on_step.append(rec)
    d.note(f"marker: axis {np.round(ax, 3)}, cap centre {np.round(cap_c, 3)}, barrel grip {np.round(bar_c, 3)}, base x {bx:.3f}, "
           f"hands lean {math.degrees(th):.0f} deg outwards")
    d.go({}, 1.4, base_x=bx)
    d.go({"left": d.ik("left", pL - 0.06 * aL, quat=qL_), "right": d.ik("right", pR - 0.06 * aR, quat=qR_)}, 2.0)
    d.cart.set_gripper("left", 0.040); d.cart.set_gripper("right", 0.040)
    d.wait(0.3)
    d.follow_many({"left": d.plan_pose("left", pL - 0.06 * aL, qL_, pL, qL_, n_via=6),
                   "right": d.plan_pose("right", pR - 0.06 * aR, qR_, pR, qR_, n_via=6)}, 1.0)
    d.cart.grasp("right", force=20.0); d.grip_until("right", 1.0)    # 20 N per pad: 2 x 0.63 x 20 = 25 N of pull
    d.cart.grasp("left", force=20.0); d.grip_until("left", 1.0)
    d.note(f"marker: gripped, widths L {d.cart.gripper_width('left') * 1e3:.1f} R {d.cart.gripper_width('right') * 1e3:.1f} mm")
    lift = np.array([0.0, 0.0, 0.03])
    d.follow_many({"left": d.plan_pose("left", pL, qL_, pL + lift, qL_, n_via=4),
                   "right": d.plan_pose("right", pR, qR_, pR + lift, qR_, n_via=4)}, 0.8)
    d.wait(0.2)
    d.note("marker: both arms to Cartesian impedance (stiff along the axis, soft across and in rotation); pulling the cap")
    # task frame: x = marker axis, y = horizontal across it, z = up
    fr = np.column_stack([ax, np.cross([0, 0, 1.0], ax) / np.linalg.norm(np.cross([0, 0, 1.0], ax)), [0, 0, 1.0]])
    # both arms compliant across the axis and in rotation: a parallel gripper sets the yaw of what it holds, so two stiff
    # grippers a fraction of a degree apart twist the cap against the barrel and lock the 3 mm click band
    d.impedance_on("right", k=(1500.0, 150.0, 150.0), kr=(0.5, 0.5, 0.5), frame=fr, f_max=25.0, tau_max=1.0)
    imp = d.impedance_on("left", k=(1500.0, 150.0, 150.0), kr=(0.5, 0.5, 0.5), frame=fr, f_max=25.0, tau_max=1.0)
    d.wait(0.3)
    p0 = imp.pos.copy()
    n = int(round(2.0 / d.dt))
    for i, s_ in enumerate(R.min_jerk(n)):
        imp.pos = p0 + ax * 0.05 * s_
        d.step(1)
    d.wait(0.4)
    d.impedance_off("left"); d.impedance_off("right")
    off = log[-1]["cap_off_mm"]
    d.note(f"marker: cap {off:.1f} mm off the seat; peak pull {max(r['pull_N'] for r in log):.2f} N")
    # set the cap and the marker down on the paper and let go, then back out along the approach
    legs = {}
    for side, aa in (("left", aL), ("right", aR)):
        pe, qe = d.cart.tcp_pose(side)
        legs[side] = d.plan_pose(side, pe, qe, np.array([pe[0], pe[1], z_g + 0.004]), qe, n_via=4)
    d.follow_many(legs, 0.8)
    d.open_wait({"left": 0.04, "right": 0.04})
    legs = {}
    for side, aa in (("left", aL), ("right", aR)):
        pe, qe = d.cart.tcp_pose(side)
        legs[side] = d.plan_pose(side, pe, qe, pe - 0.07 * aa, qe, n_via=4)
    d.follow_many(legs, 0.8)
    d.home(1.5)
    d.wait(0.5)
    return dict(cap_off_mm=log[-1]["cap_off_mm"], peak_pull_N=max(r["pull_N"] for r in log))


# ------------------------------------------------------------------------------------------------ tea tin
def tin(d, log):
    """the left hand holds the tin body from the side, the right hand grips the lid's skirt from above with its
    approach pitched 34 deg away from the robot (fingers closing across the lid; straight down is out of reach that
    far back) and pulls it straight up with both arms in Cartesian impedance; the lid's retention rib slides along the
    neck. Then the lid is set down beside the tin. The cart parks where neither arm passes through the jar on the
    right of the desk or the other arm."""
    from yam_cart import down_quat
    d.wait(0.3)
    Tb, Tl = d.pose("tin"), d.pose("tin_lid")
    tp = Tb[:3, 3]; z_lid0 = Tl[2, 3]
    yb = 0.012                                          # side grip: TCP this far past the tin axis (contact inside the pad tips)
    p_lid = np.array([tp[0], tp[1], z_lid0 + 0.0155])   # middle of the 31 mm skirt, on the axis

    def horiz(ang, sg):                                  # horizontal approach turned `ang` from +y towards +x
        ap_ = np.array([math.sin(ang), math.cos(ang), 0.0]); return ap_, hand_quat(ap_, sg * np.cross([0, 0, 1.0], ap_))
    choice, errs = None, []
    for bx in (0.10, 0.15, 0.05, 0.12, 0.0):
        for yaw, tilt in ((0.0, 0.6), (math.pi / 2, 0.6), (0.0, 0.75), (math.pi / 2, 0.75), (0.0, 0.9)):
            q_lid = down_quat(yaw, tilt); aR = np.array([0.0, math.sin(tilt), -math.cos(tilt)])
            try:
                jR0 = d.ik("right", p_lid - aR * 0.08, quat=q_lid, base_x=bx)
                jR = d.ik("right", p_lid, quat=q_lid, base_x=bx, seed=jR0)
                d.ik("right", p_lid + [0, 0, 0.07], quat=q_lid, base_x=bx, seed=jR)
            except RuntimeError as e:
                errs.append(str(e)); continue
            for ang in np.radians([60, 45, 75, 30, 90]):
                for sb in (-1.0, 1.0):
                    ap_, qq = horiz(ang, sb)
                    for zb in (0.080, 0.060):           # body grip height above the desk: the palm clears the keyboard
                        pb = np.array([tp[0], tp[1], S.H + zb]) + ap_ * yb
                        try:
                            jL0 = d.ik("left", pb - ap_ * 0.09, quat=qq, base_x=bx); jL = d.ik("left", pb, quat=qq, base_x=bx, seed=jL0)
                            d.config_free({"left": jL0, "right": jR0}, base_x=bx); d.config_free({"left": jL, "right": jR}, base_x=bx)
                        except RuntimeError as e:
                            errs.append(str(e)); continue
                        choice = (bx, ang, qq, pb, -ap_ * 0.09, aR, q_lid, tilt); break
                    if choice: break
                if choice: break
            if choice: break
        if choice: break
    if choice is None:
        raise RuntimeError(f"tin: no collision-free grasp pair; last: {errs[-3:]}")
    bx, ang, q_side, p_body, preL, aR, q_lid, tilt = choice
    pre = -aR * 0.08
    st = {}

    def rec(dd):
        if dd.k % 5:
            return
        Tb_, Tl_ = dd.pose("tin"), dd.pose("tin_lid")
        st.setdefault("rel0", Tl_[2, 3] - Tb_[2, 3])
        log.append(dict(t=round(dd.t, 4), lid_rise_mm=1000 * (Tl_[2, 3] - Tb_[2, 3] - st["rel0"]),
                        body_rise_mm=1000 * (Tb_[2, 3] - Tb[2, 3]), lift_N=float(R.hand_force("right")[2]),
                        hold_N=float(R.hand_force("left")[2]), lid_xy_mm=1000 * float(np.linalg.norm(Tl_[:2, 3] - Tl[:2, 3]))))
    d.on_step.append(rec)
    d.note(f"tin: body at {np.round(tp, 3)}, lid grip z {p_lid[2]:.3f}, body grip z {p_body[2]:.3f}, base x {bx:.3f}, "
           f"body hand approaches at {np.degrees(ang):.0f} deg from the front, lid hand pitched {np.degrees(tilt):.0f} deg")
    d.go({}, 1.4, base_x=bx)
    d.go({"left": d.ik("left", p_body + preL, quat=q_side), "right": d.ik("right", p_lid + pre, quat=q_lid)}, 2.2)
    d.cart.set_gripper("left", 0.096); d.cart.set_gripper("right", 0.096)
    d.wait(0.4)
    d.follow("right", d.plan_pose("right", p_lid + pre, q_lid, p_lid, q_lid, n_via=6, tol=4e-3, rtol=0.05), 0.8)
    d.follow("left", d.plan_pose("left", p_body + preL, q_side, p_body, q_side, n_via=10, tol=4e-3, rtol=0.05), 0.8)
    d.wait(0.5)
    eL, eR = d.tcp("left") - p_body, d.tcp("right") - p_lid
    d.note(f"tin: TCP error L {np.round(eL * 1e3, 1)} mm, R {np.round(eR * 1e3, 1)} mm; base {np.round(d.cart.qpos()[d.cart.base_dofs], 3)}; "
           f"pad/palm forces " + ", ".join(f"{k} {np.linalg.norm(v):.1f}" for k, v in R.FORCES.items() if np.linalg.norm(v) > 0.2))
    if max(np.linalg.norm(eL), np.linalg.norm(eR)) > 0.006:
        raise RuntimeError("tin: hands did not reach the grasp poses")
    d.cart.grasp("left", force=15.0); d.cart.grasp("right", force=12.0)
    d.grip_until("left", 1.2); d.grip_until("right", 0.6)
    d.note(f"tin: gripped, widths L {d.cart.gripper_width('left') * 1e3:.1f} R {d.cart.gripper_width('right') * 1e3:.1f} mm")
    # both arms compliant across the axis and in rotation (see marker), stiff along it
    d.impedance_on("left", k=(150.0, 150.0, 2000.0), kr=(0.5, 0.5, 0.5), f_max=25.0, tau_max=1.5)
    imp = d.impedance_on("right", k=(150.0, 150.0, 1500.0), kr=(0.5, 0.5, 0.5), f_max=20.0, tau_max=1.0)
    d.wait(0.3)
    d.note("tin: pulling the lid straight up")
    p0 = imp.pos.copy(); n = int(round(2.0 / d.dt))
    for s_ in R.min_jerk(n):
        imp.pos = p0 + np.array([0, 0, 0.06]) * s_
        d.step(1)
    d.wait(0.4)
    d.impedance_off("left"); d.impedance_off("right")
    d.note(f"tin: lid {log[-1]['lid_rise_mm']:.1f} mm up; peak lift {max(r['lift_N'] for r in log):.2f} N")
    # body hand lets go, backs off and goes home; the lid hand sets the lid down on a clear patch beside the tin
    d.open_wait({"left": 0.096})
    pl, ql = d.cart.tcp_pose("left")
    d.follow("left", d.plan_pose("left", pl, ql, pl + preL, ql, n_via=6, tol=4e-3), 0.8)
    d.go({"left": d.cart.home_qpos[d.cart.arm_dofs["left"]]}, 1.4)
    a, qa = d.cart.tcp_pose("right")
    route, errs = None, []
    for dx, dy in ((0.0, -0.11), (0.10, -0.06), (-0.11, -0.02), (0.11, 0.0), (0.06, -0.10), (-0.13, 0.0)):
        c = np.array([tp[0] + dx, tp[1] + dy, S.H + 0.0155 + 0.002])   # lid bottom 2 mm above the desk when let go
        b_ = np.array([c[0], c[1], a[2]])
        try:
            l1 = d.plan_pose("right", a, qa, b_, qa, n_via=8)
            l2 = d.plan_pose("right", b_, qa, c, qa, n_via=6, seed=l1[-1])
            l3 = d.plan_pose("right", c, qa, c - aR * 0.08, qa, n_via=4, seed=l2[-1])
        except RuntimeError as e:
            errs.append(str(e)); continue
        route = (l1, l2, l3); break
    if route is None:
        raise RuntimeError("tin: nowhere reachable to set the lid down: " + " || ".join(e[:110] for e in errs[-6:]))
    d.follow("right", route[0], 1.2); d.follow("right", route[1], 1.0)
    d.open_wait({"right": 0.096})
    d.follow("right", route[2], 0.8)
    d.home(1.5)
    d.wait(0.3)
    return dict(lid_rise_mm=log[-1]["lid_rise_mm"], peak_lift_N=max(r["lift_N"] for r in log),
                body_rise_max_mm=max(r["body_rise_mm"] for r in log))


# ------------------------------------------------------------------------------------------------ jar
def _flip_lid(d, J, Tl, lc):
    """turn the upside-down lid right side up with two hands, without a handover: the left hand pinches the lid's +x
    wall (top-down, fingers closing radially across the wall) and pivots the lid up about its -x edge, which stays on the
    desk, to about 70 deg; the right hand's closed fingers come against the lid's top face from the front and take its
    weight; the left lets go and backs away; the right pushes the lid on past vertical and it falls right side up on
    the -x side. (A hand that pinched the rising edge would end up where the lid falls.)"""
    from yam_cart import down_quat
    r_out, Lh = J["lid"]["r_out"], J["lid"]["height"]
    wall = r_out - 0.0015
    pin0 = np.array([lc[0] + wall, lc[1], S.H + 0.004])
    Rp = lambda ph: Rot.from_rotvec([0.0, -ph, 0.0])      # lifts the +x side about the -x edge
    def rot_q(ph, q):
        x_, y_, z_, w_ = (Rp(ph) * Rot.from_quat([q[1], q[2], q[3], q[0]])).as_quat(); return np.array([w_, x_, y_, z_])
    def lid_pose(ph, sl, piv):
        T = np.eye(4); T[:3, :3] = Rp(ph).as_matrix() @ Tl[:3, :3]; T[:3, 3] = piv + Rp(ph).apply(Tl[:3, 3] + [0, sl, 0] - piv); return T
    def push_plan(ph, piv, ly, bx, assume):
        """the right hand's push for the lid standing at tilt ph on its -x edge (piv): its closed -x finger's outer face
        (8.5 mm off the TCP along the closing axis +x, 15 mm up the pad) stops 3 mm short of the lid's top face at height
        h, then moves straight to -x until it is 35 mm past the pivot: the lid goes over vertical and falls on its -x
        side. -> (pre joints, [touch waypoints], [push waypoints], approach)"""
        errs_ = []
        for h in (0.045, 0.038, 0.052):
            if 2 * r_out * math.sin(ph) < h + 0.008:
                continue
            xf = piv[0] + h / math.tan(ph)
            K = np.array([xf, ly, S.H + h]); cx = np.array([1.0, 0.0, 0.0])
            for apd in ((-0.3, 0.7, -0.6), (-0.2, 0.6, -0.77), (-0.4, 0.6, -0.7), (0, 0.7, -0.7), (-0.3, 0.4, -0.85), (0, 0, -1)):
                a_ = np.asarray(apd, float); a_ /= np.linalg.norm(a_)
                cx = np.array([1.0, 0.0, 0.0]) - a_ * a_[0]; cx /= np.linalg.norm(cx)   # closing axis ~ +x, square to a
                qR = hand_quat(a_, cx)
                touch = K + 0.0085 * cx + 0.015 * a_ + 0.003 * cx
                pre = touch - a_ * 0.06 + [0.02, 0.0, 0.0]
                end = touch - [xf - piv[0] + 0.035, 0.0, 0.0]
                try:
                    j0 = d.ik("right", pre, quat=qR, base_x=bx, assume=assume)
                    l_in = d.plan_pose("right", pre, qR, touch, qR, n_via=6, seed=j0, base_x=bx, assume=assume)
                    l_push = d.plan_pose("right", touch, qR, end, qR, n_via=10, seed=l_in[-1], base_x=bx,
                                         assume={"left": d.cart.home_qpos[d.cart.arm_dofs["left"]]})
                except RuntimeError as e:
                    errs_.append(str(e)); continue
                return j0, l_in, l_push, a_, h
        raise RuntimeError("no push: " + " || ".join(x[:90] for x in errs_[-2:]))

    # fingers at their working widths before planning (the fingertips are part of every collision check)
    d.cart.set_gripper("left", 0.022); d.cart.set_gripper("right", 0.0); d.wait(1.0)
    choice, errs = None, []
    # the lid lies 6 cm from the desk's front edge, too close to the robot for the second hand to get in low: the left
    # hand may first slide it (pinched) back along +y
    for sl, bx, tq in [(s_, b_, t_) for s_ in (0.0, 0.06) for b_ in (0.35, 0.40, 0.30, 0.45) for t_ in (0.6, 0.9, 0.3)]:
        q0 = down_quat(0.0, tq)                           # fingers close along x (radially across the +x wall); the
        pin = pin0 + [0, sl, 0]                           # approach may pitch away from the robot
        piv = np.array([lc[0] - r_out, lc[1] + sl, S.H]); F0 = np.array([lc[0], lc[1] + sl, S.H])   # F0: top face centre
        for ph1 in np.radians([70, 65, 75, 60]):
            phs = np.linspace(0.0, ph1, 9)
            try:
                j_pre = d.ik("left", pin0 + [0, 0, 0.06], quat=q0, base_x=bx)
                j_pin = d.ik("left", pin0, quat=q0, base_x=bx, seed=j_pre)
                slide = d.plan_pose("left", pin0, q0, pin, q0, n_via=6, seed=j_pin, base_x=bx) if sl else [j_pin]
                lift, seed = [], slide[-1]
                for k_ in range(len(phs) - 1):
                    leg = d.plan_pose("left", piv + Rp(phs[k_]).apply(pin - piv), rot_q(phs[k_], q0),
                                      piv + Rp(phs[k_ + 1]).apply(pin - piv), rot_q(phs[k_ + 1], q0), n_via=2, seed=seed,
                                      tol=4e-3, rtol=0.05, base_x=bx); seed = leg[-1]
                    lift += leg if not lift else leg[1:]
            except RuntimeError as e:
                errs.append(f"lift {np.degrees(ph1):.0f}: {e}"); continue
            # the right hand: closed, its -x finger's outer face (8.5 mm off the TCP along the closing axis, 15 mm up the
            # pad) against the top face, 60% of the way up it
            try:
                pp = push_plan(ph1, piv, lc[1] + sl, bx, {"left": lift[-1]})
            except RuntimeError as e:
                errs.append(f"push b{bx} t{tq} s{sl} ph{np.degrees(ph1):.0f}: {e}"); continue
            choice = (bx, ph1, j_pre, lift, q0, sl, slide, pin, piv); break
        if choice: break
    if choice is None:
        raise RuntimeError("jar: no two-hand flip found: " + " || ".join(x[:110] for x in errs[::max(1, len(errs) // 12)]))
    bx, ph1, j_pre, lift, q0, sl, slide, pin, piv = choice
    d.note(f"jar: flipping the lid: left hand slides it {100 * sl:.0f} cm back and pivots it up to {np.degrees(ph1):.0f} deg about "
           f"its -x edge, then the right hand pushes it over; base x {bx:.2f}")
    d.go({}, 1.4, base_x=bx)
    d.go({"left": j_pre}, 1.8); d.wait(0.3)
    d.follow("left", d.plan_pose("left", pin0 + [0, 0, 0.06], q0, pin0, q0, n_via=6, seed=j_pre), 1.0)
    d.cart.grasp("left", force=10.0); d.grip_until("left", 1.0)
    d.note(f"jar: lid wall pinched, width {d.cart.gripper_width('left') * 1e3:.1f} mm")
    if sl:
        d.follow("left", slide, 1.5)
    d.follow("left", lift, 2.5)
    if getattr(d, "dry", False):
        d._poses["jar_lid"] = lid_pose(ph1, sl, piv)
    d.wait(0.3)
    tilt = lambda: float(np.arccos(np.clip(-d.pose("jar_lid")[2, 2], -1, 1)))
    ph_now = tilt()
    d.note(f"jar: lid up at {np.degrees(ph_now):.0f} deg (aimed {np.degrees(ph1):.0f})")
    if ph_now < np.radians(35):
        raise RuntimeError("jar: the lid slipped out of the left hand's pinch")
    # the right hand's push, planned for the tilt the lid really has
    jR0, leg_in, push, a_, h = push_plan(ph_now, piv, lc[1] + sl, bx, {"left": d.cart.q_target[d.cart.arm_dofs["left"]]})
    d.go({"right": jR0}, 1.8)
    d.follow("right", leg_in, 1.0)
    d.wait(0.3)
    d.open_wait({"left": 0.04})
    pe, qe = d.cart.tcp_pose("left")
    from yam_cart import mat_from_quat
    apL = mat_from_quat(qe)[:, 2]
    for back_ in (0.06, 0.04, 0.03):
        try:
            d.move_pose("left", pe, qe, pe - apL * back_ + [0, 0, 0.02], qe, 0.8, n_via=5); break
        except RuntimeError:
            continue
    d.go({"left": d.cart.home_qpos[d.cart.arm_dofs["left"]]}, 1.4)
    d.note(f"jar: lid leaning on the right hand at {np.degrees(tilt()):.0f} deg; pushing it over at {1000 * h:.0f} mm up")
    d.follow("right", push, 2.0)
    if getattr(d, "dry", False):                          # (the dry run's stand-in for the fall: rim on the desk, or a
        d._poses["jar_lid"] = lid_pose(math.pi, sl, piv) + np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, Lh], [0, 0, 0, 0]])
        if os.environ.get("DRY_LID_AFTER_PUSH"):          # recorded pose, DRY_LID_AFTER_PUSH=states.npz:frame)
            f_, k_ = os.environ["DRY_LID_AFTER_PUSH"].rsplit(":", 1)
            d._poses["jar_lid"] = np.load(f_)["pose__jar_lid"][int(k_)]
            d._dry_lid_upright = lid_pose(math.pi, sl, piv) + np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, Lh], [0, 0, 0, 0]])
    d.wait(1.0)

    def back_off(side="right"):
        pe, qe = d.cart.tcp_pose(side)
        sx = 1.0 if side == "right" else -1.0
        for mv in ([0.05 * sx, 0, 0.05], [0.03 * sx, 0, 0.06], [0, -0.05, 0.05], [0, 0, 0.06]):
            try:
                d.move_pose(side, pe, qe, pe + mv, qe, 0.8, n_via=5); return
            except RuntimeError:
                continue
    back_off(); d.wait(0.5)
    # a lid pushed to vertical can stay standing on its rim (its 20 mm depth is a stable base, and a push at mid height
    # slides it): push its top face, near the top, along the lid's own axis; a lid that fell back gets the push again
    for attempt in range(3):
        T = d.pose("jar_lid"); zl = T[:3, 2]; up = float(zl[2])
        if up > 0.9:
            break
        if abs(up) < 0.45:                                # standing on its rim, closed top facing +zl
            ah = np.array([zl[0], zl[1], 0.0]); ah /= np.linalg.norm(ah)
            face = T[:3, 3] + zl * Lh                     # closed top's centre (the origin is the rim's centre)
            plan = None
            # whichever hand reaches the top face (it can end up facing any way), near the top, from any of these sides
            for side, hk, apd in [(s_, h_, a__) for s_ in ("right", "left") for h_ in (0.012, 0.02, 0.03)
                                  for a__ in ((0, 0, -1), (0, 0.6, -0.8), (0, 0.9, -0.4), (0.4, 0.6, -0.7), (-0.4, 0.6, -0.7),
                                              (-ah[0] * 0.6, -ah[1] * 0.6, -0.8))]:
                a_ = np.asarray(apd, float); a_ /= np.linalg.norm(a_)
                c_ = ah - a_ * (ah @ a_)
                if np.linalg.norm(c_) < 0.2:
                    continue
                c_ /= np.linalg.norm(c_)
                qT = hand_quat(a_, c_)
                K = face + np.array([0.0, 0.0, r_out - hk]) + ah * 0.003
                touch = K + 0.0085 * c_ + 0.015 * a_; pre = touch - a_ * 0.05 + ah * 0.02; end = touch - ah * 0.05
                try:
                    d.cart.set_gripper(side, 0.0)
                    j0 = d.ik(side, pre, quat=qT)
                    l1 = d.plan_pose(side, pre, qT, touch, qT, n_via=5, seed=j0)
                    l2 = d.plan_pose(side, touch, qT, end, qT, n_via=8, seed=l1[-1])
                except RuntimeError:
                    continue
                plan = (side, j0, l1, l2); break
            if plan is None:
                raise RuntimeError("jar: the lid stands on its rim and its top is out of reach")
            side = plan[0]
            d.note(f"jar: lid standing on its rim; the {side} hand pushes its top over (try {attempt + 1})")
            d.go({side: plan[1]}, 1.2); d.follow(side, plan[2], 0.8); d.follow(side, plan[3], 1.2)
            d.wait(1.0); back_off(side); d.wait(0.3)
            if getattr(d, "_dry_lid_upright", None) is not None:
                d._poses["jar_lid"] = d._dry_lid_upright
            if side == "left":
                d.go({"left": d.cart.home_qpos[d.cart.arm_dofs["left"]]}, 1.4)
            continue

        else:                                             # leaning back towards upside down: the mid-height push again
            ph_now = float(np.arccos(np.clip(-up, -1, 1)))
            d.note(f"jar: lid fell back to {np.degrees(ph_now):.0f} deg; pushing again (try {attempt + 1})")
            jR0, leg_in, push, a_, h = push_plan(max(ph_now, np.radians(40)), piv, T[1, 3], d.base[0], None)
            d.go({"right": jR0}, 1.2); d.follow("right", leg_in, 0.8); d.follow("right", push, 1.6)
        d.wait(1.0); back_off(); d.wait(0.5)


def jar(d, log):
    """the lid lies upside down; the robot turns it right side up with both hands (_flip_lid: one pivots it up on its
    edge, the other takes it and pushes it over), moves it next to the jar if it landed out of the screwing pose's
    reach, takes it top-down across its diameter, sets it on the jar's thread at the height for its measured phase,
    and screws it on in wrist strokes under Cartesian impedance (stiff against tilting, 4 N down, at most 0.6 N m of
    twist; a regrip between strokes) until it seats. The 98 mm jar is wider than the gripper's 96 mm opening, so only
    its weight, the table friction and the hand's push keep it from turning."""the lid lies upside down; the robot turns it right side up with both hands (_flip_lid: one pivots it up on its
    edge, the other takes it and pushes it over), takes it top-down across its diameter, sets it on the jar's thread at
    the right height for the lid's measured phase, and screws it on with jars-station wrist strokes (joint 6 torque-capped at 0.25 N m, regrasp between
    strokes) until it seats. The 98 mm jar is wider than the gripper's 96 mm opening, so only its weight, the table
    friction and the hand's downward push keep it from turning."""
    import build_assets as BA
    from yam_cart import down_quat
    grip_N = float(os.environ.get("JAR_GRIP", "8"))       # N per finger on the lid (env JAR_GRIP for tests)
    J = BA.make_jar(**BA.ROBOT_FITS["jar"])[1]
    P, r_out, Lh = J["p"], J["lid"]["r_out"], J["lid"]["height"]
    d.wait(0.3)
    Tj = d.pose("jar"); jx, jy = Tj[0, 3], Tj[1, 3]
    Tl = d.pose("jar_lid"); lc = Tl[:3, 3] + Tl[:3, 2] * Lh / 2            # lid centre (upside down: local z points down)
    bx = 0.45
    st = dict(yaw_u=0.0, yaw_p=None)

    def rel_yaw(dd):
        Rj, Rl = dd.pose("jar")[:3, :3], dd.pose("jar_lid")[:3, :3]
        Rr = Rj.T @ Rl
        return math.atan2(Rr[1, 0], Rr[0, 0]), Rr[2, 2]

    def rec(dd):
        if dd.k % 5:
            return
        Tl_, Tj_ = dd.pose("jar_lid"), dd.pose("jar")
        y, up = rel_yaw(dd)
        if st["yaw_p"] is not None and up > 0.9:
            st["yaw_u"] += (y - st["yaw_p"] + math.pi) % (2 * math.pi) - math.pi
        st["yaw_p"] = y
        log.append(dict(t=round(dd.t, 4), lid_up=float(up), lid_above_seat_mm=1000 * float(Tl_[2, 3] - Tj_[2, 3] - J["lid_seated_z"]),
                        lid_off_axis_mm=1000 * float(np.linalg.norm(Tl_[:2, 3] - Tj_[:2, 3])), turns=-st["yaw_u"] / (2 * math.pi),
                        jar_turns=0.0, wrist_Nm=float(tensor_to_array(d.cart.entity.get_dofs_control_force([int(d.cart.arm_dofs["right"][5])]))[0])))
    d.on_step.append(rec)
    d.note(f"jar: lid centre {np.round(lc, 3)} (up {Tl[2, 2]:+.2f}), jar at ({jx:.3f}, {jy:.3f}), base x {bx}")

    upright = Tl[2, 2] > 0.9
    if not upright:
        _flip_lid(d, J, Tl, lc)
        Tl = d.pose("jar_lid"); lc = Tl[:3, 3] + Tl[:3, 2] * Lh / 2
        if Tl[2, 2] < 0.9:
            raise RuntimeError(f"jar: the lid did not land right side up (up {Tl[2, 2]:+.2f})")
        d.note(f"jar: lid now right side up at {np.round(lc, 3)}")
    # the right hand takes the upright lid top-down across its diameter (96 mm opening on 87.8 mm: 4 mm a side, so it
    # aims at the lid's measured centre) and lifts it
    z_g = Tl[2, 3] + 0.004
    # the cart goes to the landing position (0.45: the top-down reach over the jar works from there), then the left arm
    # parks out to the left (the right reaches across to the lid)
    d.go({}, 1.4, base_x=0.45)
    lb = d.cart.base_pos + (d.base[0] - 0.305, 0.225, 0.75 - d.lift)      # the left arm base
    for park in ([-0.25, 0.05, 0.15], [-0.22, 0.10, 0.12], [-0.28, 0.0, 0.18], [-0.18, 0.12, 0.15], [-0.15, -0.05, 0.25]):
        try:
            d.go({"left": d.ik("left", lb + park, yaw=0.0, tol=5e-3, rtol=0.1)}, 1.4); break
        except RuntimeError:
            continue
    zc = Tj[2, 3] + J["lip_top"] + 0.012

    def grab_plan(c, bxs, carry):
        """right-hand top-down grasp of the upright lid at c across its diameter, from cart x in bxs; with carry, the
        rise to the lip height and the slide over the jar must be reachable too"""
        errs_ = []
        for bx_ in bxs:
            for yw in (0.0, math.pi / 2, math.pi / 4, -math.pi / 4):
                try:
                    j0 = d.ik("right", (c[0], c[1], z_g + 0.05), yaw=yw, tol=4e-3, rtol=0.05, base_x=bx_)
                    ldown = d.plan_pose("right", np.array([c[0], c[1], z_g + 0.05]), down_quat(yw), np.array([c[0], c[1], z_g]),
                                        down_quat(yw), n_via=6, seed=j0, base_x=bx_)
                    lup = d.plan_pose("right", np.array([c[0], c[1], z_g]), down_quat(yw), np.array([c[0], c[1], z_g + 0.04]),
                                      down_quat(0.0), n_via=8, seed=ldown[-1], base_x=bx_)
                    if carry:                             # the rise to the lip height and the slide over the jar
                        lr = d.plan_pose("right", np.array([c[0], c[1], z_g + 0.04]), down_quat(0.0), np.array([c[0], c[1], zc]),
                                         down_quat(0.0), n_via=10, seed=lup[-1], base_x=bx_)
                        d.plan_pose("right", np.array([c[0], c[1], zc]), down_quat(0.0), np.array([jx, jy, zc]), down_quat(0.0),
                                    n_via=10, seed=lr[-1], base_x=bx_)
                except RuntimeError as e:
                    errs_.append(str(e)); continue
                return bx_, yw, j0, ldown, lup
        raise RuntimeError(" || ".join(x[:100] for x in errs_[-3:]))

    def take(plan):
        bx_, yw, j0, ldown, lup = plan
        d.go({}, 1.4, base_x=bx_)
        d.cart.set_gripper("right", 0.096)
        d.go({"right": j0}, 1.6); d.wait(0.4)
        d.follow("right", ldown, 1.0)
        d.cart.grasp("right", force=grip_N); d.grip_until("right", 1.0)
        d.note(f"jar: right hand holds the upright lid across its diameter, width {d.cart.gripper_width('right') * 1e3:.1f} mm")
        return lup
    # the cart stays at 0.45 for the carry and the landing (no cart move with the lid in the hand: a held lid is not in
    # the collision model and would be carried through the jar). If the lid fell where that grasp + carry can't reach,
    # the right hand first moves it, low, to a staging spot that works, and takes it again.
    try:
        grab = grab_plan(lc, (0.45,), True)
    except RuntimeError as e0:
        grab, stage = None, None
        for st_ in ([0.44, -0.20], [0.42, -0.22], [0.46, -0.18], [0.40, -0.21], [0.43, -0.17]):   # 10-13 cm from the jar
            sc_ = np.array([st_[0], st_[1], lc[2]])
            try:
                grab_plan(sc_, (0.45,), True)
                pick = grab_plan(lc, (0.45, 0.40, 0.35, 0.30, 0.25, 0.50), False)
            except RuntimeError:
                continue
            stage = (sc_, pick); break
        if stage is None:
            raise RuntimeError(f"jar: the upright lid is out of reach ({e0})")
        sc_, pick = stage
        d.note(f"jar: the lid landed at {np.round(lc[:2], 3)}, out of the landing grasp's reach; moving it to {np.round(sc_[:2], 3)} first")
        lup0 = take(pick)
        d.follow("right", lup0[:max(2, len(lup0) // 2)], 0.6)  # 2 cm up
        p0, q0_ = d.cart.tcp_pose("right")
        d.follow("right", d.plan_pose("right", p0, q0_, np.array([sc_[0], sc_[1], p0[2]]), q0_, n_via=10), 1.4)
        d.follow("right", d.plan_pose("right", d.tcp("right"), q0_, np.array([sc_[0], sc_[1], z_g + 0.003]), q0_, n_via=4), 0.6)
        d.open_wait({"right": 0.096})
        pe, qe = d.cart.tcp_pose("right")
        d.follow("right", d.plan_pose("right", pe, qe, pe + [0, 0, 0.05], qe, n_via=4), 0.6)
        d.wait(0.3)
        Tl = d.pose("jar_lid"); lc = Tl[:3, 3] + Tl[:3, 2] * Lh / 2; z_g = Tl[2, 3] + 0.004
        grab = grab_plan(lc, (0.45,), True)
    lup = take(grab)
    d.follow("right", lup, 1.0)                           # lift 4 cm, turning to yaw 0 for the strokes (the lid turns along)
    Tl = d.pose("jar_lid")
    off = Tl[2, 3] - d.tcp("right")[2]
    # 5. land on the thread at the height that matches the lid's measured phase, about one pitch out
    y0, _ = rel_yaw(d)
    n_up = 1.0 + ((y0 / (2 * math.pi)) % 1.0)
    z_land = Tj[2, 3] + J["lid_seated_z"] + n_up * P + 0.4e-3
    yaw_hand = 0.0
    # top-down over the jar mouth only reaches to a TCP height of ~0.97 m (dbg/jar_reach.py): rise in place, slide over
    # the jar with the skirt just clear of the lip, set it down
    z_skirt = max(z_land + 0.006, Tj[2, 3] + J["lip_top"] + 0.004)   # slide across with the skirt 4 mm above the lip
    z_pre = z_skirt - off; pn = d.tcp("right")

    def lid_off():                                       # lid centre - TCP, horizontal: the lid sits anywhere along the pads
        T_ = d.pose("jar_lid"); return (T_[:3, 3] + T_[:3, 2] * Lh / 2 - d.tcp("right"))[:2]
    d.go_paths({"right": ((pn, 0.0, 0.0), ((pn[0], pn[1], z_pre), 0.0, 0.0))}, 1.0)
    o1 = lid_off(); tx, ty = jx - o1[0], jy - o1[1]
    d.go_paths({"right": (((pn[0], pn[1], z_pre), 0.0, 0.0), ((tx, ty, z_pre), 0.0, 0.0))}, 1.6, n_via=10)
    d.wait(0.2); o2 = lid_off(); tx, ty = jx - o2[0], jy - o2[1]
    d.note(f"jar: lid sits {1000 * np.linalg.norm(o2):.1f} mm off the TCP in the grip; aiming the lid at the jar axis")
    pc = d.tcp("right")
    d.go_paths({"right": ((pc, 0.0, 0.0), ((tx, ty, z_pre), 0.0, 0.0))}, 0.5)
    d.go_paths({"right": (((tx, ty, z_pre), 0.0, 0.0), ((tx, ty, z_land - off), 0.0, 0.0))}, 0.8)

    d.wait(0.3)
    d.note(f"jar: lid landed {log[-1]['lid_above_seat_mm']:.2f} mm above the seat (aimed {n_up:.2f} pitches), off axis {log[-1]['lid_off_axis_mm']:.2f} mm")
    # re-grip centred over the lid, so the wrist strokes turn it about its own axis (an off-centre grip swings it round)
    d.open_wait({"right": 0.096})
    T_ = d.pose("jar_lid"); c = T_[:3, 3] + T_[:3, 2] * Lh / 2; pc = d.tcp("right")
    d.go_paths({"right": ((pc, 0.0, 0.0), ((c[0], c[1], pc[2]), 0.0, 0.0))}, 0.6)
    d.cart.grasp("right", force=grip_N); d.grip_until("right", 0.8)
    jx, jy = float(c[0]), float(c[1])
    d.note(f"jar: re-gripped over the lid centre ({1000 * np.linalg.norm(c[:2] - pc[:2]):.1f} mm from the landing grip)")
    # 6. strokes, as the virtual-hand demo does it: the hand is stiff across the axis, soft along it with a constant 3 N
    # push down (a lid resting on the thread crests drops into the groove when its phase comes round; a height that
    # follows a schedule holds it on the crest and jams it), and twists clockwise from above with at most 0.3 N m.
    # The wrist has ~0.6 turn of travel: between strokes the hand lets go, turns back and grips again.
    dof6 = int(d.cart.arm_dofs["right"][5])
    # (env JAR_STROKE="kz,push_N,kr_xy,kr_z,tau_max" for tests)
    kz, push_N, kr_xy, kr_z, tau_cap = (float(x) for x in os.environ.get("JAR_STROKE", "600,4,15,3,0.6").split(","))
    if os.environ.get("JAR_BACKTURN", "0") == "1":
        # the way a person finds the thread: turn it backwards (counter-clockwise from above) under the same push until
        # it drops, the lid's thread start falling past the jar's, then screw it on
        q6a = float(d.cart.qpos()[dof6]); h0 = log[-1]["lid_above_seat_mm"]
        imp = d.impedance_on("right", k=(2000.0, 2000.0, kz), kr=(kr_xy, kr_xy, kr_z), f_max=15.0, tau_max=tau_cap)
        R0 = imp.R.copy(); span = max(0.0, q6a + 1.85); n = int(round((span / 0.8 + 0.4) / d.dt)); dropped = False
        for i in range(n):
            th = min(0.8 * (i * d.dt), span)
            imp.R = Rot.from_rotvec([0.0, 0.0, th]).as_matrix() @ R0
            tc = d.tcp("right")
            imp.pos = np.array([jx, jy, max(tc[2] - push_N / kz, Tj[2, 3] + J["lid_seated_z"] - off - 0.002)])
            d.step(1)
            if i % 10 == 0 and log[-1]["lid_above_seat_mm"] < h0 - 1.5:
                dropped = True; break
        d.impedance_off("right")
        d.note(f"jar: turned back {(q6a - float(d.cart.qpos()[dof6])) / (2 * math.pi):.3f} turn: lid {log[-1]['lid_above_seat_mm']:.2f} mm "
               f"above seat (was {h0:.2f}), {'dropped into the thread' if dropped else 'no drop'}")
        d.open_wait({"right": 0.096})
        q = d.cart.q_target[d.cart.arm_dofs["right"]].copy(); q[5] = -1.9; d.go({"right": q}, 1.0)
        d.cart.grasp("right", force=grip_N); d.grip_until("right", 0.8)
    for k in range(8):
        q6a = float(d.cart.qpos()[dof6])
        if q6a > 0.6:                                   # too little wrist travel left: regrasp first
            d.open_wait({"right": 0.096})
            q = d.cart.q_target[d.cart.arm_dofs["right"]].copy(); q[5] = -1.9; d.go({"right": q}, 1.0)
            d.cart.grasp("right", force=grip_N); d.grip_until("right", 0.8)
            q6a = float(d.cart.qpos()[dof6])
        lid0, h0 = log[-1]["turns"], log[-1]["lid_above_seat_mm"]
        imp = d.impedance_on("right", k=(2000.0, 2000.0, kz), kr=(kr_xy, kr_xy, kr_z), f_max=15.0, tau_max=tau_cap)
        R0 = imp.R.copy(); span = 1.85 - q6a; n = int(round((span / 1.2 + 0.4) / d.dt)); stalled = False
        for i in range(n):
            th = min(1.2 * (i * d.dt), span)
            imp.R = Rot.from_rotvec([0.0, 0.0, -th]).as_matrix() @ R0
            tc = d.tcp("right")                         # (never aimed below the TCP height of a seated lid)
            imp.pos = np.array([jx, jy, max(tc[2] - push_N / kz, Tj[2, 3] + J["lid_seated_z"] - off - 0.002)])
            d.step(1)
            if i > 60 and i % 15 == 0:
                lag = th - (float(d.cart.qpos()[dof6]) - q6a)
                if lag > 0.6:                           # the twist is capped and the lid is not turning
                    stalled = True; break
        d.impedance_off("right")
        q6 = float(d.cart.qpos()[dof6])
        d.note(f"jar: stroke {k + 1}: wrist {(q6 - q6a) / (2 * math.pi):.3f} turn, lid {log[-1]['turns'] - lid0:.3f} turn, "
               f"{log[-1]['lid_above_seat_mm']:.2f} mm above seat (was {h0:.2f}), stalled {stalled}")
        if log[-1]["lid_above_seat_mm"] < 0.3 or (stalled and log[-1]["lid_above_seat_mm"] < 1.0):
            break                                       # seated
        # regrasp: open, turn the wrist back, close
        d.open_wait({"right": 0.096})
        q = d.cart.q_target[d.cart.arm_dofs["right"]].copy(); q[5] = -1.9
        d.go({"right": q}, 1.0)
        d.cart.grasp("right", force=grip_N); d.grip_until("right", 0.8)
    d.open_wait({"right": 0.096})
    p, qn = d.cart.tcp_pose("right")                     # up off the lid (the reach ends near 0.97 m), the wrist as the
    leg, _, _ = d.plan_first("right", p, qn, [p + v for v in ([0, 0, 0.035], [0, -0.04, 0.03], [0, 0, 0.025])], n_via=6)
    d.follow("right", leg, 1.0)                           # strokes left it
    d.home(1.4); d.wait(0.5)
    return dict(turns=log[-1]["turns"], lid_above_seat_mm=log[-1]["lid_above_seat_mm"], lid_off_axis_mm=log[-1]["lid_off_axis_mm"],
                lid_up=log[-1]["lid_up"])


# ------------------------------------------------------------------------------------------------ monitor
# per-demo hinge settings, applied before the scene is built (the drive stiffness is read once, at init). The monitor's
# tilt hinge holds ~4.5 N m by friction: more than the panel's own weight (3.4 N m forward, COM 53 mm in front of the
# hinge), so it stays where it is put. kappa = drive (m_a + m_b) / dt^2 ~ 1500 N m/rad, slip = hold / kappa; with the
# robot's fingertips on the bezel part of a push goes into contact friction (the face slides on the pads as it turns).
HINGE_ENV = dict(jar=dict(DESK_SHIFT_jar_lid="0,0,-0.02", DESK_EULER_jar_lid="0,0,0") if os.environ.get("JAR_UPRIGHT", "0") == "1" else {},
                 monitor=dict(DESK_DRIVE_monitor="0.00606", DESK_SLIP_monitor="0.00267"),   # holds ~4.5 N m (dbg/hinge_hold.py: 5.0 -> 5.66 measured)
                 # the pad cannot be slid on the desk in libuipc: pressed-and-dragged (20 N) or pushed by its edge (9 N) it
                 # does not move although Coulomb friction lets it go at ~2 N (dbg/pad_press.py, at 1 and 3 mm thickness).
                 # So this demo starts with the pad (and the mouse on it) 6.4 cm further right, overhanging the desk edge
                 # by 6 cm; in the photo it is 4 mm inboard.
                 pad_pick=dict(DESK_SHIFT_mousepad="0.064,0,0", DESK_SHIFT_mouse="0.064,0,0"),
                 # Ziga, 2026-10-14: "the cloth is too stiff". Bending length ~1.2 cm instead of 2.4 (bending stiffness
                 # scales with its cube: 0.66 -> 0.08) and a stretchier knit (E t 200 N/m instead of 2000). This demo
                 # only; the virtual-hand cloth demos keep scene.FABRIC.
                 cloth_fold=dict(DESK_FABRIC_cloth="E=2e5,bending_stiffness=0.08"))


def monitor(d, log):
    """1. the closed left hand pushes the panel's lower bezel back on a circle about the hinge axis (impedance, at most
    30 N): the screen tips forward ~6 deg and the friction hinge holds it there after the hand lets go.
    2. the left hand takes the panel by its left side edge, low down, coming in 30 deg off the panel's width (straight
    along it is out of reach) with its fingers across the 6 cm thickness, and swings the grip point forward about the
    hinge (impedance, at most 60 N): the screen tilts back ~12 deg against its weight and the hinge friction, and stays
    there when the hand lets go."""
    from yam_cart import quat_from_mat
    h = d.E["_hinges"]["monitor"]; J = S.SC["joints"]["monitor"]; q0 = J["q0"]
    hax = np.asarray(J["axis"], float); hax /= np.linalg.norm(hax)
    piv = d.pose_any("monitor_panel")[:3, 3].copy()
    tilt_deg = lambda: float(np.degrees(h.angle + q0))
    arc = lambda p, phi: piv + Rot.from_rotvec(hax * phi).apply(np.asarray(p) - piv)

    def W(p):                                            # panel link frame (origin on the hinge), current pose -> world
        Tp = d.pose_any("monitor_panel"); return Tp[:3, :3] @ np.asarray(p, float) + Tp[:3, 3]

    def rec(dd):
        if dd.k % 5 == 0:
            F = R.hand_force("left")
            log.append(dict(t=round(dd.t, 4), tilt_deg=tilt_deg(), hand_N=float(np.linalg.norm(F)), F=np.round(F, 2).tolist(),
                            tcp=np.round(d.tcp("left"), 4).tolist()))
    d.on_step.append(rec)
    d.wait(0.3)
    d.note(f"monitor: hinge at {np.round(piv, 3)}, tilt {tilt_deg():.2f} deg; friction drive {h.drive_strength:g}, slip {h.slip:.2e} rad")
    # 1. push the lower bezel back: the top comes forward
    bx = 0.0; d.go({}, 1.2, base_x=bx)
    # fingers 4 cm apart, both tips on the bezel (closed, the two pads squeeze each other and their contact chatters)
    d.cart.set_gripper("left", float(os.environ.get("MONITOR_PUSH_W", "0.04")))
    q = None
    zb = None
    for zr in (-0.12, -0.14, -0.16, -0.10):                # well above the bottom edge (-0.232): fingertips fully on the face
        for xb in (-0.15, -0.05, -0.25):
            b0 = W((xb, -0.053 - 0.03 - 0.04, zr))         # 4 cm in front of the front face
            try:
                q = d.ik("left", b0, yaw=0.0, tilt=math.pi / 2, tol=5e-3); zb = zr; break
            except RuntimeError:
                q = None
        if q is not None:
            break
    if q is None:
        raise RuntimeError("monitor: the lower bezel is out of reach")
    d.go({"left": q}, 2.0)
    imp = d.impedance_on("left", k=(1500.0, 1500.0, 1500.0), kr=(5.0, 5.0, 5.0), f_max=60.0, tau_max=3.0)
    p0 = imp.pos.copy(); n = int(round(2.5 / d.dt)); fwd = math.radians(-6.5)
    contact = p0 + (W((xb, -0.053 - 0.03, zb)) - b0)      # where the fingertips meet the bezel
    for s_ in R.min_jerk(n):
        imp.pos = p0 + (contact - p0) * min(1.0, 3 * s_) + (arc(contact, fwd * max(0.0, (s_ - 1 / 3) * 1.5)) - contact); d.step(1)
    d.wait(0.5)
    pushed = tilt_deg()
    imp.pos = imp.pos + (p0 - contact) * 1.5; d.wait(0.8); d.impedance_off("left")
    d.wait(1.0)
    held1 = tilt_deg()
    d.note(f"monitor: pushed {100 * -zb:.0f} cm below the hinge to {pushed:.2f} deg; after letting go {held1:.2f} deg; "
           f"peak force {max(r['hand_N'] for r in log):.0f} N")
    d.home(1.2)
    if os.environ.get("MONITOR_PULL", "0") != "1":
        d.wait(1.0)
        return dict(tilt_start_deg=log[0]["tilt_deg"], tilt_pushed_deg=pushed, tilt_held1_deg=held1, tilt_end_deg=log[-1]["tilt_deg"],
                    push_height_below_hinge_m=-zb)
    # 2. take the left edge and pull its bottom forward: the top goes back
    d.go({}, 1.2, base_x=-0.15)
    Tp = d.pose_any("monitor_panel"); inward, fwdv = Tp[:3, 0], -Tp[:3, 1]
    grip = None
    for z_rel in (-0.19, -0.16, -0.215):
        for a_deg in (30, 25, 35):
            a = math.radians(a_deg); ap = math.cos(a) * inward - math.sin(a) * fwdv
            E = W((-0.375 + 0.035, -0.053, z_rel))
            for sgn in (-1.0, 1.0):
                cl = sgn * np.cross([0, 0, 1.0], ap); cl /= np.linalg.norm(cl)
                qq = quat_from_mat(np.column_stack([cl, np.cross(ap, cl), ap]))
                try:
                    qa = d.ik("left", E - ap * 0.09, quat=qq)
                    qs = d.plan_pose("left", E - ap * 0.09, qq, E, qq, n_via=8, seed=qa)
                except RuntimeError:
                    continue
                grip = (E, qq, ap, z_rel, a_deg, qa, qs); break
            if grip: break
        if grip: break
    if grip is None:
        raise RuntimeError("monitor: no reachable grasp on the panel's left edge")
    E, qq, ap, z_rel, a_deg, qa, qs = grip
    d.cart.set_gripper("left", 0.095); d.wait(0.3)
    d.go({"left": qa}, 2.0); d.follow("left", qs, 1.2)
    d.cart.grasp("left", force=25.0); d.grip_until("left", 1.0)
    d.note(f"monitor: left edge gripped {100 * -z_rel:.0f} cm below the hinge, approach {a_deg} deg, width {d.cart.gripper_width('left') * 1e3:.1f} mm")
    imp = d.impedance_on("left", k=(2500.0, 2500.0, 2500.0), kr=(15.0, 15.0, 15.0), f_max=60.0, tau_max=5.0)
    p0, R0 = imp.pos.copy(), imp.R.copy(); back = math.radians(12.0); n = int(round(3.0 / d.dt))
    for s_ in R.min_jerk(n):
        imp.pos = arc(p0, back * s_); imp.R = Rot.from_rotvec(hax * back * s_).as_matrix() @ R0; d.step(1)
    d.wait(0.5)
    pulled = tilt_deg()
    d.open_wait({"left": 0.095})
    pe, qe = d.cart.tcp_pose("left"); d.impedance_off("left")
    try:
        d.move_pose("left", pe, qe, pe - ap * 0.10, qe, 0.8, n_via=6)
    except RuntimeError:
        pass
    d.wait(1.5)
    held2 = tilt_deg()
    d.note(f"monitor: pulled back to {pulled:.2f} deg; after letting go {held2:.2f} deg; peak force {max(r['hand_N'] for r in log):.0f} N")
    d.home(1.5); d.wait(0.8)
    return dict(tilt_start_deg=log[0]["tilt_deg"], tilt_pushed_deg=pushed, tilt_held1_deg=held1, tilt_pulled_deg=pulled,
                tilt_held2_deg=held2, tilt_end_deg=log[-1]["tilt_deg"])


# ------------------------------------------------------------------------------------------------ bottle + stick
def bottle(d, log):
    """the right hand pinches the wooden stick near its top, draws it out of the bottle along its own axis, carries it to a
    clear patch of the mouse pad and lets it fall over; then the closed hand pushes the bottle over sideways (towards the
    tea tin, away from the desk edge). The cart parks where the right arm's links clear the jar at the desk's front
    right corner (cart x 0.27, straight in front of the bottle, puts the upper arm through it)."""
    d.wait(0.3)
    Ts, Tb = d.pose("stick"), d.pose("bottle")
    ax = Ts[:3, 2] * np.sign(Ts[2, 2]); top = Ts[:3, 3] + ax * 0.1
    grip = top - ax * 0.025; bpos = Tb[:3, 3]
    tilt = math.pi / 2; yaw = 0.0
    out = grip + ax * 0.20                               # drawn out along its own axis
    zc = S.H + 0.13
    pa = np.array([bpos[0] + 0.0375 + 0.016, bpos[1], zc]); pb_ = pa - np.array([0.15, 0.0, 0.0])
    pre = np.array([0.0, -0.08, 0.0])
    obst = [(dd_, r_) for dd_, r_ in ((d.pose("mouse")[:2, 3], 0.08), (d.pose("jar")[:2, 3], 0.08), (bpos[:2], 0.07),
                                      (d.pose("tin")[:2, 3], 0.07))]
    edge = 0.5 * np.asarray(json_layout()["table"]["size"][:2])

    def drop_spots(qd):
        """(TCP, flange quat, turn) to hold the stick upright over the pad with its lower end 1.8 cm up, the hand turned
        `turn` rad about the vertical (the stick's lean turns with it); the 20 cm it falls along its lean must land
        clear of the mouse, jar, bottle and tin, on the desk"""
        out_ = []
        for turn in (0.0, 0.5, -0.5, 0.9, -0.9, 1.3):
            Rz = Rot.from_rotvec([0, 0, turn]); lean = Rz.apply(ax); lh = lean[:2] / np.linalg.norm(lean[:2])
            x_, y_, z_, w_ = (Rz * Rot.from_quat([qd[1], qd[2], qd[3], qd[0]])).as_quat(); qT = np.array([w_, x_, y_, z_])
            for x in (0.56, 0.52, 0.48, 0.60, 0.44):
                for y in (-0.08, -0.04, 0.0, -0.12, 0.04):
                    lo = np.array([x, y])                # lower end of the stick, below the hand
                    seg = lo[None] + np.linspace(0, 0.21, 8)[:, None] * lh[None]
                    if any(np.linalg.norm(seg - c_, axis=1).min() < r_ for c_, r_ in obst) or (np.abs(seg) > edge - 0.02).any():
                        continue
                    tcp = np.array([x, y, S.H + 0.018]) + Rz.apply(ax) * 0.175
                    out_.append((tcp, qT, turn))
        return out_

    choice, errs = None, []
    for bx in (0.10, 0.05, 0.15, 0.0, 0.35, 0.40):
        try:
            j0 = d.ik("right", grip + pre, yaw=yaw, tilt=tilt, base_x=bx); j1 = d.ik("right", grip, yaw=yaw, tilt=tilt, base_x=bx, seed=j0)
            from yam_cart import down_quat
            qd = down_quat(yaw, tilt)
            d.P.chain("right", d.q_ref(bx), grip, qd, out, qd, n_via=10, seed=j1)
            lay = None
            for cand, q_lay, turn in drop_spots(qd):
                try:
                    jl = d.ik("right", cand + [0, 0, 0.10], quat=q_lay, tol=4e-3, rtol=0.06, base_x=bx)
                    d.ik("right", cand, quat=q_lay, tol=4e-3, rtol=0.06, base_x=bx, seed=jl)
                    lay = (cand, q_lay, turn); break
                except RuntimeError as e:
                    errs.append(str(e))
            if lay is None:
                raise RuntimeError("no reachable spot to drop the stick")
            yp = d.grasp_yaw("right", pa, math.pi / 2, tilt=tilt, base_x=bx)
            jp = d.ik("right", pa + [0, -0.10, 0], yaw=yp, tilt=tilt, base_x=bx)
            ja = d.ik("right", pa, yaw=yp, tilt=tilt, base_x=bx, seed=jp)
            qp = down_quat(yp, tilt)
            d.P.chain("right", d.q_ref(bx), pa, qp, pb_, qp, n_via=10, seed=ja)
        except RuntimeError as e:
            errs.append(f"bx {bx}: {e}"); continue
        choice = (bx, lay, yp); break
    if choice is None:
        raise RuntimeError("bottle: no cart position works: " + " || ".join(e[:120] for e in errs[-4:]))
    bx, lay, yp = choice

    def rec(dd):
        if dd.k % 5 == 0:
            T1, T2 = dd.pose("stick"), dd.pose("bottle")
            log.append(dict(t=round(dd.t, 4), stick_top_z_mm=1000 * float((T1[:3, 3] + T1[:3, 2] * 0.1 * np.sign(T1[2, 2]))[2] - S.H),
                            bottle_tilt_deg=float(np.degrees(np.arccos(np.clip(T2[2, 2], -1, 1)))), hand_N=float(np.linalg.norm(R.hand_force("right")))))
    d.on_step.append(rec)
    d.note(f"bottle: stick top {np.round(top, 3)}, axis {np.round(ax, 2)}, bottle {np.round(bpos, 3)}, base x {bx:.2f}")
    d.go({}, 1.4, base_x=bx)
    d.go({"right": d.ik("right", grip + pre, yaw=yaw, tilt=tilt)}, 2.0)
    d.cart.set_gripper("right", 0.03); d.wait(0.3)
    d.go_paths({"right": ((grip + pre, yaw, tilt), (grip, yaw, tilt))}, 1.0)
    d.cart.grasp("right", force=10.0); d.grip_until("right", 1.0)
    d.note(f"bottle: stick gripped, width {d.cart.gripper_width('right') * 1e3:.1f} mm")
    d.go_paths({"right": ((grip, yaw, tilt), (out, yaw, tilt))}, 2.0, n_via=10)
    # carry it to a clear patch of the mouse pad and hold it upright with its lower end 1.5 cm above the pad; let go and
    # it falls over the way it leans (back and left, clear of the bottle)
    lay, q_lay, turn = lay
    from yam_cart import down_quat
    qd = down_quat(yaw, tilt)
    d.note(f"bottle: dropping the stick at {np.round(lay, 3)}, hand turned {np.degrees(turn):.0f} deg")
    d.follow("right", d.plan_pose("right", out, qd, lay + [0, 0, 0.10], q_lay, n_via=10), 2.0)
    d.follow("right", d.plan_pose("right", lay + [0, 0, 0.10], q_lay, lay, q_lay, n_via=6), 1.2)
    d.open_wait({"right": 0.04})
    back = Rot.from_rotvec([0, 0, turn]).apply([0, -0.06, 0.04])
    d.follow("right", d.plan_pose("right", lay, q_lay, lay + back, q_lay, n_via=4), 0.8)
    d.wait(1.0)
    d.note(f"bottle: stick let go and fell; its top is {log[-1]['stick_top_z_mm']:.0f} mm above the desk")
    d.cart.set_gripper("right", 0.0)
    # the closed hand points forward (+y) with its fingers stacked vertically, so it is only a finger wide in x: it comes
    # in from the front just right of the bottle and sweeps towards -x high on the bottle's body, tipping it over
    d.go({"right": d.ik("right", pa + [0, -0.10, 0], yaw=yp, tilt=tilt)}, 1.6)
    d.go_paths({"right": ((pa + [0, -0.10, 0], yp, tilt), (pa, yp, tilt))}, 1.0)
    d.note("bottle: pushing it over")
    d.go_paths({"right": ((pa, yp, tilt), (pb_, yp, tilt))}, 1.6, n_via=10)
    d.wait(1.0)
    try:
        d.go_paths({"right": ((pb_, yp, tilt), (pb_ + [0, -0.10, 0.05], yp, tilt))}, 1.0)
    except RuntimeError:
        d.go_paths({"right": ((pb_, yp, tilt), (pb_ + [0, -0.06, 0.08], yp, tilt))}, 1.0)
    d.home(1.5); d.wait(0.8)
    return dict(bottle_tilt_end_deg=log[-1]["bottle_tilt_deg"], stick_top_end_mm=log[-1]["stick_top_z_mm"])


# ------------------------------------------------------------------------------------------------ cloth
def _hang_corners(V):
    hang = V[V[:, 2] < S.H - 0.05]
    lo = hang[hang[:, 2] < hang[:, 2].min() + 0.02]
    return lo[np.argmin(lo[:, 0])].copy(), lo[np.argmax(lo[:, 0])].copy()


def plan_side_pinch(d, side, point, inward, depth=0.025, base_x=None):
    """approach for pinching the side edge of a sheet hanging over the desk's front edge at `point` (below the
    tabletop): the hand turns sideways in front of the desk (above its height), drops below the slab, slides in under
    the edge, and comes in horizontally along `inward` (+-x, turned up to 40 deg towards +y if needed), fingers closing
    horizontally across the sheet. A joint-space move straight to the pre-pinch pose cuts through the slab.
    -> (joints in front of the desk, [legs of joint waypoints], flange quat, TCP point) or None if out of reach."""
    from yam_cart import quat_from_mat
    y_out = -0.5 * json_layout()["table"]["size"][1] - 0.09
    for ang in np.radians([0, 20, 40, -20]):
        ap_ = np.array([inward * math.cos(ang), math.sin(ang), 0.0])
        for sgn in (1.0, -1.0):
            cl = sgn * np.cross([0, 0, 1.0], ap_); qq = quat_from_mat(np.column_stack([cl, np.cross(ap_, cl), ap_]))
            p_in = point + ap_ * depth; pre = p_in - ap_ * 0.07
            hi, lo = np.array([pre[0], y_out, S.H + 0.06]), np.array([pre[0], y_out, pre[2]])
            try:
                q_hi = d.ik(side, hi, quat=qq, base_x=base_x, tol=4e-3, rtol=0.05, samples=60)
                kw = dict(tol=5e-3, rtol=0.06, base_x=base_x)
                down = d.plan_pose(side, hi, qq, lo, qq, n_via=6, seed=q_hi, **kw)
                under = d.plan_pose(side, lo, qq, pre, qq, n_via=6, seed=down[-1], **kw)
                into = d.plan_pose(side, pre, qq, p_in, qq, n_via=8, seed=under[-1], **kw)
            except RuntimeError:
                continue
            return q_hi, [down, under, into], qq, p_in
    return None


def pinch_side(d, side, point, inward, force=10.0, opening=0.06, depth=0.025, plan=None):
    """pinch the side edge of a hanging sheet at `point` (see plan_side_pinch). Returns (quat, TCP point)."""
    plan = plan or plan_side_pinch(d, side, point, inward, depth)
    if plan is None:
        raise RuntimeError(f"{side}: no reachable side pinch at {np.round(point, 3)}")
    q_hi, legs, qq, p_in = plan
    d.cart.set_gripper(side, opening)
    d.go({side: q_hi}, 1.8)
    for leg, dur in zip(legs, (0.9, 0.7, 0.75)):
        d.follow(side, leg, dur)
    d.cart.grasp(side, force=force); d.grip_until(side, 1.0)
    return qq, p_in


def cloth_fold(d, log):
    """the cloth hangs over the desk's front edge. Its hanging part is narrower than the two arms can straddle at once
    (each side pinch needs its own shoulder ~13 cm outboard of the hand, dbg/cloth_reach.py), so it is folded up one
    corner at a time: a hand reaches under the edge from the side, pinches the hanging side edge 6 cm below the
    tabletop, draws it back out in front of the desk, swings it up and lays it on the cloth on the desk, lets go; the
    cart moves over and the other hand does the other side."""
    from genesis.utils.misc import tensor_to_array
    from yam_cart import down_quat
    V = lambda: tensor_to_array(d.E["cloth"].get_state().pos)[0]
    edge = -0.5 * json_layout()["table"]["size"][1]

    def rec(dd):
        if dd.k % 5 == 0:
            v = V(); log.append(dict(t=round(dd.t, 4), on_desk_frac=float((v[:, 2] > S.H - 0.005).mean()), cloth_minz_mm=1000 * float(v[:, 2].min() - S.H)))
    d.on_step.append(rec)
    d.wait(1.0)                                          # let the drape settle
    zp = S.H - 0.06
    widths = {}
    for side, inward in (("left", +1.0), ("right", -1.0)):
        v = V(); band = v[np.abs(v[:, 2] - zp) < 0.01]
        if len(band) < 3:
            d.note(f"cloth: nothing left hanging at the {side} side"); continue
        e = band[np.argmin(band[:, 0])].copy() if inward > 0 else band[np.argmax(band[:, 0])].copy()
        e[1] = band[np.abs(band[:, 0] - e[0]) < 0.03, 1].mean(); e[2] = zp
        plan, bx = None, None
        for off in (0.20, 0.25, 0.15, 0.30):                # the cart (between the shoulders) sits on the approach side
            b = float(e[0] + inward * off)
            plan = plan_side_pinch(d, side, e, inward, base_x=b)
            if plan:
                bx = b; break
        if plan is None:
            raise RuntimeError(f"cloth: no cart position reaches the {side} side edge at {np.round(e, 3)}")
        d.go({}, 1.4, base_x=bx)
        qq, p_in = pinch_side(d, side, e, inward, plan=plan)
        widths[side] = d.cart.gripper_width(side)
        d.note(f"cloth: {side} side edge at x {e[0]:.3f} pinched (base x {bx:.2f}), width {widths[side] * 1e3:.1f} mm")
        if widths[side] < 1.2e-3:
            raise RuntimeError(f"cloth: the {side} hand closed on nothing")
        _, legs, _, _ = plan
        for leg, dur in zip(reversed(legs), (0.75, 0.7, 0.9)):   # back out from under the slab, holding the cloth
            d.follow(side, leg[::-1], dur)
        p_hi, _ = d.cart.tcp_pose(side)
        lays = [np.array([e[0] + inward * dx, edge + dy, S.H + 0.035]) for dy in (0.10, 0.07, 0.13) for dx in (0.04, 0.08, 0.0)]
        ups = [np.array([p_hi[0] + inward * dx, y, S.H + h]) for h in (0.14, 0.18, 0.10) for dx in (0.12, 0.08, 0.16) for y in (-0.40, -0.36)]
        alts = (down_quat(0.0, 0.5), down_quat(math.pi / 2, 0.5), down_quat(math.pi / 2, 0.0))
        # the swing up and over is a joint-space move between collision-free poses (from the side pinch, a straight TCP
        # line through the wrist turn passes near singular poses)
        leg_up, top, q_top = d.plan_to(side, ups, (qq, *alts))
        d.follow(side, leg_up, 1.4)
        leg_lay, lay, _ = d.plan_to(side, lays, (q_top, *alts))
        d.follow(side, leg_lay, 1.6)
        # let go wide open and back the fingers straight out along the approach before anything else moves (a half-open
        # hand rising off the pile snagged it and dragged it to the edge)
        d.wait(0.3); d.open_wait({side: 0.096})
        pe, qe = d.cart.tcp_pose(side)
        from yam_cart import mat_from_quat
        ap = mat_from_quat(qe)[:, 2]
        for back_ in (0.08, 0.06, 0.04):
            try:
                d.move_pose(side, pe, qe, pe - ap * back_, qe, 0.8, n_via=6); break
            except RuntimeError:
                continue
        else:
            d.note(f"cloth: the {side} hand could not back straight out of the pile")
        d.note(f"cloth: {side} corner laid on the desk at {np.round(lay, 3)}; {100 * log[-1]['on_desk_frac']:.0f}% of the cloth on the desk")
        d.home(1.2)
    d.wait(1.0)
    return dict(on_desk_frac=log[-1]["on_desk_frac"], cloth_minz_mm=log[-1]["cloth_minz_mm"],
                widths_mm={k: round(v * 1e3, 1) for k, v in widths.items()})


def cloth_drop(d, log):
    """one hand pinches the hanging cloth's side edge near its bottom corner, lifts the cloth 30 cm so it hangs from
    that point, carries it over the desk and lets go: it falls and heaps."""
    from genesis.utils.misc import tensor_to_array
    V = lambda: tensor_to_array(d.E["cloth"].get_state().pos)[0]

    def rec(dd):
        if dd.k % 5 == 0:
            v = V(); log.append(dict(t=round(dd.t, 4), on_desk_frac=float((v[:, 2] > S.H - 0.005).mean()), cloth_maxz_mm=1000 * float(v[:, 2].max() - S.H)))
    d.on_step.append(rec)
    d.wait(1.0)
    side, zp = "left", S.H - 0.06
    v = V(); band = v[np.abs(v[:, 2] - zp) < 0.01]; eL = band[np.argmin(band[:, 0])].copy()
    eL[1] = band[band[:, 0] < eL[0] + 0.03, 1].mean(); eL[2] = zp   # the hanging layers' mid-plane near the edge
    plan, bx = None, None
    for off in (0.20, 0.25, 0.15, 0.30):                    # the cart sits right of the left hand (dbg/cloth_reach.py)
        plan = plan_side_pinch(d, side, eL, +1.0, base_x=float(eL[0] + off))
        if plan:
            bx = float(eL[0] + off); break
    if plan is None:
        raise RuntimeError(f"cloth: no cart position reaches the left side edge at {np.round(eL, 3)}")
    d.go({}, 1.4, base_x=bx)
    q1, p1 = pinch_side(d, side, eL, +1.0, plan=plan)
    d.note(f"cloth: left side edge at x {eL[0]:.3f} pinched (base x {bx:.2f}), width {d.cart.gripper_width(side) * 1e3:.1f} mm")
    if d.cart.gripper_width(side) < 1.2e-3:
        raise RuntimeError("cloth: the hand closed on nothing")
    from yam_cart import down_quat
    _, legs, _, _ = plan
    for leg, dur in zip(reversed(legs), (0.75, 0.7, 0.9)):   # back out from under the slab, holding the cloth
        d.follow(side, leg[::-1], dur)
    p_hi, _ = d.cart.tcp_pose(side)
    # the hand lifts inboard, towards the cart (dbg/cloth_lift_reach.py: straight up from here is out of reach)
    alts = (down_quat(0.0, 0.5), down_quat(math.pi / 2, 0.5), down_quat(math.pi / 2, 0.0))
    ups = [np.array([p_hi[0] + dx, y, S.H + h]) for h in (0.24, 0.20, 0.16) for dx in (0.16, 0.12, 0.08) for y in (-0.40, -0.36)]
    leg1, up, qa = d.plan_first(side, p_hi, q1, ups, alts)
    overs = [up + [0.0, dy, -0.02] for dy in (0.12, 0.09, 0.06)]
    leg2, over, _ = d.plan_first(side, up, qa, overs, alts, n_via=8)
    d.note(f"cloth: lifting it to {1000 * (up[2] - S.H):.0f} mm above the desk, then over the desk to y {over[1]:.2f}")
    d.follow(side, leg1, 2.2); d.follow(side, leg2, 1.4)
    d.wait(0.5)
    d.note(f"cloth: lifted, top {log[-1]['cloth_maxz_mm']:.0f} mm above the desk; letting go")
    d.open_wait({side: 0.05})
    d.wait(2.0)
    d.home(1.5); d.wait(0.5)
    return dict(on_desk_frac=log[-1]["on_desk_frac"])


# ------------------------------------------------------------------------------------------------ sheets: drag to the edge
EDGE_Y = None


def _drag(d, side, start, dy, press=10.0, dur=2.5, dx=0.0, yaw=0.0, tilt=0.0):
    """press the closed gripper down on a sheet at `start` (impedance: stiff in x, y; ~`press` N down) and drag it
    `dy` metres along y (negative = towards the robot) and `dx` along x."""
    d.cart.set_gripper(side, 0.0)
    q = d.ik(side, start + [0, 0, 0.05], yaw=yaw, tilt=tilt)
    d.go({side: q}, 1.4)
    d.go_paths({side: ((start + [0, 0, 0.05], yaw, tilt), (start + [0, 0, 0.01], yaw, tilt))}, 0.8)
    kz = 800.0
    imp = d.impedance_on(side, k=(2500.0, 2500.0, kz), kr=(3.0, 3.0, 3.0), f_max=press + 25.0, tau_max=2.0)
    imp.pos = start - [0, 0, press / kz]
    d.wait(0.6)
    p0 = imp.pos.copy(); n = int(round(dur / d.dt))
    for s_ in R.min_jerk(n):
        imp.pos = p0 + [dx * s_, dy * s_, 0]
        d.step(1)
    imp.pos = imp.pos + [0, 0, 0.06]; d.wait(0.6)
    d.impedance_off(side)


def paper_fold(d, log):
    """The marker is taken off the desk for this demo. Both closed hands press on the top sheet near its front corners
    and drag it toward the robot until its whole front edge overhangs the desk. Each hand then pinches its front corner
    from the front (fingers closing vertically, the lower one under the overhang, holding the last 12 mm of the edge),
    and together they swing the edge over in a half circle about the sheet's middle line, folding it in half. The left
    hand keeps the edge down while the right lets go and presses the crease with its closed fingers; then the left
    lets go too. Holding is stiff joint control throughout (a hand held in impedance jitters); only the presses use
    impedance."""
    from genesis.utils.misc import tensor_to_array
    from yam_cart import down_quat
    V = lambda: tensor_to_array(d.E["paper_top"].get_state().pos)[0]
    edge = -0.5 * json_layout()["table"]["size"][1]
    zz = np.array([0.0, 0.0, 1.0])

    def frame(v):
        """front-left / front-right corners, edge direction e (left -> right), m (into the sheet), depth along m"""
        FL = v[np.argmin(v[:, 0] + v[:, 1])].copy(); FR = v[np.argmax(v[:, 0] - v[:, 1])].copy()
        e = (FR - FL) * [1, 1, 0]; e /= np.linalg.norm(e); m = np.array([-e[1], e[0], 0.0])
        if m[1] < 0:
            m = -m
        depth = float(((v - FL) @ m).max())
        return FL, FR, e, m, depth

    def rec(dd):
        if dd.k % 5 == 0:
            v = V(); log.append(dict(t=round(dd.t, 4), overhang_mm=1000 * float(edge - v[:, 1].min()), sheet_maxz_mm=1000 * float(v[:, 2].max() - S.H)))
    d.on_step.append(rec)
    d.wait(0.5)
    v = V(); FL, FR, e, m, depth = frame(v)
    drag = (max(FL[1], FR[1]) - edge) + 0.045               # both front corners end >= 4.5 cm over the edge
    sL = FL + e * 0.05 + m * 0.05; sR = FR - e * 0.05 + m * 0.05
    sL[2] = sR[2] = S.H + 0.002
    up, dy = np.array([0, 0, 0.05]), np.array([0, -drag, 0])
    choice, errs = None, []
    for bx in 0.5 * (FL[0] + FR[0]) + np.array([0.0, 0.04, -0.04, 0.08]):
        for td in (0.0, 0.3, 0.6):
            try:
                qa = {s_: d.ik(s_, p_ + up, yaw=0.0, tilt=td, base_x=bx) for s_, p_ in (("left", sL), ("right", sR))}
                qe = {s_: d.ik(s_, p_ + dy, yaw=0.0, tilt=td, base_x=bx, seed=qa[s_]) for s_, p_ in (("left", sL), ("right", sR))}
                d.config_free(qa, base_x=bx); d.config_free(qe, base_x=bx)
            except RuntimeError as er:
                errs.append(str(er)); continue
            choice = (bx, td); break
        if choice: break
    if choice is None:
        raise RuntimeError("paper: no cart position for the two-hand drag: " + " || ".join(x[:100] for x in errs[-3:]))
    bx, td = choice
    d.note(f"paper: sheet {1000 * np.linalg.norm(FR - FL):.0f} x {1000 * depth:.0f} mm, front edge at y {FL[1]:.3f} / {FR[1]:.3f} "
           f"(desk edge {edge:.3f}); both hands drag it {1000 * drag:.0f} mm, base x {bx:.2f}, hands pitched {np.degrees(td):.0f} deg")
    d.go({}, 1.2, base_x=bx)
    d.cart.set_gripper("left", 0.0); d.cart.set_gripper("right", 0.0)
    q_ = down_quat(0.0, td)
    for stroke in range(3):                               # the sheet slips under the hands: drag again until both front
        if stroke:                                        # corners hang >= 4 cm over the edge
            v = V(); FL, FR, e, m, depth = frame(v)
            short = 0.045 - (edge - max(FL[1], FR[1]))
            if short < 0.005:
                break
            dy = np.array([0, -(short + 0.01), 0])
            sL = FL + e * 0.05 + m * 0.05; sR = FR - e * 0.05 + m * 0.05; sL[2] = sR[2] = S.H + 0.002
            d.note(f"paper: corners still {1000 * (edge - FL[1]):.0f} / {1000 * (edge - FR[1]):.0f} mm over; dragging another "
                   f"{1000 * (short + 0.01):.0f} mm")
        try:
            d.go({s_: d.ik(s_, p_ + up, yaw=0.0, tilt=td) for s_, p_ in (("left", sL), ("right", sR))}, 1.6)
            d.follow_many({s_: d.plan_pose(s_, p_ + up, q_, p_ + [0, 0, 0.008], q_, n_via=4) for s_, p_ in (("left", sL), ("right", sR))}, 0.8)
        except RuntimeError as er:
            d.note(f"paper: no further drag stroke ({str(er)[:80]})"); break
        kz, press = 800.0, 10.0
        imp = {s_: d.impedance_on(s_, k=(2500.0, 2500.0, kz), kr=(5.0, 5.0, 5.0), f_max=press + 25.0, tau_max=2.0) for s_ in ("left", "right")}
        imp["left"].pos = sL - [0, 0, press / kz]; imp["right"].pos = sR - [0, 0, press / kz]
        d.wait(0.6)
        p0 = {s_: im.pos.copy() for s_, im in imp.items()}; n = int(round(2.5 / d.dt))
        for s_k in R.min_jerk(n):
            for s_, im in imp.items():
                im.pos = p0[s_] + dy * s_k
            d.step(1)
        for s_, im in imp.items():
            im.pos = im.pos + [0, 0, 0.05]
        d.wait(0.6)
        d.impedance_off("left"); d.impedance_off("right")
        if getattr(d, "dry", False):
            break
    v = V(); FL, FR, e, m, depth = frame(v)
    d.note(f"paper: front corners now {1000 * (edge - FL[1]):.0f} / {1000 * (edge - FR[1]):.0f} mm over the edge")
    # 2. each hand pinches its front corner from the front, 12 mm into the sheet, 2 cm in from the side
    PL = FL + e * 0.02 + m * 0.012; PR = FR - e * 0.02 + m * 0.012
    PL[2] = PR[2] = float(np.median(v[:, 2]))
    r = 0.5 * depth - 0.012                                   # pinch to the middle line (the crease)
    CL, CR = PL + m * r, PR + m * r
    th = np.linspace(0.0, math.pi, 9)
    choice, errs = None, []
    for pitch in (0.0, 0.3, 0.5):                             # pinch approach along m, level or pitched down
        a0 = math.cos(pitch) * m - math.sin(pitch) * zz
        for sc in (1.0, -1.0):
            qp = hand_quat(a0, sc * (math.sin(pitch) * m + math.cos(pitch) * zz))
            try:
                pre = {s_: d.ik(s_, P_ - a0 * 0.06, quat=qp) for s_, P_ in (("left", PL), ("right", PR))}
                at = {s_: d.ik(s_, P_, quat=qp, seed=pre[s_]) for s_, P_ in (("left", PL), ("right", PR))}
                d.config_free(pre); d.config_free(at)
            except RuntimeError as er:
                errs.append(str(er)); continue
            # the fold: the pinch points swing over the crease on half circles; the hands pitch further down on the way
            # (by up to 70 deg: less of the edge is bent back in the fingers)
            for p_end in (1.2, 0.9, 0.6, 0.3, 0.0):
                a1 = math.cos(pitch + p_end) * m - math.sin(pitch + p_end) * zz
                q1 = hand_quat(a1, sc * (math.sin(pitch + p_end) * m + math.cos(pitch + p_end) * zz))
                arcs = {s_: [C_ - r * math.cos(t_) * m + np.array([0, 0, P_[2] + 0.9 * r * math.sin(t_) - C_[2]
                                                                  + (0.003 if t_ == math.pi else 0.0)]) for t_ in th]
                        for s_, P_, C_ in (("left", PL, CL), ("right", PR, CR))}
                try:
                    from scipy.spatial.transform import Slerp
                    sl = Slerp([0, 1], Rot.from_quat([[*qp[1:], qp[0]], [*q1[1:], q1[0]]]))
                    qs = [np.r_[x[3], x[:3]] for x in sl(th / math.pi).as_quat()]
                    legs = {}
                    for s_ in ("left", "right"):
                        other = "right" if s_ == "left" else "left"
                        seed, L_ = at[s_], []
                        for k_ in range(len(th) - 1):
                            leg = d.plan_pose(s_, arcs[s_][k_], qs[k_], arcs[s_][k_ + 1], qs[k_ + 1], n_via=3, seed=seed,
                                              tol=5e-3, rtol=0.06, assume={other: at[other]}); seed = leg[-1]
                            L_ += leg if not L_ else leg[1:]
                        legs[s_] = L_
                except RuntimeError as er:
                    errs.append(f"fold pitch {p_end}: {er}"); continue
                choice = (pitch, sc, a0, qp, pre, at, legs, p_end, q1); break
            if choice: break
        if choice: break
    if choice is None:
        raise RuntimeError("paper: no reachable two-hand pinch and fold: " + " || ".join(x[:110] for x in errs[-3:]))
    pitch, sc, a0, qp, pre, at, legs, p_end, q1 = choice
    d.note(f"paper: pinching both front corners (approach pitched {np.degrees(pitch):.0f} deg down), fold radius {1000 * r:.0f} mm, "
           f"hands pitch {np.degrees(p_end):.0f} deg further on the way over")
    d.cart.set_gripper("left", 0.02); d.cart.set_gripper("right", 0.02)
    d.go(pre, 1.8)
    d.follow_many({s_: d.plan_pose(s_, P_ - a0 * 0.06, qp, P_, qp, n_via=5, seed=pre[s_]) for s_, P_ in (("left", PL), ("right", PR))}, 1.0)
    d.cart.grasp("left", force=8.0); d.cart.grasp("right", force=8.0)
    d.grip_until("left", 1.0); d.grip_until("right", 1.0)
    d.note(f"paper: corners pinched, widths L {d.cart.gripper_width('left') * 1e3:.1f} R {d.cart.gripper_width('right') * 1e3:.1f} mm")
    v0 = V().copy(); strip = ((v0 - PL) @ m) < r + 0.012          # the front half: it should end mirrored over the crease
    d.follow_many(legs, 3.5)
    d.wait(0.3)
    # 3. the right hand lets go and backs out; the left keeps the edge down
    d.open_wait({"right": 0.03})
    pe, qe = d.cart.tcp_pose("right")
    from yam_cart import mat_from_quat
    ap = mat_from_quat(qe)[:, 2]
    for back_ in (0.05, 0.035, 0.02):
        try:
            d.move_pose("right", pe, qe, pe - ap * back_ + [0, 0, 0.03], qe, 0.8, n_via=5); break
        except RuntimeError:
            continue
    # 4. the closed right hand presses the crease: four presses along it, from the right end to the middle
    d.cart.set_gripper("right", 0.0)
    crease = lambda u: 0.5 * (CL + CR) + e * u - m * 0.004
    for u in (0.10, 0.05, 0.0, -0.05):
        p = crease(u); p[2] = S.H + 0.003
        q = None
        for th_t in (0.0, 0.3, 0.6):
            try:
                q = d.ik("right", p + [0, 0, 0.04], yaw=0.0, tilt=th_t)
                qt = down_quat(0.0, th_t)
                d.check_legs({"right": d.plan_pose("right", p + [0, 0, 0.04], qt, p - [0, 0, 0.012], qt, n_via=6, seed=q)})
                break
            except RuntimeError:
                q = None
        if q is None:
            d.note(f"paper: crease press at {np.round(p, 3)} skipped (out of reach or would hit the left hand)")
            continue
        try:
            d.go({"right": q}, 0.7)
        except RuntimeError:
            continue
        im = d.impedance_on("right", k=(2500.0, 2500.0, 800.0), kr=(5.0, 5.0, 5.0), f_max=20.0, tau_max=2.0)
        im.pos = p - [0, 0, 10.0 / 800.0]; d.wait(0.5); im.pos = p + [0, 0, 0.04]; d.wait(0.4); d.impedance_off("right")
    # 5. the left lets go and backs out; both home
    d.open_wait({"left": 0.03})
    pe, qe = d.cart.tcp_pose("left"); ap = mat_from_quat(qe)[:, 2]
    for back_ in (0.05, 0.035, 0.02):
        try:
            d.move_pose("left", pe, qe, pe - ap * back_ + [0, 0, 0.03], qe, 0.8, n_via=5); break
        except RuntimeError:
            continue
    d.home(1.5); d.wait(1.0)
    v = V()
    mir = v0[strip] - 2 * (((v0[strip] - CL) @ m)[:, None] * m)
    folded = float((np.linalg.norm(v[strip][:, :2] - mir[:, :2], axis=1) < 0.015).mean())
    on_top = float((v[strip][:, 2] > np.median(v[~strip][:, 2]) + 0.0002).mean())
    d.note(f"paper: {100 * folded:.0f}% of the front half within 15 mm of its folded position, {100 * on_top:.0f}% lying on the "
           f"back half; sheet top {1000 * (v[:, 2].max() - S.H):.0f} mm")
    return dict(folded_frac=folded, on_top_frac=on_top, sheet_maxz_mm=1000 * float(v[:, 2].max() - S.H),
                fold_radius_mm=1000 * float(r), drag_mm=1000 * float(drag))


def paper_dogear(d, log):
    """the left hand presses on the top sheet near its front-left corner (clear of the marker lying on the sheet's right
    half) and drags it ~6 cm towards the robot until its front strip overhangs the desk; it pinches the overhang near
    the front-left corner (fingers closing vertically) and dog-ears the corner: the pinch point swings on a half circle
    about a diagonal crease 7 cm in from the corner while the right hand presses the sheet down inside the crease (a
    straight fold of the whole front strip from one pinch lifts a tent of the stiff sheet with it); then it backs out
    of the fold, presses the crease down and lets go. The crease-plastic paper keeps the fold."""
    from genesis.utils.misc import tensor_to_array
    from yam_cart import down_quat
    V = lambda: tensor_to_array(d.E["paper_top"].get_state().pos)[0]
    edge = -0.5 * json_layout()["table"]["size"][1]
    F, Hh = "left", "right"                              # fold hand, hold hand
    tilt = math.pi / 2

    def rec(dd):
        if dd.k % 5 == 0:
            v = V(); log.append(dict(t=round(dd.t, 4), overhang_mm=1000 * float(edge - v[:, 1].min()), sheet_maxz_mm=1000 * float(v[:, 2].max() - S.H)))
    d.on_step.append(rec)
    d.wait(0.5)
    v = V(); x0 = float(np.percentile(v[:, 0], 15)); yf = float(v[:, 1].min())
    d.paper_strip = (v[:, 1] < yf + 0.02) & (np.abs(v[:, 0] - x0) < 0.03)    # the front 2 cm of the sheet around the pinch
    drag = -(yf - edge) - 0.045
    yc = edge + 0.03; a = np.array([x0, edge - 0.006, S.H + 0.0005]); L = yc - a[1]
    # one cart position for the drag, the pinch and the hold (all checked before anything moves). The cart stands back
    # and its deck is lowered (CART): the pinch is at desk height just in front of the edge, where a hand from a deck
    # level with the desk folds its wrist into the deck. The drag may pitch the closed hand away from the robot.
    choice, errs = None, []
    for b in x0 + np.array([0.05, 0.10, 0.0, 0.15, 0.20, -0.05, 0.25]):
        try:
            for td in (0.0, 0.3, 0.6, 0.9):
                try:
                    d.ik(F, np.array([x0, yf + 0.04, S.H + 0.05]), yaw=0.0, tilt=td, base_x=b)
                    d.ik(F, np.array([x0, yf + 0.04 + drag, S.H]), yaw=0.0, tilt=td, base_x=b)
                    break
                except RuntimeError as e:
                    errs.append(str(e))
            else:
                raise RuntimeError("drag out of reach")
            for tp_ in (tilt, tilt - 0.3, tilt - 0.5):     # pinch approach: towards +y, level or pitched down
                try:
                    yawp = d.grasp_yaw(F, a, math.pi / 2, tilt=tp_, base_x=b)
                    apx = np.array([0.0, math.sin(tp_), -math.cos(tp_)])
                    d.ik(F, a - 0.06 * apx, yaw=yawp, tilt=tp_, base_x=b)
                    d.ik(F, np.array([x0, yc + L, S.H + 0.002]), yaw=yawp, tilt=tp_, base_x=b, tol=5e-3, rtol=0.06)
                    break
                except RuntimeError as e:
                    errs.append(str(e))
            else:
                raise RuntimeError("pinch out of reach")
        except RuntimeError as e:
            errs.append(f"b {b:.2f}: {e}"); continue
        hold = None
        for dxh in (0.08, 0.06, 0.10, 0.12):
            cand = np.array([x0 + dxh, yc + L + 0.03, S.H + 0.002])
            for th in (0.0, 0.3, 0.6):
                try:
                    d.ik(Hh, cand + [0, 0, 0.05], yaw=0.0, tilt=th, base_x=b); d.ik(Hh, cand, yaw=0.0, tilt=th, base_x=b); hold = cand; break
                except RuntimeError as e:
                    errs.append(str(e))
            if hold is not None:
                break
        if hold is not None:
            choice = (b, yawp, tp_, td); break
    if choice is None:
        raise RuntimeError("paper: no cart position reaches the drag, the pinch and the hold: " + " || ".join(e[:100] for e in errs[-4:]))
    bx, yawp, tilt, td = choice
    apx = np.array([0.0, math.sin(tilt), -math.cos(tilt)])
    d.go({}, 1.2, base_x=bx)
    d.note(f"paper: front edge at y {yf:.3f} (desk edge {edge:.3f}); dragging at x {x0:.3f}, base x {bx:.2f}, hand pitched "
           f"{np.degrees(td):.0f} deg; pinch approach pitched {np.degrees(math.pi / 2 - tilt):.0f} deg down")
    _drag(d, F, np.array([x0, yf + 0.04, S.H + 0.002]), drag, tilt=td)
    v = V(); yf = float(v[:, 1].min())
    d.note(f"paper: overhang {1000 * (edge - yf):.0f} mm")
    # dog-ear: pinch the overhang near the front-left corner (approach towards +y, fingers closing vertically, the lower
    # one under the overhang beyond the desk edge) and swing the pinch point on a half circle about a diagonal crease
    # 7 cm in from the corner, so only that corner triangle turns over
    v = V(); front = v[:, 1] < yf + 0.01; corner = v[front][np.argmin(v[front][:, 0])].copy(); corner[2] = S.H
    n = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0); C = corner + n * 0.07     # crease: through C, normal n
    P = np.array([corner[0] + 0.015, edge - 0.006, S.H + 0.0005])            # pinch point, in the overhang
    r = -float((P - C) @ n); Pc = P + r * n; Pf = Pc + r * n                  # foot on the crease, landing point
    d.paper_strip = (v[:, 1] - corner[1]) + (v[:, 0] - corner[0]) < 0.05 * math.sqrt(2.0)   # the corner triangle
    d.paper_v0 = v.copy()
    q = d.ik(F, P - 0.06 * apx, yaw=yawp, tilt=tilt)
    d.go({F: q}, 1.4); d.cart.set_gripper(F, 0.02); d.wait(0.4)
    d.go_paths({F: ((P - 0.06 * apx, yawp, tilt), (P, yawp, tilt))}, 1.0)
    d.cart.grasp(F, force=8.0); d.grip_until(F, 1.0)
    d.note(f"paper: corner overhang pinched at {np.round(P, 3)}, width {d.cart.gripper_width(F) * 1e3:.1f} mm; crease 7 cm from the corner, "
           f"fold radius {1000 * r:.0f} mm")
    # the other hand presses the sheet down just inside the crease. Its spot is chosen together with the fold arc: the
    # whole arc is planned against the hold hand already pressing there (the folded corner lands next to it)
    d.cart.set_gripper(Hh, 0.0)
    th = np.linspace(0, math.pi, 9); _, qf = d.cart.tcp_pose(F); tdir = np.array([-1.0, 1.0, 0.0]) / math.sqrt(2.0)
    choice, errs = None, []
    # close to the crease first (it anchors the sheet; 6 cm out, the corner slid instead of folding). Pitched towards
    # the robot (negative tilt) the hold hand's palm sits on the far side, away from the folding hand
    for k, s_ in ((0.02, 0.0), (0.03, 0.0), (0.02, 0.03), (0.02, -0.03), (0.04, 0.0), (0.05, 0.0)):
        cand = C + n * (r + k) + tdir * s_; cand[2] = S.H + 0.002
        for th_h in (-0.5, -0.3, 0.0, -0.7, 0.3, 0.6):
            for yh in (0.0, math.pi / 2):
                try:
                    jh0 = d.ik(Hh, cand + [0, 0, 0.05], yaw=yh, tilt=th_h); jh = d.ik(Hh, cand, yaw=yh, tilt=th_h, seed=jh0)
                except RuntimeError as e:
                    errs.append(str(e)); continue
                for lift in (0.7, 0.85, 1.0):            # a flat arc lifts less of the sheet; the whole arc is solved first
                    path = [Pc - r * math.cos(t_) * n + np.array([0, 0, 0.0015 + lift * r * math.sin(t_)]) for t_ in th]
                    try:
                        legs, seed = [], None
                        for kk in range(len(path) - 1):
                            legs.append(d.plan_pose(F, path[kk], qf, path[kk + 1], qf, n_via=3, seed=seed, tol=5e-3, rtol=0.06,
                                                    assume={Hh: jh}, clear=0.008)); seed = legs[-1][-1]
                    except RuntimeError as e:
                        errs.append(f"hold k {k} s {s_}, arc lift {lift}: {e}"); continue
                    choice = (cand, th_h, yh, legs); break
                if choice: break
            if choice: break
        if choice: break
    if choice is None:
        raise RuntimeError("paper: no hold spot leaves the fold arc clear: " + " || ".join(x[:140] for x in errs[-3:]))
    hold, th_h, yh, legs = choice
    d.go({Hh: d.ik(Hh, hold + [0, 0, 0.05], yaw=yh, tilt=th_h)}, 1.2)
    d.go_paths({Hh: ((hold + [0, 0, 0.05], yh, th_h), (hold + [0, 0, 0.01], yh, th_h))}, 0.6)
    # rotationally stiff enough that pressing does not tip the hand (its palm sits 12 cm behind the fingertips)
    impH = d.impedance_on(Hh, k=(2500.0, 2500.0, 800.0), kr=(20.0, 20.0, 20.0), f_max=40.0, tau_max=4.0)
    impH.pos = hold - [0, 0, 15.0 / 800.0]; d.wait(0.5)
    d.note(f"paper: {Hh} hand holds the sheet at {np.round(hold, 3)}")
    try:                                                 # against where the pressing hand really is now
        d.check_legs({F: [q_ for leg in legs for q_ in leg]})
    except RuntimeError as e:
        d.note(f"paper: re-planning the fold arc against the pressing hand ({str(e)[:80]})")
        legs = None
        for lift in (0.7, 0.85, 1.0, 0.55):
            path = [Pc - r * math.cos(t_) * n + np.array([0, 0, 0.0015 + lift * r * math.sin(t_)]) for t_ in th]
            try:
                legs, seed = [], None
                for kk in range(len(path) - 1):
                    legs.append(d.plan_pose(F, path[kk], qf, path[kk + 1], qf, n_via=3, seed=seed, tol=5e-3, rtol=0.06)); seed = legs[-1][-1]
                break
            except RuntimeError:
                legs = None
        if legs is None:
            raise RuntimeError("paper: the fold arc collides with the pressing hand")
    for leg in legs:
        d.follow(F, leg, 0.35)
    d.open_wait({F: 0.03})
    # back the fingers straight out of the fold, towards the robot, then up (whatever of that is reachable)
    try:
        pe, qe = d.cart.tcp_pose(F)
        leg, pb, _ = d.plan_first(F, pe, qe, [pe + [0, -dy, 0.004] for dy in (0.05, 0.04, 0.03)], n_via=6)
        d.follow(F, leg, 0.8)
        leg, _, _ = d.plan_first(F, pb, qe, [pb + [0, -dy, dz] for dz in (0.05, 0.035, 0.02) for dy in (0.02, 0.0)], n_via=6)
        d.follow(F, leg, 0.6)
    except RuntimeError:
        pass
    # the hold hand lets go and goes home (its job ends with the fold), then the closed fold hand presses the crease
    # down with three taps along it; with the hold hand still there the two hands would meet over the crease
    impH.pos = hold + [0, 0, 0.05]; d.wait(0.5); d.impedance_off(Hh)
    try:
        d.go({Hh: d.cart.home_qpos[d.cart.arm_dofs[Hh]]}, 1.2)
    except RuntimeError as e:
        d.note(f"paper: the hold hand stays up ({e})")
    d.cart.set_gripper(F, 0.0)
    tdir = np.array([-1.0, 1.0, 0.0]) / math.sqrt(2.0)
    for u in (-0.025, 0.0, 0.025):
        p = C + tdir * u - n * 0.012; p[2] = S.H + 0.002
        q = None
        for th_t in (0.0, 0.3, 0.6):
            try:                                         # the press itself runs in impedance: check it down to its lowest point
                q = d.ik(F, p + [0, 0, 0.04], yaw=0.0, tilt=th_t)
                qt = down_quat(0.0, th_t)
                d.check_legs({F: d.plan_pose(F, p + [0, 0, 0.04], qt, p - [0, 0, 0.012], qt, n_via=6, seed=q)})
                break
            except RuntimeError:
                q = None
        if q is None:
            d.note(f"paper: crease tap at {np.round(p, 3)} skipped (would hit the other hand)")
            continue
        try:
            d.go({F: q}, 0.7)
        except RuntimeError:
            continue
        imp = d.impedance_on(F, k=(2500.0, 2500.0, 800.0), kr=(3.0, 3.0, 3.0), f_max=20.0, tau_max=2.0)
        imp.pos = p - [0, 0, 8.0 / 800.0]; d.wait(0.5); imp.pos = p + [0, 0, 0.04]; d.wait(0.4); d.impedance_off(F)
    v = V(); strip = d.paper_strip
    flipped = float((((v[strip] - C) @ n) > 0).mean())
    # a fold puts each corner vertex where its mirror image across the crease was (sliding the corner over the crease
    # line also puts it "past the crease")
    mir = d.paper_v0[strip] - 2 * (((d.paper_v0[strip] - C) @ n)[:, None] * n)
    folded = float((np.linalg.norm(v[strip][:, :2] - mir[:, :2], axis=1) < 0.015).mean())
    d.note(f"paper: {100 * flipped:.0f}% of the corner triangle lies past the crease, {100 * folded:.0f}% within 15 mm of its "
           f"folded position; sheet top {1000 * (v[:, 2].max() - S.H):.0f} mm")
    d.home(1.5); d.wait(1.0)
    v = V()
    mir = d.paper_v0[d.paper_strip] - 2 * (((d.paper_v0[d.paper_strip] - C) @ n)[:, None] * n)
    return dict(sheet_maxz_mm=log[-1]["sheet_maxz_mm"], overhang_mm=log[-1]["overhang_mm"],
                corner_past_crease_frac=float((((v[d.paper_strip] - C) @ n) > 0).mean()),
                corner_folded_frac=float((np.linalg.norm(v[d.paper_strip][:, :2] - mir[:, :2], axis=1) < 0.015).mean()))


def pad_pick(d, log):
    """the right hand, top-down with its fingers 7 cm apart, pushes the mouse pad sideways by its left edge (a push in
    the pad's plane only has to beat the pad's own sliding friction; pressing on its top cannot drag a rubber-backed
    pad), so that its right side hangs 6 cm over the desk's right edge. It then pinches that overhang from the front
    with one finger above and one below it, and lifts the pad by it: the pad hangs like heavy cloth and the mouse
    slides off."""
    from genesis.utils.misc import tensor_to_array
    from yam_cart import down_quat
    V = lambda: tensor_to_array(d.E["mousepad"].get_state().pos)[0]
    ex = 0.5 * json_layout()["table"]["size"][0]

    def rec(dd):
        if dd.k % 5 == 0:
            v = V(); log.append(dict(t=round(dd.t, 4), overhang_mm=1000 * float(v[:, 0].max() - ex), pad_maxz_mm=1000 * float(v[:, 2].max() - S.H),
                                     mouse_z_mm=1000 * float(dd.pose("mouse")[2, 3] - S.H), hand_N=float(np.linalg.norm(R.hand_force("right")))))
    d.on_step.append(rec)
    d.wait(0.5)
    v = V(); x0, x1 = float(v[:, 0].min()), float(v[:, 0].max()); y0 = float(v[:, 1].min())
    bx = 0.35
    d.go({}, 1.4, base_x=bx)
    if x1 - ex >= 0.035:
        d.note(f"pad: starts with its right side {1000 * (x1 - ex):.0f} mm over the desk edge (layout shifted for this demo)")
        return _pad_lift(d, log, V, ex)
    yc = y0 + 0.07                                       # front part of the left edge, clear of the mouse
    d.note(f"pad: x {x0:.3f}..{x1:.3f} (desk edge {ex:.3f}); pushing its left edge at y {yc:.3f}")
    # the fingertip pad boxes reach the TCP. Under a soft vertical impedance they climb the pad's rounded 3 mm edge and
    # ride over it, so the push runs in stiff joint-position control with the boxes 1 mm above the desk: they meet
    # the edge square on and cannot climb
    a = np.array([x0 - 0.015, yc, S.H + 0.001]); yaw = d.grasp_yaw("right", a, math.pi / 2)
    d.cart.set_gripper("right", 0.07)
    d.go({"right": d.ik("right", a + [0, 0, 0.05], yaw=yaw)}, 1.6)
    d.go_paths({"right": ((a + [0, 0, 0.05], yaw, 0.0), (a, yaw, 0.0))}, 1.0)
    push = (ex - x1) + 0.06 + 0.015; b = a + [push, 0, 0]
    d.go_paths({"right": ((a, yaw, 0.0), (b, yaw, 0.0))}, 3.0, n_via=12)
    d.wait(0.4)
    d.note(f"pad: pushed; overhang {log[-1]['overhang_mm']:.0f} mm, peak hand force {max(r['hand_N'] for r in log):.1f} N")
    d.go_paths({"right": ((b, yaw, 0.0), (b + [-0.01, 0, 0.05], yaw, 0.0))}, 0.8)
    v = V()
    if v[:, 0].max() - ex < 0.035:
        raise RuntimeError("pad: the push did not make enough overhang")
    return _pad_lift(d, log, V, ex)


def _pad_lift(d, log, V, ex):
    """pinch the pad's overhang from the front, just beyond the desk's right side (approach towards +y, level or
    pitched down, fingers closing vertically: one over the pad, one under it, both clear of the desk), and lift the pad
    by that corner. A side approach along -x from beyond the desk's right edge only reaches with the arm folded on
    itself."""
    from yam_cart import down_quat
    v = V()
    x_edge = float(v[:, 0].max()); yfr = float(v[:, 1].min()); choice = None; errs = []
    alts = (down_quat(0.0, 1.2), down_quat(math.pi / 2, 1.2), down_quat(0.0, 0.9))
    # the pinch is only taken in an arm configuration from which the lift is collision-free as well
    for b in (0.20, 0.25, 0.15, 0.30):
        for pitch in (0.3, 0.0, 0.5):
            ap = np.array([0.0, math.cos(pitch), -math.sin(pitch)])
            for sc in (1.0, -1.0):
                qq = hand_quat(ap, [0.0, 0.0, sc])
                for dx in (0.017, 0.022):
                    # the pad's front edge where the fingers go in (it is not square to the desk), and a bite deep enough
                    # that the whole 4 cm finger pad lies over pad material (a 1 cm bite closed on nothing)
                    px = x_edge - dx; col_ = v[np.abs(v[:, 0] - px) < 0.015]
                    y_loc = float(col_[:, 1].min()) if len(col_) else yfr
                    near = v[np.linalg.norm(v[:, :2] - [px, y_loc + 0.045], axis=1) < 0.02]
                    zo = float(np.median(near[:, 2])) if len(near) else S.H
                    pin = np.array([px, y_loc + 0.045, zo]); pre = pin - ap * 0.09
                    lifts = [pin + [ddx, dy_, h] for h in (0.22, 0.18, 0.14) for ddx, dy_ in ((0.0, -0.04), (-0.04, -0.06), (0.0, 0.0))]
                    for q_pre in d.ik_all("right", pre, quat=qq, base_x=b, tol=4e-3, rtol=0.05)[:4]:
                        try:
                            leg_in = d.plan_pose("right", pre, qq, pin, qq, n_via=6, seed=q_pre, tol=5e-3, rtol=0.06, base_x=b)
                        except RuntimeError as e:
                            errs.append(str(e)); continue
                        for tgt, qa in [(t_, q_) for t_ in lifts for q_ in (qq, *alts)]:
                            try:
                                leg_up = d.plan_pose("right", pin, qq, tgt, qa, n_via=10, seed=leg_in[-1], base_x=b)
                            except RuntimeError as e:
                                errs.append(str(e)); continue
                            choice = (b, pin, pre, q_pre, leg_in, qq, leg_up, tgt, zo); break
                        if choice: break
                    if choice: break
                if choice: break
            if choice: break
        if choice: break
    if choice is None:
        raise RuntimeError("pad: no reachable front pinch and lift of the overhang: " + " || ".join(e[:100] for e in errs[-3:]))
    b, pin, pre, q_pre, leg_in, qq, leg_up, top, zo = choice
    d.go({}, 1.2, base_x=b); d.cart.set_gripper("right", 0.04)
    d.go({"right": q_pre}, 1.8)
    d.follow("right", leg_in, 1.2)
    d.cart.grasp("right", force=15.0); d.grip_until("right", 1.0)
    w = d.cart.gripper_width("right")
    d.note(f"pad: overhang pinched at {np.round(pin, 3)}, width {w * 1e3:.1f} mm")
    # the finger joints read anywhere from 0 to 1.7 mm on the 3 mm pad (the soft Genesis-IPC coupling), so whether the
    # pad is held is tested by lifting a little: the pad next to the fingers must come up with them
    k3 = max(2, len(leg_up) // 6)
    d.follow("right", leg_up[:k3 + 1], 0.6)
    p_now = d.tcp("right"); v = V(); near = v[np.linalg.norm(v[:, :2] - p_now[:2], axis=1) < 0.03]
    rise = float(near[:, 2].max() - zo) if len(near) else 0.0
    d.note(f"pad: hand {1000 * (p_now[2] - pin[2]):.0f} mm up, pad beside the fingers {1000 * rise:.0f} mm up")
    if rise < 0.5 * (p_now[2] - pin[2]) and not getattr(d, "dry", False):   # (a dry run moves no pad)
        raise RuntimeError("pad: the hand closed on nothing")
    d.note(f"pad: lifting it by the corner to {1000 * (top[2] - S.H):.0f} mm")
    d.follow("right", leg_up[k3:], 2.2)
    d.wait(2.0)
    return dict(pad_maxz_mm=max(r["pad_maxz_mm"] for r in log), mouse_z_end_mm=log[-1]["mouse_z_mm"],
                push_peak_N=max(r["hand_N"] for r in log))


def json_layout():
    import json as _j
    return _j.load(open(os.path.join(S.ROOT, "layout.json")))


DEMOS = dict(
    marker=(marker, ("paper_top", "paper_bottom"), dict(pos=(0.13, 0.14, 0.95), lookat=(0.11, -0.22, 0.81), fov=38)),
    tin=(tin, (), dict(pos=(0.78, 0.02, 1.02), lookat=(0.30, 0.10, 0.86), fov=42)),
    jar=(jar, (), dict(pos=(0.80, 0.10, 1.10), lookat=(0.45, -0.25, 0.86), fov=42)),
    monitor=(monitor, (), dict(pos=(-0.90, -0.05, 1.27), lookat=(-0.05, 0.27, 1.04), fov=40)),
    bottle=(bottle, ("mousepad",), dict(pos=(0.95, -0.15, 1.10), lookat=(0.50, 0.10, 0.85), fov=40)),
    cloth_fold=(cloth_fold, ("cloth",), dict(pos=(-0.95, -0.80, 1.05), lookat=(-0.28, -0.35, 0.72), fov=40)),
    paper_fold=(paper_fold, ("paper_top", "paper_bottom"), dict(pos=(0.55, -0.70, 1.05), lookat=(0.12, -0.32, 0.78), fov=36)),
    pad_pick=(pad_pick, ("mousepad",), dict(pos=(1.15, 0.25, 1.15), lookat=(0.55, -0.08, 0.85), fov=42)),
    cloth_drop=(cloth_drop, ("cloth",), dict(pos=(0.25, -1.05, 1.20), lookat=(-0.30, -0.30, 0.85), fov=46)),
)

# per-demo cart placement (robot_desk.Desk lift / back): the lift lowers the arm deck below the tabletop, back parks the
# cart further from the desk's front edge; env CART_LIFT / CART_BACK override
# rigid objects a demo takes off the desk first (the paper fold needs the sheet clear)
DROP = dict(paper_fold=("marker", "marker_cap"))

CART = dict(paper_fold=dict(back=0.30, lift=0.15), cloth_fold=dict(back=0.30, lift=0.20), pad_pick=dict(back=0.15, lift=0.10))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=list(DEMOS)); ap.add_argument("--out", default=os.path.join(S.ROOT, "out", "robot"))
    ap.add_argument("--res", type=int, nargs=2, default=(960, 720)); ap.add_argument("--nyx", action="store_true")
    ap.add_argument("--bare", action="store_true")   # no deformables (quick debugging)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fn, keep, cam = DEMOS[a.demo]
    for k, v in HINGE_ENV.get(a.demo, {}).items():
        os.environ.setdefault(k, v)
    gs.init(backend=gs.gpu, logging_level="warning")
    t0 = time.time()
    cc = CART.get(a.demo, {})
    lift = float(os.environ.get("CART_LIFT", cc.get("lift", 0.0))); back = float(os.environ.get("CART_BACK", cc.get("back", 0.0)))
    d = R.Desk(keep=() if a.bare else keep, cart_x=0.0, cams={"demo": dict(res=tuple(a.res), **cam)}, lift=lift, back=back,
               drop=DROP.get(a.demo, ()))
    d.note(f"cart: lift {lift:.3f} m, back {back:.3f} m")
    print(f"build {time.time() - t0:.0f}s", flush=True)
    log = []
    d.start_video(os.path.join(a.out, f"{a.demo}.mp4")); d.start_record(os.path.join(a.out, f"{a.demo}_states.npz"))
    d.start_export(os.path.join(a.out, "export"), camera=cam); t0 = time.time()
    try:
        res = fn(d, log)
    finally:
        d.stop_video(); d.save_record(); d.close_export()
        json.dump(dict(log=log, events=d.events), open(os.path.join(a.out, f"{a.demo}.json"), "w"), indent=1)
    print("RESULT", a.demo, json.dumps(res), f"{(time.time() - t0) / max(d.k, 1) * 1000:.0f} ms/step", flush=True)
    json.dump(dict(result=res, log=log, events=d.events), open(os.path.join(a.out, f"{a.demo}.json"), "w"), indent=1)
