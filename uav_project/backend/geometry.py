"""
3D geometry model of the aircraft, used to drive the Three.js visualization.

This module is the SINGLE SOURCE OF TRUTH for the geometry math. The JS
port (frontend/js/geometry.js) must reproduce every formula here exactly.
Run this file directly to execute the numeric verification tests.

Frames
------
Body frame:  x=forward, y=right, z=down  (same frame used in dynamics_6dof.py)
World/plot frame: (X,Y,Z) = (East, North, Up) = (ned_e, ned_n, -ned_d)

Rotation
--------
Body->NED uses the same 3-2-1 (yaw-pitch-roll) direction-cosine matrix as
the physics (dynamics_6dof.dynamics / trim.py), so the model drawn on
screen is kinematically identical to the simulated aircraft.

Dimensions
----------
REAL (drive both physics and geometry, from aircraft_params):
    b (wingspan), c (wing chord), S_tail, l_tail
COSMETIC ONLY (visual fidelity, have NO effect on the physics):
    nose_len, tail_ext, fuse_r, tail_AR, vfin_frac, aileron_span_f,
    aileron_chord_f, elevator_chord_f, rudder_chord_f
"""
import numpy as np
import aircraft_params as P

# ---------------------------------------------------------------------------
# Cosmetic-only dimensions (NOT physics -- purely for the 3D model)
# ---------------------------------------------------------------------------
NOSE_LEN = 0.35
TAIL_EXT = 0.15
FUSE_R = 0.05
TAIL_AR = 4.0
VFIN_FRAC = 0.5
AILERON_SPAN_F = 0.25
AILERON_CHORD_F = 0.3
ELEVATOR_CHORD_F = 0.35
RUDDER_CHORD_F = 0.4

TAIL_SPAN = np.sqrt(P.S_tail * TAIL_AR)
TAIL_CHORD = P.S_tail / TAIL_SPAN


def real_from_backend():
    return {"b": P.b, "c": P.c, "S_tail": P.S_tail, "l_tail": P.l_tail,
            "S": P.S, "AR": P.AR}


def cosmetic_only():
    return {"nose_len": NOSE_LEN, "tail_ext": TAIL_EXT, "fuse_r": FUSE_R,
            "tail_AR": TAIL_AR, "tail_span": TAIL_SPAN, "tail_chord": TAIL_CHORD,
            "vfin_frac": VFIN_FRAC, "aileron_span_f": AILERON_SPAN_F,
            "aileron_chord_f": AILERON_CHORD_F,
            "elevator_chord_f": ELEVATOR_CHORD_F,
            "rudder_chord_f": RUDDER_CHORD_F}


# ---------------------------------------------------------------------------
# Body <-> world transforms
# ---------------------------------------------------------------------------
def dcm_body_to_ned(phi, theta, psi):
    """3-2-1 (yaw-pitch-roll) body->NED direction cosine matrix. Identical
    in form to the position-kinematics used in dynamics_6dof.dynamics()."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R = np.array([
        [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
        [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
        [-sth,       sphi * cth,                       cphi * cth],
    ])
    return R


def body_vec_to_world(v_body, R):
    """Rotate a body-frame vector into the world (East,North,Up) frame."""
    ned = R @ np.asarray(v_body, dtype=float)  # (n, e, d)
    return np.array([ned[1], ned[0], -ned[2]])  # (E, N, Up)


def body_point_to_world(point_body, R, pos_ned):
    """Transform a body-fixed point (already including any control-surface
    hinge deflections applied in body coordinates) to world coordinates,
    given the aircraft's NED position (pn, pe, pd)."""
    pos_world = np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])
    return body_vec_to_world(point_body, R) + pos_world


# ---------------------------------------------------------------------------
# Hinge-rotation primitives
# ---------------------------------------------------------------------------
def rotate_xz(x, z, pivot_x, pivot_z, angle):
    """Rotate point (x,z) about (pivot_x,pivot_z) in the body x-z plane.

    Convention (verified by test_elevator_sign / test_aileron_sign below):
    +angle moves a point AFT of the pivot (smaller/more-negative local x)
    in the +z (DOWN) direction -- i.e. a trailing edge behind a hinge
    deflects downward for positive angle, matching Cm_de<0, Cm_dtail<0.
    """
    dx, dz = x - pivot_x, z - pivot_z
    new_dx = dx * np.cos(angle) + dz * np.sin(angle)
    new_dz = -dx * np.sin(angle) + dz * np.cos(angle)
    return pivot_x + new_dx, pivot_z + new_dz


def rotate_xy(x, y, pivot_x, pivot_y, angle):
    """Rotate point (x,y) about (pivot_x,pivot_y) in the body x-y plane, for
    the rudder's vertical hinge. Convention chosen (and documented) here:
    +angle (positive rudder deflection) moves a point aft of the hinge in
    the +y (RIGHT) direction, matching CY_dr>0 / Cn_dr<0 (a rudder that
    kicks the tail right, yawing the nose left). This convention is checked by
    test_rudder_sign()."""
    dx, dy = x - pivot_x, y - pivot_y
    # Body x is forward and y is right. This sign convention makes a
    # positive rudder move its aft edge to +y, matching CY_dr > 0 and
    # Cn_dr < 0 in the aerodynamic model.
    new_dx = dx * np.cos(angle) + dy * np.sin(angle)
    new_dy = -dx * np.sin(angle) + dy * np.cos(angle)
    return pivot_x + new_dx, pivot_y + new_dy


# ---------------------------------------------------------------------------
# Part generators (local, undeflected, body-fixed coordinates)
# All points are (x, y, z) tuples/arrays in the BODY frame with the
# aircraft CG at the origin.
# ---------------------------------------------------------------------------
def generate_fuselage():
    """9-point tapered-box fuselage, 14 triangles.
    1 nose tip + 4-point front ring + 4-point rear ring."""
    nose = np.array([NOSE_LEN, 0.0, 0.0])

    front_x = 0.0
    front_ring = np.array([
        [front_x,  FUSE_R,  FUSE_R],
        [front_x,  FUSE_R, -FUSE_R],
        [front_x, -FUSE_R, -FUSE_R],
        [front_x, -FUSE_R,  FUSE_R],
    ])

    rear_x = -(P.l_tail + TAIL_EXT)
    rear_r = FUSE_R * 0.4  # taper
    rear_ring = np.array([
        [rear_x,  rear_r,  rear_r],
        [rear_x,  rear_r, -rear_r],
        [rear_x, -rear_r, -rear_r],
        [rear_x, -rear_r,  rear_r],
    ])

    points = np.vstack([nose[None, :], front_ring, rear_ring])  # 9 x 3
    # indices: 0=nose, 1-4=front ring (i), 5-8=rear ring (i)
    tris = []
    # nose cone: 4 triangles (nose -> each front-ring edge)
    for i in range(4):
        a = 1 + i
        b_ = 1 + (i + 1) % 4
        tris.append((0, a, b_))
    # side walls: front ring -> rear ring, 4 quads -> 8 triangles
    for i in range(4):
        f0 = 1 + i
        f1 = 1 + (i + 1) % 4
        r0 = 5 + i
        r1 = 5 + (i + 1) % 4
        tris.append((f0, f1, r1))
        tris.append((f0, r1, r0))
    # rear cap: 1 quad -> 2 triangles
    tris.append((5, 6, 7))
    tris.append((5, 7, 8))

    assert len(points) == 9
    assert len(tris) == 14
    return points, np.array(tris)


def _quad(p1, p2, p3, p4):
    """Return 4 points + 2 triangles (as local indices 0-3) for a flat quad."""
    return np.array([p1, p2, p3, p4]), np.array([(0, 1, 2), (0, 2, 3)])


def generate_wing():
    """Wing: fixed root/forward panels + two outer aft aileron sub-panels."""
    half_b = P.b / 2.0
    LE_x, TE_x = P.c / 2.0, -P.c / 2.0
    hinge_x = TE_x + AILERON_CHORD_F * P.c  # aileron hinge line (chordwise)
    aileron_y0 = half_b * (1.0 - AILERON_SPAN_F)  # inboard edge of aileron

    parts = {}

    # Forward fixed strip: full span, LE to hinge_x
    parts["wing_fixed_forward"] = _quad(
        (LE_x, -half_b, 0), (LE_x, half_b, 0),
        (hinge_x, half_b, 0), (hinge_x, -half_b, 0))

    # Root aft fixed strip (inboard of ailerons), hinge_x to TE
    parts["wing_fixed_root_aft"] = _quad(
        (hinge_x, -aileron_y0, 0), (hinge_x, aileron_y0, 0),
        (TE_x, aileron_y0, 0), (TE_x, -aileron_y0, 0))

    # Aileron sub-panels (outer span, aft chord)
    right_pts, right_tris = _quad(
        (hinge_x, aileron_y0, 0), (hinge_x, half_b, 0),
        (TE_x, half_b, 0), (TE_x, aileron_y0, 0))
    left_pts, left_tris = _quad(
        (hinge_x, -half_b, 0), (hinge_x, -aileron_y0, 0),
        (TE_x, -aileron_y0, 0), (TE_x, -half_b, 0))

    parts["aileron_right"] = (right_pts, right_tris, hinge_x)
    parts["aileron_left"] = (left_pts, left_tris, hinge_x)
    return parts


def generate_tail():
    """Horizontal tailplane (tail_fixed + elevator) and vertical fin
    (fin_fixed + rudder), both mounted at x = -l_tail."""
    tail_LE_x = -P.l_tail
    tail_TE_x = -P.l_tail - TAIL_CHORD
    elev_hinge_x = tail_TE_x + ELEVATOR_CHORD_F * TAIL_CHORD
    half_span = TAIL_SPAN / 2.0

    tail_fixed = _quad(
        (tail_LE_x, -half_span, 0), (tail_LE_x, half_span, 0),
        (elev_hinge_x, half_span, 0), (elev_hinge_x, -half_span, 0))
    elevator_pts, elevator_tris = _quad(
        (elev_hinge_x, -half_span, 0), (elev_hinge_x, half_span, 0),
        (tail_TE_x, half_span, 0), (tail_TE_x, -half_span, 0))

    fin_height = VFIN_FRAC * TAIL_SPAN
    fin_TE_x = tail_TE_x  # same chord scale, same x-station as tail
    fin_LE_x = tail_LE_x
    rud_hinge_x = fin_TE_x + RUDDER_CHORD_F * TAIL_CHORD

    fin_fixed = _quad(
        (fin_LE_x, 0, 0), (fin_LE_x, 0, -fin_height),
        (rud_hinge_x, 0, -fin_height), (rud_hinge_x, 0, 0))
    rudder_pts, rudder_tris = _quad(
        (rud_hinge_x, 0, 0), (rud_hinge_x, 0, -fin_height),
        (fin_TE_x, 0, -fin_height), (fin_TE_x, 0, 0))

    return {
        "tail_fixed": tail_fixed,
        "elevator": (elevator_pts, elevator_tris, elev_hinge_x),
        "fin_fixed": fin_fixed,
        "rudder": (rudder_pts, rudder_tris, rud_hinge_x),
        "tail_LE_x": tail_LE_x,
    }


# ---------------------------------------------------------------------------
# Apply control-surface deflections (still in BODY / local coordinates)
# ---------------------------------------------------------------------------
def deflect_elevator(elevator_pts, elev_hinge_x, tail_LE_x, de, tail_incidence):
    """Composed rotation: tail_incidence first rotates the WHOLE tail
    (including the elevator and its hinge) about the tail's leading-edge
    hinge; THEN de rotates the elevator about its own (now-moved) hinge."""
    pts = elevator_pts.copy()
    # Step 1: tail_incidence rotates elevator points about tail LE hinge (x-z plane, y unaffected)
    for i in range(len(pts)):
        x, z = rotate_xz(pts[i, 0], pts[i, 2], tail_LE_x, 0.0, tail_incidence)
        pts[i, 0], pts[i, 2] = x, z
    # The elevator hinge point itself also moves under the tail_incidence rotation
    new_hinge_x, new_hinge_z = rotate_xz(elev_hinge_x, 0.0, tail_LE_x, 0.0, tail_incidence)
    # Step 2: de rotates the elevator about its NEW hinge position
    for i in range(len(pts)):
        x, z = rotate_xz(pts[i, 0], pts[i, 2], new_hinge_x, new_hinge_z, de)
        pts[i, 0], pts[i, 2] = x, z
    return pts


def deflect_tail_fixed(tail_fixed_pts, tail_LE_x, tail_incidence):
    pts = tail_fixed_pts.copy()
    for i in range(len(pts)):
        x, z = rotate_xz(pts[i, 0], pts[i, 2], tail_LE_x, 0.0, tail_incidence)
        pts[i, 0], pts[i, 2] = x, z
    return pts


def deflect_aileron(pts, hinge_x, da, side):
    """side: 'right' -> angle=-da, 'left' -> angle=+da (opposite signs)."""
    angle = -da if side == "right" else da
    out = pts.copy()
    for i in range(len(out)):
        x, z = rotate_xz(out[i, 0], out[i, 2], hinge_x, 0.0, angle)
        out[i, 0], out[i, 2] = x, z
    return out


def deflect_rudder(pts, hinge_x, dr):
    out = pts.copy()
    for i in range(len(out)):
        x, y = rotate_xy(out[i, 0], out[i, 1], hinge_x, 0.0, dr)
        out[i, 0], out[i, 1] = x, y
    return out


# ---------------------------------------------------------------------------
# Full assembly: local (deflected) body coords -> world coords
# ---------------------------------------------------------------------------
def assemble_world(state, controls, tail_incidence=0.0):
    """state: [pn,pe,pd,u,v,w,phi,theta,psi,p,q,r] (only pos+attitude used)
    controls: [de,da,dr,throttle] (throttle unused for geometry)
    Returns a dict of part_name -> (world_points (Nx3), triangle indices)."""
    pn, pe, pd = state[0], state[1], state[2]
    phi, theta, psi = state[6], state[7], state[8]
    de, da, dr, _throttle = controls

    R = dcm_body_to_ned(phi, theta, psi)
    pos_ned = (pn, pe, pd)

    def to_world(local_pts):
        return np.array([body_point_to_world(p, R, pos_ned) for p in local_pts])

    out = {}

    fus_pts, fus_tris = generate_fuselage()
    out["fuselage"] = (to_world(fus_pts), fus_tris)

    wing = generate_wing()
    wf_pts, wf_tris = wing["wing_fixed_forward"]
    wr_pts, wr_tris = wing["wing_fixed_root_aft"]
    out["wing_fixed_forward"] = (to_world(wf_pts), wf_tris)
    out["wing_fixed_root_aft"] = (to_world(wr_pts), wr_tris)

    ar_pts, ar_tris, ar_hinge = wing["aileron_right"]
    al_pts, al_tris, al_hinge = wing["aileron_left"]
    out["aileron_right"] = (to_world(deflect_aileron(ar_pts, ar_hinge, da, "right")), ar_tris)
    out["aileron_left"] = (to_world(deflect_aileron(al_pts, al_hinge, da, "left")), al_tris)

    tail = generate_tail()
    tf_pts, tf_tris = tail["tail_fixed"]
    out["tail_fixed"] = (to_world(deflect_tail_fixed(tf_pts, tail["tail_LE_x"], tail_incidence)), tf_tris)

    ev_pts, ev_tris, ev_hinge = tail["elevator"]
    out["elevator"] = (to_world(deflect_elevator(ev_pts, ev_hinge, tail["tail_LE_x"], de, tail_incidence)), ev_tris)

    out["fin_fixed"] = tail["fin_fixed"][0], tail["fin_fixed"][1]
    out["fin_fixed"] = (to_world(tail["fin_fixed"][0]), tail["fin_fixed"][1])

    rud_pts, rud_tris, rud_hinge = tail["rudder"]
    out["rudder"] = (to_world(deflect_rudder(rud_pts, rud_hinge, dr)), rud_tris)

    return out


# ---------------------------------------------------------------------------
# Numeric verification tests
# ---------------------------------------------------------------------------
def test_nose_tip_sanity():
    for theta in np.radians([-30, -5, 0, 5, 20, 45]):
        state = [0, 0, 0, 0, 0, 0, 0.0, theta, 0.0, 0, 0, 0]
        controls = [0, 0, 0, 0]
        world = assemble_world(state, controls)
        fus_world, _ = world["fuselage"]
        nose_world = fus_world[0]  # nose tip is point index 0
        expected_up = NOSE_LEN * np.sin(theta)
        assert abs(nose_world[2] - expected_up) < 1e-9, (theta, nose_world[2], expected_up)
    print("test_nose_tip_sanity: PASS")


def test_elevator_sign():
    state = [0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0]
    de = np.radians(10.0)
    w_zero = assemble_world(state, [0, 0, 0, 0], tail_incidence=0.0)
    w_defl = assemble_world(state, [de, 0, 0, 0], tail_incidence=0.0)
    up_zero = w_zero["elevator"][0][:, 2].mean()
    up_defl = w_defl["elevator"][0][:, 2].mean()
    assert up_defl < up_zero, (up_defl, up_zero)
    print(f"test_elevator_sign: PASS (up_zero={up_zero:.4f}, up_defl(+10deg)={up_defl:.4f})")


def test_aileron_sign():
    state = [0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0]
    da = np.radians(10.0)
    world = assemble_world(state, [0, da, 0, 0])
    right_up = world["aileron_right"][0][:, 2].mean()
    left_up = world["aileron_left"][0][:, 2].mean()
    assert right_up > left_up, (right_up, left_up)
    print(f"test_aileron_sign: PASS (right_up={right_up:.4f}, left_up={left_up:.4f})")


def test_rudder_sign():
    tail = generate_tail()
    rudder_pts, _, hinge_x = tail["rudder"]
    dr = np.radians(10.0)
    deflected = deflect_rudder(rudder_pts, hinge_x, dr)
    aft = rudder_pts[:, 0] < hinge_x - 1e-9
    assert np.all(deflected[aft, 1] > rudder_pts[aft, 1]), (rudder_pts, deflected)
    print("test_rudder_sign: PASS (+dr moves aft rudder edge to +y/right)")


def test_dcm_matches_dynamics():
    """Cross-check dcm_body_to_ned against the position-kinematics formulas
    used directly in dynamics_6dof.dynamics()."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi, theta, psi = rng.uniform(-1.0, 1.0, 3)
        v_body = rng.uniform(-5, 5, 3)
        R = dcm_body_to_ned(phi, theta, psi)
        ned = R @ v_body
        u, v, w = v_body
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        pn_dot = (cth * cpsi) * u + (sphi * sth * cpsi - cphi * spsi) * v + \
                 (cphi * sth * cpsi + sphi * spsi) * w
        pe_dot = (cth * spsi) * u + (sphi * sth * spsi + cphi * cpsi) * v + \
                 (cphi * sth * spsi - sphi * cpsi) * w
        pd_dot = (-sth) * u + (sphi * cth) * v + (cphi * cth) * w
        assert np.allclose(ned, [pn_dot, pe_dot, pd_dot], atol=1e-9)
    print("test_dcm_matches_dynamics: PASS")


def run_all_tests():
    test_dcm_matches_dynamics()
    test_nose_tip_sanity()
    test_elevator_sign()
    test_aileron_sign()
    test_rudder_sign()
    print("All geometry tests passed.")


if __name__ == "__main__":
    run_all_tests()
