// geometry.js
// Faithful JS port of backend/geometry.py. Every formula here must match
// the Python reference exactly (see backend/geometry.py docstring).
// Real dimensions (b, c, S_tail, l_tail) come from the live /api/geometry
// response (real_from_backend) -- NEVER hardcoded here. Cosmetic-only
// constants below match cosmetic_only() in the Python module 1:1 and are
// documented as having no effect on physics.

const COSMETIC = {
  NOSE_LEN: 0.35,
  TAIL_EXT: 0.15,
  FUSE_R: 0.05,
  TAIL_AR: 4.0,
  VFIN_FRAC: 0.5,
  AILERON_SPAN_F: 0.25,
  AILERON_CHORD_F: 0.3,
  ELEVATOR_CHORD_F: 0.35,
  RUDDER_CHORD_F: 0.4,
};

// ---------------------------------------------------------------------
// Body <-> world transforms (identical to geometry.py)
// ---------------------------------------------------------------------
function dcmBodyToNed(phi, theta, psi) {
  const cphi = Math.cos(phi), sphi = Math.sin(phi);
  const cth = Math.cos(theta), sth = Math.sin(theta);
  const cpsi = Math.cos(psi), spsi = Math.sin(psi);
  return [
    [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
    [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
    [-sth, sphi * cth, cphi * cth],
  ];
}

function matVec3(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

function bodyVecToWorld(vBody, R) {
  const ned = matVec3(R, vBody); // (n, e, d)
  return [ned[1], ned[0], -ned[2]]; // (E, N, Up)
}

function bodyPointToWorld(pointBody, R, posNed) {
  const posWorld = [posNed[1], posNed[0], -posNed[2]];
  const w = bodyVecToWorld(pointBody, R);
  return [w[0] + posWorld[0], w[1] + posWorld[1], w[2] + posWorld[2]];
}

// ---------------------------------------------------------------------
// Hinge-rotation primitives (identical convention to geometry.py)
// ---------------------------------------------------------------------
function rotateXZ(x, z, pivotX, pivotZ, angle) {
  const dx = x - pivotX, dz = z - pivotZ;
  const c = Math.cos(angle), s = Math.sin(angle);
  const newDx = dx * c + dz * s;
  const newDz = -dx * s + dz * c;
  return [pivotX + newDx, pivotZ + newDz];
}

function rotateXY(x, y, pivotX, pivotY, angle) {
  const dx = x - pivotX, dy = y - pivotY;
  const c = Math.cos(angle), s = Math.sin(angle);
  const newDx = dx * c + dy * s;
  const newDy = -dx * s + dy * c;
  return [pivotX + newDx, pivotY + newDy];
}

// ---------------------------------------------------------------------
// Part generators. `real` = {b, c, S_tail, l_tail} from /api/geometry.
// ---------------------------------------------------------------------
function tailDims(real) {
  const tailSpan = Math.sqrt(real.S_tail * COSMETIC.TAIL_AR);
  const tailChord = real.S_tail / tailSpan;
  return { tailSpan, tailChord };
}

function generateFuselage(real) {
  const nose = [COSMETIC.NOSE_LEN, 0, 0];
  const frontX = 0;
  const fr = COSMETIC.FUSE_R;
  const frontRing = [
    [frontX, fr, fr], [frontX, fr, -fr],
    [frontX, -fr, -fr], [frontX, -fr, fr],
  ];
  const rearX = -(real.l_tail + COSMETIC.TAIL_EXT);
  const rr = fr * 0.4;
  const rearRing = [
    [rearX, rr, rr], [rearX, rr, -rr],
    [rearX, -rr, -rr], [rearX, -rr, rr],
  ];
  const points = [nose, ...frontRing, ...rearRing]; // 9 points
  const tris = [];
  for (let i = 0; i < 4; i++) {
    const a = 1 + i, b = 1 + ((i + 1) % 4);
    tris.push([0, a, b]);
  }
  for (let i = 0; i < 4; i++) {
    const f0 = 1 + i, f1 = 1 + ((i + 1) % 4);
    const r0 = 5 + i, r1 = 5 + ((i + 1) % 4);
    tris.push([f0, f1, r1]);
    tris.push([f0, r1, r0]);
  }
  tris.push([5, 6, 7]);
  tris.push([5, 7, 8]);
  return { points, tris }; // 9 pts, 14 tris
}

function quad(p1, p2, p3, p4) {
  return { points: [p1, p2, p3, p4], tris: [[0, 1, 2], [0, 2, 3]] };
}

function generateWing(real) {
  const halfB = real.b / 2.0;
  const LE = real.c / 2.0, TE = -real.c / 2.0;
  const hingeX = TE + COSMETIC.AILERON_CHORD_F * real.c;
  const aileronY0 = halfB * (1.0 - COSMETIC.AILERON_SPAN_F);

  const wingFixedForward = quad(
    [LE, -halfB, 0], [LE, halfB, 0], [hingeX, halfB, 0], [hingeX, -halfB, 0]);
  const wingFixedRootAft = quad(
    [hingeX, -aileronY0, 0], [hingeX, aileronY0, 0], [TE, aileronY0, 0], [TE, -aileronY0, 0]);
  const aileronRight = quad(
    [hingeX, aileronY0, 0], [hingeX, halfB, 0], [TE, halfB, 0], [TE, aileronY0, 0]);
  const aileronLeft = quad(
    [hingeX, -halfB, 0], [hingeX, -aileronY0, 0], [TE, -aileronY0, 0], [TE, -halfB, 0]);

  return {
    wingFixedForward, wingFixedRootAft,
    aileronRight: { ...aileronRight, hingeX },
    aileronLeft: { ...aileronLeft, hingeX },
  };
}

function generateTail(real) {
  const { tailSpan, tailChord } = tailDims(real);
  const tailLEx = -real.l_tail;
  const tailTEx = -real.l_tail - tailChord;
  const elevHingeX = tailTEx + COSMETIC.ELEVATOR_CHORD_F * tailChord;
  const halfSpan = tailSpan / 2.0;

  const tailFixed = quad(
    [tailLEx, -halfSpan, 0], [tailLEx, halfSpan, 0], [elevHingeX, halfSpan, 0], [elevHingeX, -halfSpan, 0]);
  const elevator = quad(
    [elevHingeX, -halfSpan, 0], [elevHingeX, halfSpan, 0], [tailTEx, halfSpan, 0], [tailTEx, -halfSpan, 0]);

  const finHeight = COSMETIC.VFIN_FRAC * tailSpan;
  const finTEx = tailTEx, finLEx = tailLEx;
  const rudHingeX = finTEx + COSMETIC.RUDDER_CHORD_F * tailChord;

  const finFixed = quad(
    [finLEx, 0, 0], [finLEx, 0, -finHeight], [rudHingeX, 0, -finHeight], [rudHingeX, 0, 0]);
  const rudder = quad(
    [rudHingeX, 0, 0], [rudHingeX, 0, -finHeight], [finTEx, 0, -finHeight], [finTEx, 0, 0]);

  return {
    tailFixed,
    elevator: { ...elevator, hingeX: elevHingeX },
    finFixed,
    rudder: { ...rudder, hingeX: rudHingeX },
    tailLEx,
  };
}

// ---------------------------------------------------------------------
// Deflection (still body/local coordinates)
// ---------------------------------------------------------------------
function deflectElevator(pts, elevHingeX, tailLEx, de, tailIncidence) {
  let out = pts.map((p) => p.slice());
  for (let i = 0; i < out.length; i++) {
    const [x, z] = rotateXZ(out[i][0], out[i][2], tailLEx, 0.0, tailIncidence);
    out[i][0] = x; out[i][2] = z;
  }
  const [newHingeX, newHingeZ] = rotateXZ(elevHingeX, 0.0, tailLEx, 0.0, tailIncidence);
  for (let i = 0; i < out.length; i++) {
    const [x, z] = rotateXZ(out[i][0], out[i][2], newHingeX, newHingeZ, de);
    out[i][0] = x; out[i][2] = z;
  }
  return out;
}

function deflectTailFixed(pts, tailLEx, tailIncidence) {
  return pts.map((p) => {
    const [x, z] = rotateXZ(p[0], p[2], tailLEx, 0.0, tailIncidence);
    return [x, p[1], z];
  });
}

function deflectAileron(pts, hingeX, da, side) {
  const angle = side === "right" ? -da : da;
  return pts.map((p) => {
    const [x, z] = rotateXZ(p[0], p[2], hingeX, 0.0, angle);
    return [x, p[1], z];
  });
}

function deflectRudder(pts, hingeX, dr) {
  return pts.map((p) => {
    const [x, y] = rotateXY(p[0], p[1], hingeX, 0.0, dr);
    return [x, y, p[2]];
  });
}

// ---------------------------------------------------------------------
// Full assembly -> world coordinates
// state: {pn,pe,pd,phi,theta,psi}, controls: {de,da,dr}, tailIncidence
// ---------------------------------------------------------------------
function assembleWorld(real, state, controls, tailIncidence = 0.0) {
  const { pn, pe, pd, phi, theta, psi } = state;
  const { de, da, dr } = controls;
  const R = dcmBodyToNed(phi, theta, psi);
  const posNed = [pn, pe, pd];

  const toWorld = (pts) => pts.map((p) => bodyPointToWorld(p, R, posNed));

  const out = {};

  const fus = generateFuselage(real);
  out.fuselage = { points: toWorld(fus.points), tris: fus.tris };

  const wing = generateWing(real);
  out.wingFixedForward = { points: toWorld(wing.wingFixedForward.points), tris: wing.wingFixedForward.tris };
  out.wingFixedRootAft = { points: toWorld(wing.wingFixedRootAft.points), tris: wing.wingFixedRootAft.tris };
  out.aileronRight = {
    points: toWorld(deflectAileron(wing.aileronRight.points, wing.aileronRight.hingeX, da, "right")),
    tris: wing.aileronRight.tris,
  };
  out.aileronLeft = {
    points: toWorld(deflectAileron(wing.aileronLeft.points, wing.aileronLeft.hingeX, da, "left")),
    tris: wing.aileronLeft.tris,
  };

  const tail = generateTail(real);
  out.tailFixed = {
    points: toWorld(deflectTailFixed(tail.tailFixed.points, tail.tailLEx, tailIncidence)),
    tris: tail.tailFixed.tris,
  };
  out.elevator = {
    points: toWorld(deflectElevator(tail.elevator.points, tail.elevator.hingeX, tail.tailLEx, de, tailIncidence)),
    tris: tail.elevator.tris,
  };
  out.finFixed = { points: toWorld(tail.finFixed.points), tris: tail.finFixed.tris };
  out.rudder = {
    points: toWorld(deflectRudder(tail.rudder.points, tail.rudder.hingeX, dr)),
    tris: tail.rudder.tris,
  };

  return out;
}

// ---------------------------------------------------------------------
// Console self-test (open devtools to see) -- mirrors geometry.py tests
// ---------------------------------------------------------------------
function runSelfTests(real) {
  // nose tip sanity
  for (const thetaDeg of [-30, -5, 0, 5, 20, 45]) {
    const theta = (thetaDeg * Math.PI) / 180;
    const world = assembleWorld(real, { pn: 0, pe: 0, pd: 0, phi: 0, theta, psi: 0 }, { de: 0, da: 0, dr: 0 });
    const nose = world.fuselage.points[0];
    const expected = COSMETIC.NOSE_LEN * Math.sin(theta);
    console.assert(Math.abs(nose[2] - expected) < 1e-9, "nose tip sanity failed", thetaDeg);
  }
  // elevator sign
  {
    const st = { pn: 0, pe: 0, pd: 0, phi: 0, theta: 0, psi: 0 };
    const wZero = assembleWorld(real, st, { de: 0, da: 0, dr: 0 }, 0);
    const wDefl = assembleWorld(real, st, { de: (10 * Math.PI) / 180, da: 0, dr: 0 }, 0);
    const upZero = wZero.elevator.points.reduce((s, p) => s + p[2], 0) / wZero.elevator.points.length;
    const upDefl = wDefl.elevator.points.reduce((s, p) => s + p[2], 0) / wDefl.elevator.points.length;
    console.assert(upDefl < upZero, "elevator sign test failed", upZero, upDefl);
  }
  // aileron sign
  {
    const st = { pn: 0, pe: 0, pd: 0, phi: 0, theta: 0, psi: 0 };
    const w = assembleWorld(real, st, { de: 0, da: (10 * Math.PI) / 180, dr: 0 }, 0);
    const rightUp = w.aileronRight.points.reduce((s, p) => s + p[2], 0) / w.aileronRight.points.length;
    const leftUp = w.aileronLeft.points.reduce((s, p) => s + p[2], 0) / w.aileronLeft.points.length;
    console.assert(rightUp > leftUp, "aileron sign test failed", rightUp, leftUp);
  }
  console.log("geometry.js self-tests complete (see any console.assert failures above)");
}

window.UAVGeometry = {
  COSMETIC, dcmBodyToNed, bodyVecToWorld, bodyPointToWorld,
  rotateXZ, rotateXY, generateFuselage, generateWing, generateTail,
  assembleWorld, runSelfTests, tailDims,
};
