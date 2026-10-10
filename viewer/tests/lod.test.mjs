// `node --test viewer/tests/` — pure function, no THREE.js/browser needed
// (see viewer/src/lod-select.js's module docstring for why it's split out).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { selectLodLevel, clampLevel } from '../src/lod-select.js';
import {
  boundingSphere,
  maxSphereScreenHeight,
  sphereScreenHeight,
  transformSphere,
} from '../src/lod-math.js';

// Same numbers as the design review's own example:
// thresholds 0.125/0.25, 15% hysteresis. LOD1→LOD0 must not fire at 0.251 —
// it must wait for 0.25 * 1.15 = 0.2875.
const TWO_LEVEL_LODS = [
  { level: 0, screenHeight: 0.25 },
  { level: 1, screenHeight: 0.125 },
];

test('first pick (no currentLevel) has no hysteresis, just clamp', () => {
  const level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 0, currentLevel: null, projectedHeight: 0.05 });
  assert.equal(level, 1); // below every threshold — coarsest fallback
});

test('upgrade waits for the TARGET level\'s own threshold * (1 + hysteresis), not the current level\'s', () => {
  // Just past LOD0's raw threshold (0.25) but short of 0.2875 — must NOT
  // upgrade yet. The pre-fix code checked the *current* (LOD1) level's
  // threshold (0.125 * 1.15 = 0.14375), which 0.26 already clears, so it
  // upgraded immediately; that was the bug.
  let level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 0, currentLevel: 1, projectedHeight: 0.26 });
  assert.equal(level, 1, 'must not upgrade before clearing the target level\'s own hysteresis band');

  level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 0, currentLevel: 1, projectedHeight: 0.2875 });
  assert.equal(level, 0, 'upgrades once at/past target threshold * (1 + hysteresis)');
});

test('downgrade waits for the CURRENT level\'s own threshold * (1 - hysteresis)', () => {
  // 0.25 * (1 - 0.15) = 0.2125.
  let level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 0, currentLevel: 0, projectedHeight: 0.22 });
  assert.equal(level, 0, 'must not downgrade above the hysteresis band yet');

  level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 0, currentLevel: 0, projectedHeight: 0.21 });
  assert.equal(level, 1, 'downgrades once strictly below current threshold * (1 - hysteresis)');
});

test('round trip at one boundary never chatters between the two thresholds', () => {
  let level = 1;
  // Camera drifting back and forth across the LOD0/LOD1 boundary; only the
  // two "past the band" samples should actually switch.
  const walk = [
    [0.20, 1], // well below — stays coarse
    [0.26, 1], // past raw threshold, short of upgrade band — must not chatter
    [0.29, 0], // past upgrade band — switches
    [0.22, 0], // past raw threshold, short of downgrade band — must not chatter back
    [0.20, 1], // past downgrade band — switches back
    [0.26, 1], // short of upgrade band again — stays coarse (no repeat-flip bug)
    [0.30, 0], // past upgrade band again — switches again
  ];
  for (const [projectedHeight, expected] of walk) {
    level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 0, currentLevel: level, projectedHeight });
    assert.equal(level, expected, `at projectedHeight=${projectedHeight}`);
  }
});

// Three levels — a second, independent boundary (LOD1/LOD2) so "every
// boundary", not just one, gets a back-and-forth check.
const THREE_LEVEL_LODS = [
  { level: 0, screenHeight: 0.5 },
  { level: 1, screenHeight: 0.25 },
  { level: 2, screenHeight: 0.1 },
];

test('round trip at the LOD1/LOD2 boundary', () => {
  let level = 2;
  const walk = [
    [0.26, 2], // past raw LOD1 threshold (0.25), short of upgrade band (0.2875) — no switch
    [0.29, 1], // past upgrade band — switches
    [0.22, 1], // past raw threshold, short of downgrade band (0.2125) — no switch
    [0.21, 2], // past downgrade band — switches back
  ];
  for (const [projectedHeight, expected] of walk) {
    level = selectLodLevel({ lods: THREE_LEVEL_LODS, tierMaxLod: 0, currentLevel: level, projectedHeight });
    assert.equal(level, expected, `at projectedHeight=${projectedHeight}`);
  }
});

test('a jump straight past two boundaries at once still lands on the correct target level', () => {
  // From LOD2, a huge projectedHeight should upgrade directly to LOD0 once
  // past LOD0's own (highest) threshold * (1 + hysteresis), not get stuck
  // requiring two separate frames.
  const level = selectLodLevel({ lods: THREE_LEVEL_LODS, tierMaxLod: 0, currentLevel: 2, projectedHeight: 1.0 });
  assert.equal(level, 0);
});

test('tierMaxLod clamps the floor even on the first pick', () => {
  const level = selectLodLevel({ lods: TWO_LEVEL_LODS, tierMaxLod: 1, currentLevel: null, projectedHeight: 5.0 });
  assert.equal(level, 1, 'tier never shows anything finer than its cap, even though LOD0 would otherwise be desired');
});

test('clamp snaps to the nearest existing level, not a tierMaxLod that has no file', () => {
  // This scan only has LOD0 and LOD2 (LOD1 missing — e.g. a profile/scan
  // that skips a level). tierMaxLod=1 alone (Math.max(level, 1)) would
  // "clamp" to 1, a level this scan doesn't actually have.
  const gappyLods = [
    { level: 0, screenHeight: 0.5 },
    { level: 2, screenHeight: 0.1 },
  ];
  const level = selectLodLevel({ lods: gappyLods, tierMaxLod: 1, currentLevel: null, projectedHeight: 0.05 });
  assert.equal(level, 2, 'falls back to the nearest existing level at/past the floor');
});

test('clampLevel (used when a tier switch re-clamps an already-current level) also snaps to an existing level', () => {
  const gappyLods = [
    { level: 0, screenHeight: 0.5 },
    { level: 2, screenHeight: 0.1 },
  ];
  assert.equal(clampLevel(0, gappyLods, 1), 2);
  assert.equal(clampLevel(2, gappyLods, 0), 2);
});

// --- sphere projection (the reviewer geometry review 22.09, test 5) --------------
// `lod-math.js` is pure, so the whole camera side is testable here too:
// depth vs. distance, scale, model rotation, near plane, two eyes.


const P11 = 1 / Math.tan((60 * Math.PI) / 180 / 2); // 60 deg vertical FOV

// Column-major view matrix of a camera at `position` looking down -Z.
function viewAt([x, y, z]) {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -x, -y, -z, 1];
}

// Column-major model matrix: uniform scale then translation.
function modelMatrix({ scale = 1, position = [0, 0, 0] } = {}) {
  return [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, ...position, 1];
}

// Column-major rotation about Y, to turn a (non-spherical) model.
function rotationY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
}

const BODY = boundingSphere([-0.22, 0, -0.22], [0.22, 1.6, 0.22]); // dev-sample bodyBounds

test('sideways movement at the same depth does not change the screen size', () => {
  const sphere = transformSphere(BODY, modelMatrix());
  const centred = sphereScreenHeight(sphere, viewAt([0, 0.8, 4]), P11);
  const offset = sphereScreenHeight(sphere, viewAt([3, 0.8, 4]), P11);
  assert.ok(Math.abs(centred - offset) < 1e-12, `${centred} vs ${offset}`);
  // ... while the euclidean distance the old code used differs a lot:
  assert.ok(Math.hypot(3, 4) > 4 * 1.2);
});

test('doubling the model scale roughly doubles the screen size', () => {
  const view = viewAt([0, 0.8, 8]);
  const small = sphereScreenHeight(transformSphere(BODY, modelMatrix({ scale: 1 })), view, P11);
  const big = sphereScreenHeight(transformSphere(BODY, modelMatrix({ scale: 2 })), view, P11);
  assert.ok(big / small > 1.9 && big / small < 2.1, `ratio ${big / small}`);
});

test('turning the model does not change the screen size (a sphere has no facing)', () => {
  const view = viewAt([0, 0.8, 4]);
  const straight = sphereScreenHeight(transformSphere(BODY, modelMatrix()), view, P11);
  const turned = sphereScreenHeight(transformSphere(BODY, rotationY(Math.PI / 3)), view, P11);
  assert.ok(Math.abs(straight - turned) < 1e-9, `${straight} vs ${turned}`);
});

test('a sphere reaching the camera plane asks for the most detailed level', () => {
  const sphere = transformSphere(BODY, modelMatrix());
  const size = sphereScreenHeight(sphere, viewAt([0, 0.8, sphere.radius * 0.5]), P11);
  assert.equal(size, Infinity);
  const level = selectLodLevel({ lods: THREE_LEVEL_LODS, tierMaxLod: 0, currentLevel: 2, projectedHeight: size });
  assert.equal(level, 0);
});

test('with two eyes the bigger projection wins, and a near-plane hit wins outright', () => {
  const sphere = transformSphere(BODY, modelMatrix());
  const far = { view: viewAt([-0.03, 0.8, 6]), projection11: P11 };
  const near = { view: viewAt([0.03, 0.8, 3]), projection11: P11 };
  const both = maxSphereScreenHeight(sphere, [far, near]);
  assert.equal(both, sphereScreenHeight(sphere, near.view, P11));
  assert.ok(both > sphereScreenHeight(sphere, far.view, P11));

  const clipping = { view: viewAt([0, 0.8, 0.1]), projection11: P11 };
  assert.equal(maxSphereScreenHeight(sphere, [far, clipping]), Infinity);
});

test('the dev-sample at its default view still selects LOD0, as it did with heightM', () => {
  // app.js: camera at (0, 1.6, 2.5) looking at the pivot (0, 0.9, 0),
  // 60 deg FOV. Old formula: heightM / distance / (2 tan(fov/2)).
  const heightM = 1.6;
  const distance = Math.hypot(1.6 - 0.9, 2.5);
  const oldProjected = heightM / distance / (2 * Math.tan((60 * Math.PI) / 180 / 2));
  const lods = [
    { level: 0, screenHeight: 0.25 },
    { level: 1, screenHeight: 0.125 },
    { level: 2, screenHeight: 0.062 },
  ];
  // The camera looks slightly down at the body; depth along the view axis
  // is what the sphere formula uses.
  const sphere = transformSphere(BODY, modelMatrix());
  const toPivot = [0 - 0, 0.9 - 1.6, 0 - 2.5];
  const forward = toPivot.map((c) => c / Math.hypot(...toPivot));
  const toCenter = [sphere.center[0] - 0, sphere.center[1] - 1.6, sphere.center[2] - 2.5];
  const depth = toCenter.reduce((sum, c, i) => sum + c * forward[i], 0);
  const lateral = Math.sqrt(Math.max(0, toCenter.reduce((s, c) => s + c * c, 0) - depth * depth));
  const newProjected = sphereScreenHeight({ center: [0, lateral, -depth], radius: sphere.radius },
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], P11);

  assert.equal(selectLodLevel({ lods, tierMaxLod: 0, currentLevel: null, projectedHeight: oldProjected }), 0);
  assert.equal(selectLodLevel({ lods, tierMaxLod: 0, currentLevel: null, projectedHeight: newProjected }), 0);
  // the sphere is a little more generous than the body height, but not wildly
  assert.ok(newProjected / oldProjected > 1 && newProjected / oldProjected < 1.3,
    `ratio ${newProjected / oldProjected}`);
});
