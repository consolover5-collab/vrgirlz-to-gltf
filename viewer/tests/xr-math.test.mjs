// `node --test viewer/tests/` — pure trig, no THREE.js/browser needed (see
// xr-math.js's module docstring for why the yaw fix lives here).

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { panelPlacement, yawTowardsXZ } from '../src/xr-math.js';

// Rebuilds THREE's rotation.y convention independently (Matrix4.makeRotationY,
// applied to local forward (0,0,-1)) so the round-trip check below doesn't
// just re-assert xr-math.js's own formula against itself.
function forwardFromYaw(yaw) {
  return { x: -Math.sin(yaw), z: -Math.cos(yaw) };
}

test('pivot straight ahead (default rig forward already correct) needs no yaw', () => {
  const yaw = yawTowardsXZ(0, 1.5, 0, 0); // rig 1.5m in front of the pivot along +Z
  assert.ok(Math.abs(yaw) < 1e-9, `expected ~0, got ${yaw}`);
});

test('pivot to the rig\'s left (-X) needs a +90 degree yaw', () => {
  const yaw = yawTowardsXZ(0, 0, -1, 0);
  assert.ok(Math.abs(yaw - Math.PI / 2) < 1e-9, `expected PI/2, got ${yaw}`);
});

test('pivot to the rig\'s right (+X) needs a -90 degree yaw', () => {
  const yaw = yawTowardsXZ(0, 0, 1, 0);
  assert.ok(Math.abs(yaw + Math.PI / 2) < 1e-9, `expected -PI/2, got ${yaw}`);
});

test('pivot behind the rig (+Z) needs a 180 degree yaw', () => {
  const yaw = yawTowardsXZ(0, 0, 0, 1);
  assert.ok(Math.abs(Math.abs(yaw) - Math.PI) < 1e-9, `expected +-PI, got ${yaw}`);
});

test('the resulting yaw always makes the camera-forward direction point at the target', () => {
  const cases = [
    [0, 1.5, 0, 0],
    [2, -3, -1, 4],
    [-1.5, 0, 0.3, -2.7],
    [5, 5, -5, -5],
    [0, 0, 1, 1],
  ];
  for (const [fromX, fromZ, targetX, targetZ] of cases) {
    const yaw = yawTowardsXZ(fromX, fromZ, targetX, targetZ);
    const forward = forwardFromYaw(yaw);
    const dx = targetX - fromX;
    const dz = targetZ - fromZ;
    const len = Math.hypot(dx, dz);
    const dot = forward.x * (dx / len) + forward.z * (dz / len);
    assert.ok(dot > 1 - 1e-9, `case ${JSON.stringify([fromX, fromZ, targetX, targetZ])}: dot=${dot}`);
  }
});

// VR panel placement (vr-panel.js's `show`): in front of the user, at a
// fixed height, turned to face them.
test('the panel lands `distance` metres along the camera forward direction', () => {
  const place = panelPlacement(0, 0, 0, 0.7, 1.2); // facing -Z
  assert.ok(Math.abs(place.x) < 1e-9, `expected x ~0, got ${place.x}`);
  assert.ok(Math.abs(place.z + 0.7) < 1e-9, `expected z ~-0.7, got ${place.z}`);
  assert.equal(place.y, 1.2);
});

test('panel height is the fixed comfort height, never the head height', () => {
  for (const yaw of [0, 1, -2.5, Math.PI]) {
    assert.equal(panelPlacement(3, -4, yaw, 0.8, 1.2).y, 1.2);
  }
});

test('the panel face (+Z, PlaneGeometry front) always points back at the camera', () => {
  const cases = [
    [0, 0, 0],
    [1.5, -2, Math.PI / 2],
    [-3, 4, -1.1],
    [0.2, 0.3, Math.PI],
  ];
  for (const [camX, camZ, camYaw] of cases) {
    const place = panelPlacement(camX, camZ, camYaw, 0.7, 1.2);
    // +Z of an object rotated by `yaw` around Y (Matrix4.makeRotationY).
    const faceX = Math.sin(place.yaw);
    const faceZ = Math.cos(place.yaw);
    const dx = camX - place.x;
    const dz = camZ - place.z;
    const len = Math.hypot(dx, dz);
    const dot = faceX * (dx / len) + faceZ * (dz / len);
    assert.ok(dot > 1 - 1e-9, `case ${JSON.stringify([camX, camZ, camYaw])}: dot=${dot}`);
  }
});

test('an open panel is always exactly `distance` away from the camera', () => {
  for (const yaw of [0, 0.4, 2.2, -3]) {
    const place = panelPlacement(1, 2, yaw, 0.65, 1.2);
    assert.ok(Math.abs(Math.hypot(place.x - 1, place.z - 2) - 0.65) < 1e-9);
  }
});
