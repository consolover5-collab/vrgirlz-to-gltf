// Pure yaw math for controls-xr.js's VR rig placement — no THREE.js
// dependency, so it's testable with plain `node --test` (see lod-select.js/
// loader-util.js for the same split).

// Yaw (radians, THREE's `rotation.y` convention: rotating an object whose
// local forward is -Z) that makes an object standing at (fromX, fromZ) face
// (targetX, targetZ), both on the Y=0 plane.
//
// `Object3D.lookAt()` would seem like the obvious tool for this, but for a
// plain (non-camera, non-light) object it points the object's own +Z axis
// at the target (see three's Object3D.js: the non-camera branch swaps
// eye/target before calling Matrix4.lookAt) — the opposite of a camera,
// which looks down -Z. Calling `rig.lookAt(pivot)` therefore aimed the rig
// (and so the camera riding inside it) 180° away from the pivot (P1, the reviewer
// review 22.09). This computes the yaw directly instead of routing through
// that mismatched convention.
export function yawTowardsXZ(fromX, fromZ, targetX, targetZ) {
  const dx = targetX - fromX;
  const dz = targetZ - fromZ;
  return Math.atan2(-dx, -dz);
}

// The mirror image for a plane: yaw that turns an object's +Z (the face of
// a PlaneGeometry, i.e. the side the VR panel's texture is on) towards
// (targetX, targetZ).
export function yawFacingXZ(fromX, fromZ, targetX, targetZ) {
  return Math.atan2(targetX - fromX, targetZ - fromZ);
}

// Where to hang the VR control panel when the user opens it: `distance`
// metres along the camera's own forward direction, at a fixed `height`
// above the floor (never at eye level exactly — looking slightly down is
// more comfortable and keeps the model in view), turned to face the camera.
// Computed once per opening: the panel then stays put in world space, since
// a panel welded to the head is what makes people sick.
export function panelPlacement(cameraX, cameraZ, cameraYaw, distance, height) {
  const x = cameraX - Math.sin(cameraYaw) * distance;
  const z = cameraZ - Math.cos(cameraYaw) * distance;
  return { x, y: height, z, yaw: yawFacingXZ(x, z, cameraX, cameraZ) };
}
