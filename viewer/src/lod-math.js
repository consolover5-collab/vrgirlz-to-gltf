// Pure projection math for LOD selection — plain numbers and arrays in and
// out, no THREE.js, so `node --test viewer/tests/lod.test.mjs` can drive it
// without a browser or an importmap. `lod.js` is the thin THREE-facing
// wrapper; `lod-select.js` turns the screen height computed here into a
// level.
//
// Matrices are three.js's own layout: a flat 16-element array in COLUMN-major
// order (`Matrix4.elements`).

// Bounding sphere of an axis-aligned box: centre in the middle, radius half
// the diagonal. A sphere over the box (rather than the box's height, which
// is what the viewer used to project) is what makes the estimate independent
// of how the model is turned — design review 22.09, point 4.
export function boundingSphere(min, max) {
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  return { center, radius: Math.sqrt(dx * dx + dy * dy + dz * dz) / 2 };
}

export function applyMatrix(elements, point) {
  const [x, y, z] = point;
  return [
    elements[0] * x + elements[4] * y + elements[8] * z + elements[12],
    elements[1] * x + elements[5] * y + elements[9] * z + elements[13],
    elements[2] * x + elements[6] * y + elements[10] * z + elements[14],
  ];
}

// Largest amount this matrix can stretch a direction. Without shear that's
// just the longest basis vector; with shear it underestimates the true
// largest singular value slightly, which the sphere's own conservatism
// covers (design review 22.09: "Без shear достаточно максимального
// масштаба").
export function maxScale(elements) {
  const lengths = [0, 1, 2].map((column) => {
    const i = column * 4;
    return Math.hypot(elements[i], elements[i + 1], elements[i + 2]);
  });
  return Math.max(...lengths);
}

// A local-space sphere (from `boundingSphere`) placed into world space by an
// object's `matrixWorld`.
export function transformSphere(sphere, matrixWorldElements) {
  return {
    center: applyMatrix(matrixWorldElements, sphere.center),
    radius: sphere.radius * maxScale(matrixWorldElements),
  };
}

// Screen height of a sphere as a fraction of viewport height, for one
// camera: `viewElements` is its `matrixWorldInverse`, `projection11` element
// [5] of its projection matrix (`2n/(t-b)`, the only part of an asymmetric
// XR frustum that matters vertically).
//
//   size = |P[5]| * r * sqrt(z^2 + y^2 - r^2) / (z^2 - r^2),   z > r
//
// z is DEPTH along the view axis, not euclidean distance to the camera: a
// sphere pushed sideways stays the same size on screen, it only moves. A
// sphere that reaches the camera plane (z <= r) is already clipping the near
// plane and gets `Infinity`, i.e. the most detailed level allowed.
export function sphereScreenHeight(sphereWorld, viewElements, projection11) {
  const [, viewY, viewZ] = applyMatrix(viewElements, sphereWorld.center);
  const z = -viewZ; // view space looks down -Z
  const r = sphereWorld.radius;
  if (!(z > r)) return Infinity;
  const denominator = z * z - r * r;
  return (Math.abs(projection11) * r * Math.sqrt(denominator + viewY * viewY)) / denominator;
}

// The largest screen height over every camera — in XR both eyes see the
// scan through slightly different frustums, and LOD must follow whichever
// eye sees it bigger (design review 22.09: "Брать максимум по глазам").
export function maxSphereScreenHeight(sphereWorld, cameras) {
  let largest = 0;
  for (const { view, projection11 } of cameras) {
    const size = sphereScreenHeight(sphereWorld, view, projection11);
    if (size === Infinity) return Infinity;
    if (size > largest) largest = size;
  }
  return largest;
}
