// Our own LOD selector — not THREE.LOD, which switches by camera distance;
// Unity's `screenHeight` thresholds are a *fraction of viewport height*
// (docs/manifest.md: "projected body height / viewport height >= screenHeight"),
// which behaves correctly under perspective/FOV changes and per-eye XR
// viewports, where distance-based switching would not.

import * as THREE from 'three';

import { DEFAULT_HYSTERESIS, clampLevel, selectLodLevel } from './lod-select.js';
import { boundingSphere, maxSphereScreenHeight, transformSphere } from './lod-math.js';

export { boundingSphere };

// Scratch view matrices — at most one flat camera or two XR eyes per frame,
// so nothing is allocated in the render loop.
const _viewMatrices = [new THREE.Matrix4(), new THREE.Matrix4()];

export class LodSelector {
  // lods: ascending array of {level, screenHeight} (level 0 = finest).
  // tierMaxLod: the current tier's floor — never select anything finer
  // (docs/manifest.md tiers[tier].maxLod: "how many trailing LOD levels a
  // weaker tier drops"; concretely a clamp on the *smallest* level number).
  // Level selection itself is `./lod-select.js`'s pure `selectLodLevel` —
  // this class only owns the mutable `currentLevel` app.js's render loop
  // relies on.
  constructor(lods, tierMaxLod, hysteresis = DEFAULT_HYSTERESIS) {
    this.lods = [...lods].sort((a, b) => a.level - b.level);
    this.tierMaxLod = tierMaxLod;
    this.hysteresis = hysteresis;
    this.currentLevel = null;
  }

  setTierMaxLod(tierMaxLod) {
    this.tierMaxLod = tierMaxLod;
    if (this.currentLevel !== null) {
      // Re-clamp immediately (not just on the next `select()`) so a tier
      // switch never leaves `currentLevel` pointing at a level the new
      // tier doesn't provide.
      this.currentLevel = clampLevel(this.currentLevel, this.lods, tierMaxLod);
    }
  }

  // projectedHeight: projected size of the body sphere / viewport height
  // (dimensionless), already tier-agnostic — `projectedSphereHeight` below
  // computes it per frame.
  select(projectedHeight) {
    this.currentLevel = selectLodLevel({
      lods: this.lods, tierMaxLod: this.tierMaxLod, hysteresis: this.hysteresis,
      currentLevel: this.currentLevel, projectedHeight,
    });
    return this.currentLevel;
  }
}

// One `{view, projection11}` per camera that will actually render this
// frame: the flat camera, or one entry per XR eye.
//
// `eye.fov` is NOT reliable in XR (three.js's own doc comment on
// WebXRManager.getCamera: "does not reflect the fov of the XR camera...
// compute manually from the projection matrix") — an eye's projection is
// usually an asymmetric frustum, so the vertical term is read straight out
// of element [5]. Likewise WebXRManager updates `eye.matrixWorld` without
// touching `eye.position`, so view matrices are inverted from the world
// matrix rather than rebuilt from position/quaternion.
export function cameraViews(renderer, flatCamera) {
  const cameras = renderer.xr.isPresenting
    ? (renderer.xr.getCamera().cameras.length ? renderer.xr.getCamera().cameras : [renderer.xr.getCamera()])
    : [flatCamera];
  return cameras.slice(0, _viewMatrices.length).map((camera, index) => ({
    view: _viewMatrices[index].copy(camera.matrixWorld).invert().elements,
    projection11: camera.projectionMatrix.elements[5],
  }));
}

// Screen height (fraction of viewport height) of a scan's body bounding
// sphere, ready for `LodSelector.select`. `localSphere` comes from
// `boundingSphere(scanEntry.bodyBounds.min, ..max)`; `objectMatrixWorld` is
// the scan group's — the sphere follows the model when it's grabbed,
// scaled or turned, which projecting a fixed `heightM` at a euclidean
// camera distance never did (design review 22.09, point 4).
export function projectedSphereHeight(renderer, flatCamera, localSphere, objectMatrixWorld) {
  const world = transformSphere(localSphere, objectMatrixWorld.elements);
  return maxSphereScreenHeight(world, cameraViews(renderer, flatCamera));
}
