// Stand-in for src/loader.js in scan.test.mjs (swapped in by a resolve
// hook there): every load returns a Promise the test settles by hand, so
// it decides which of two in-flight loads lands first.

import * as THREE from 'three';

export const pending = []; // {kind, tier, level, wet, materialSet, resolve, reject}
export const builtSets = [];

export class MaterialSet {
  static build(scanEntry, tier) {
    const set = new MaterialSet(tier);
    builtSets.push(set);
    return set;
  }

  constructor(tier) {
    this.tier = tier;
    this.disposed = false;
  }

  hasWet(name) { return name === 'body'; }
  get(name, wet) { return { name, wet, tier: this.tier }; }
  dispose() { this.disposed = true; }
}

function load(kind, tier, level, materialSet, wet, partIds) {
  return new Promise((resolve, reject) => {
    const settle = () => {
      const group = new THREE.Group();
      group.userData = { kind, tier, level, disposed: false };
      for (const id of partIds) {
        const mesh = new THREE.Mesh();
        mesh.name = id;
        mesh.userData.partId = id;
        mesh.material = materialSet.get(id, wet);
        group.add(mesh);
      }
      resolve(group);
    };
    pending.push({ kind, tier, level, wet, materialSet, resolve: settle, reject });
  });
}

export function loadStatic(basePath, scanEntry, tier, materialSet, wet) {
  return load('static', tier, null, materialSet, wet, scanEntry.parts.filter((p) => !p.lod).map((p) => p.id));
}

export function loadLod(basePath, scanEntry, tier, level, materialSet, wet) {
  return load('lod', tier, level, materialSet, wet, scanEntry.parts.filter((p) => p.lod).map((p) => p.id));
}

export function disposeGroup(group) {
  group.userData.disposed = true;
}
