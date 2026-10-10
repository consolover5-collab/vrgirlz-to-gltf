// MaterialSet: one THREE material per (scan, tier, materialName[, "wet"
// variant]), built once and reused across every LOD/static file — the
// baked-in materials inside the .gltf files are self-sufficient for
// Blender/validators but the viewer ignores them (docs/manifest.md).
//
// Also owns the actual GLTFLoader calls: static loads once, lod<K> loads on
// demand with a generation counter so a stale in-flight request (level
// changed again before the fetch resolved) is discarded instead of
// clobbering a newer one.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { tierFilename } from './manifest.js';
import { sameTextureRefs } from './loader-util.js';

const textureLoader = new THREE.TextureLoader();
const gltfLoader = new GLTFLoader();

export class MaterialSet {
  constructor() {
    this.base = new Map(); // materialName -> THREE.Material
    this.wet = new Map(); // materialName -> THREE.Material (only if a wet variant exists)
    this._textures = []; // every THREE.Texture we created, for dispose()
  }

  static build(scanEntry, tier, baseUrl) {
    const set = new MaterialSet();
    const textureDefs = scanEntry.textures[tier] || {};
    // Keyed by (uri, colorSpace): base and wet materials share most of
    // their texture refs (only e.g. ORM differs), and without this a
    // second `loadTex()` call for the same file created a second
    // THREE.Texture — every dry/wet pair doubled the GPU memory of its
    // shared maps (P1, design review 22.09).
    const textureCache = new Map();
    const loadTex = (key, colorSpace) => {
      const rec = textureDefs[key];
      if (!rec) return null;
      const uri = baseUrl + rec.file;
      const cacheKey = `${uri}|${colorSpace}`;
      let tex = textureCache.get(cacheKey);
      if (!tex) {
        tex = textureLoader.load(uri);
        tex.colorSpace = colorSpace;
        tex.flipY = false; // glTF UVs already address the image directly (docs/manifest.md).
        textureCache.set(cacheKey, tex);
        set._textures.push(tex);
      }
      return tex;
    };

    for (const [name, def] of Object.entries(scanEntry.materials)) {
      set.base.set(name, buildThreeMaterial(def, def.textures, loadTex));
      const wetOverride = def.variants && def.variants.wet;
      if (wetOverride) {
        const wetTextures = { ...def.textures, ...wetOverride.textures };
        // Skip building (and keeping) a wet material that would end up
        // referencing the exact same texture keys as dry — a wet-capable
        // recipe whose wet textures happen to match dry's for this scan
        // shouldn't cost a second material + duplicate `hasWet()` toggle.
        if (!sameTextureRefs(wetTextures, def.textures)) {
          set.wet.set(name, buildThreeMaterial(def, wetTextures, loadTex));
        }
      }
    }
    return set;
  }

  get(materialName, wet) {
    if (wet && this.wet.has(materialName)) return this.wet.get(materialName);
    return this.base.get(materialName);
  }

  hasWet(materialName) {
    return this.wet.has(materialName);
  }

  dispose() {
    for (const tex of this._textures) tex.dispose();
    for (const mat of this.base.values()) mat.dispose();
    for (const mat of this.wet.values()) mat.dispose();
  }
}

function buildThreeMaterial(def, textureRefs, loadTex) {
  const baseColor = textureRefs.baseColor ? loadTex(textureRefs.baseColor, THREE.SRGBColorSpace) : null;
  const alphaMode = def.alphaMode || 'OPAQUE';
  // baseColorFactor is linear RGBA (glTF); it multiplies the texture, and its
  // alpha is the material opacity for BLEND.
  const bcf = def.baseColorFactor || [1, 1, 1, 1];
  const common = {
    map: baseColor,
    color: new THREE.Color().setRGB(bcf[0], bcf[1], bcf[2], THREE.LinearSRGBColorSpace),
    opacity: bcf[3],
    transparent: alphaMode === 'BLEND',
    alphaTest: alphaMode === 'MASK' ? (def.alphaCutoff ?? 0.5) : 0,
    depthWrite: alphaMode !== 'BLEND',
    side: def.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
    // "additive" (eyes cornea): a black translucent layer whose only visible
    // contribution is the environment highlight — added on top, never darkens.
    blending: alphaMode === 'BLEND' && def.blend === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending,
  };

  if (def.type === 'unlit') {
    return new THREE.MeshBasicMaterial(common);
  }

  const normalMap = textureRefs.normal ? loadTex(textureRefs.normal, THREE.NoColorSpace) : null;
  const ormMap = textureRefs.orm ? loadTex(textureRefs.orm, THREE.NoColorSpace) : null;
  return new THREE.MeshStandardMaterial({
    ...common,
    normalMap,
    // ORM packing: R=occlusion, G=roughness, B=metalness — one texture
    // feeds all three three.js material slots (docs/manifest.md).
    aoMap: ormMap,
    roughnessMap: ormMap,
    metalnessMap: ormMap,
    roughness: def.roughnessFactor ?? 1,
    // Always 0, not `ormMap ? 1 : 0`: the export sets metallicFactor=0 (the
    // body/sofa are never metallic) and packs ORM's B channel as flat 0,
    // but JPEG (q85) can introduce a slightly non-zero B at block edges —
    // relying on metalnessMap alone to land back on ~0 isn't robust
    // (P1, design review 22.09). `metalnessMap` stays assigned so the ORM
    // texture keeps feeding roughness/AO; multiplying by a 0 factor makes
    // the map's own B value irrelevant for metalness either way.
    metalness: 0,
  });
}

// --------------------------------------------------------------------------
// GLTF loading
// --------------------------------------------------------------------------

// Loads a *.gltf, applies MaterialSet materials to every node whose name
// matches a part id, tags meshes with userData.partId, and returns the
// loaded THREE.Group. `wet` selects which material variant to apply.
async function loadGltfGroup(url, parts, materialSet, wet) {
  const gltf = await gltfLoader.loadAsync(url);
  const group = gltf.scene;
  const partByName = new Map(parts.map((p) => [p.id, p]));
  group.traverse((node) => {
    if (!node.isMesh) return;
    const part = partByName.get(node.name);
    if (!part) return;
    node.userData.partId = part.id;
    node.material = materialSet.get(part.material, wet);
    // Ф1 has no TANGENT attribute (docs/manifest.md: "extras.vrscan.tangents
    // = none"); three.js's normal map shader derives a screen-space tangent
    // frame from UV derivatives automatically when USE_TANGENT is absent —
    // no extra flag needed here.
  });
  return group;
}

export async function loadStatic(basePath, scanEntry, tier, materialSet, wet) {
  const url = basePath + tierFilename(scanEntry.static.gltf, tier);
  const parts = scanEntry.parts.filter((p) => !p.lod);
  return loadGltfGroup(url, parts, materialSet, wet);
}

// Loads one LOD level's group. Callers (scan.js) are responsible for the
// generation/cancellation bookkeeping and for disposing groups they evict.
export async function loadLod(basePath, scanEntry, tier, level, materialSet, wet) {
  const lodEntry = scanEntry.lods.find((l) => l.level === level);
  if (!lodEntry) throw new Error(`scan "${scanEntry.id}" has no lod level ${level}`);
  const url = basePath + tierFilename(lodEntry.gltf, tier);
  const parts = scanEntry.parts.filter((p) => p.lod);
  return loadGltfGroup(url, parts, materialSet, wet);
}

// Frees geometry (and any GLTF-baked materials/textures we didn't keep,
// though we always replace those with MaterialSet's before this runs) from
// a loaded group. MaterialSet's own materials/textures are disposed
// separately, once, when the whole scan tears down.
export function disposeGroup(group) {
  group.traverse((node) => {
    if (node.isMesh && node.geometry) node.geometry.dispose();
  });
}
