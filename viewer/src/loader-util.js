// Pure helper for loader.js's MaterialSet — split out (no THREE.js import)
// so it's testable with plain `node --test`, same reasoning as
// lod-select.js.

// Compares two "role -> texture key" maps — `materials[name].textures` and
// its `variants.wet.textures` merged over it (docs/manifest.md) — by the
// manifest's own texture key strings, not pixels. True when every role
// resolves to the same texture key in both, meaning a "wet" material built
// from `wetTextures` would be byte-for-byte the material `dryTextures`
// already builds, so there's no point loading/keeping a second one.
export function sameTextureRefs(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}
