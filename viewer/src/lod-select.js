// Pure LOD level selection — no THREE.js dependency, so it's testable with
// plain `node --test` (viewer/tests/lod.test.mjs) without resolving the
// `three` importmap specifier. `lod.js`'s `LodSelector` wraps this with the
// mutable `currentLevel` state app.js drives every frame; this module only
// ever computes "given where the camera is now and where we were, where
// should we be" — no THREE objects in or out.

export const DEFAULT_HYSTERESIS = 0.15;

// lods: array of {level, screenHeight} (level 0 = finest); any order.
// tierMaxLod: the current tier's floor — never select anything finer.
// currentLevel: the level currently shown, or null/undefined for the first
// pick (no hysteresis — just the raw threshold lookup, clamped).
// projectedHeight: bodyBounds height / viewport height (dimensionless).
// Returns the level that should be shown now.
export function selectLodLevel({ lods, tierMaxLod, hysteresis = DEFAULT_HYSTERESIS, currentLevel, projectedHeight }) {
  const sorted = [...lods].sort((a, b) => a.level - b.level);
  const desired = clampToExistingLevel(levelFor(sorted, projectedHeight), sorted, tierMaxLod);
  if (currentLevel === null || currentLevel === undefined) return desired;
  if (desired === currentLevel) return currentLevel;

  const currentLod = sorted.find((l) => l.level === currentLevel);
  if (!currentLod) return desired; // current level isn't valid anymore (e.g. tier just changed) — jump straight there

  if (desired < currentLevel) {
    // Wanting finer detail — require comfortably past the TARGET (more
    // detailed) level's own entry threshold, not the current (coarser)
    // level's threshold: using the current level's threshold made the
    // hysteresis band a no-op, since `desired < currentLevel` already
    // implies `projectedHeight` crossed the finer level's own (higher)
    // threshold (design review 22.09).
    const targetLod = sorted.find((l) => l.level === desired);
    return projectedHeight >= targetLod.screenHeight * (1 + hysteresis) ? desired : currentLevel;
  }
  // Wanting coarser detail — leave the current level once comfortably below
  // its own entry threshold (unchanged: this direction was already correct).
  return projectedHeight < currentLod.screenHeight * (1 - hysteresis) ? desired : currentLevel;
}

function levelFor(sortedLods, projectedHeight) {
  for (const lod of sortedLods) {
    if (projectedHeight >= lod.screenHeight) return lod.level;
  }
  // Smaller than every threshold — fall back to the coarsest level.
  return sortedLods[sortedLods.length - 1].level;
}

// `Math.max(level, tierMaxLod)` alone can land on a level number that has
// no actual entry in `lods` — different scans/profiles can have different
// LOD counts even under the same tier (docs/manifest.md), so a tier's
// generic floor doesn't guarantee a file exists at exactly that number for
// THIS scan. Snap to the nearest existing level at or coarser than the
// floor; if none exists (floor beyond the coarsest level this scan has),
// fall back to the coarsest level that does.
function clampToExistingLevel(level, sortedLods, tierMaxLod) {
  const floor = Math.max(level, tierMaxLod);
  const existingLevels = sortedLods.map((l) => l.level);
  const candidates = existingLevels.filter((l) => l >= floor);
  return candidates.length ? Math.min(...candidates) : Math.max(...existingLevels);
}

// Exposed separately (not just via `selectLodLevel`) for `LodSelector.
// setTierMaxLod`, which needs to re-clamp an already-current level against
// a *new* tier floor without pretending it's a fresh, hysteresis-free pick.
export function clampLevel(level, lods, tierMaxLod) {
  return clampToExistingLevel(level, [...lods].sort((a, b) => a.level - b.level), tierMaxLod);
}
