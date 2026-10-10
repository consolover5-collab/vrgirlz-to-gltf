// Fetch + validate manifest.json, and resolve which quality tier to use:
// URL ?tier= -> localStorage -> quality.js's hardware guess.
// See docs/manifest.md for the full shape; this module doesn't validate
// every field (that's the converter's `manifest.py` job) — just the parts
// the viewer touches before anything else can run.

import { pickTier } from './quality.js';

const SCHEMA = 'vrscan-manifest/1';
const TIER_STORAGE_KEY = 'vrscan.tier';

export async function loadManifest(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`manifest fetch failed: ${res.status} ${res.statusText} (${url})`);
  }
  const data = await res.json();
  if (data.schema !== SCHEMA) {
    throw new Error(`unsupported manifest schema "${data.schema}", expected "${SCHEMA}"`);
  }
  if (!Array.isArray(data.scans) || data.scans.length === 0) {
    throw new Error('manifest has no scans');
  }
  return data;
}

// URL param -> localStorage -> quality.js guess. Whatever wins is saved back
// to localStorage so a manual override (via ?tier=) sticks across reloads.
export function resolveTier(manifest, urlTier) {
  let tier = null;
  if (urlTier && manifest.tiers[urlTier]) {
    tier = urlTier;
  } else {
    const stored = safeGet(TIER_STORAGE_KEY);
    if (stored && manifest.tiers[stored]) {
      tier = stored;
    }
  }
  if (!tier) {
    tier = pickTier(manifest);
  }
  safeSet(TIER_STORAGE_KEY, tier);
  return tier;
}

export function setTier(tier) {
  safeSet(TIER_STORAGE_KEY, tier);
}

export function findScan(manifest, scanId) {
  if (scanId) {
    const found = manifest.scans.find((s) => s.id === scanId);
    if (found) return found;
  }
  return manifest.scans[0];
}

// Synthetic bucket for scans with no `set` (or a `set` that doesn't match
// anything in `manifest.sets`) — old manifests, or a scan whose dataset was
// removed from `sets` by hand. See docs/manifest.md.
export const OTHER_SET_ID = '__other__';

export function setIdForScan(manifest, scan) {
  if (scan?.set && manifest.sets?.[scan.set]) return scan.set;
  return OTHER_SET_ID;
}

// Ordered [id, setEntry] pairs for every set actually present among
// `manifest.scans` — one button per dataset (docs/manifest.md). Sorted by
// id rather than by `sets` object/insertion order so button order is
// deterministic regardless of conversion history (VRGirlz profile ids
// happen to sort chronologically: "vrgirlz-2015" before "vrgirlz-2016").
// `OTHER_SET_ID`, if present, always sorts last.
export function scanSets(manifest) {
  const present = new Set(manifest.scans.map((s) => setIdForScan(manifest, s)));
  const ids = [...present].filter((id) => id !== OTHER_SET_ID).sort();
  const result = ids.map((id) => [id, manifest.sets[id]]);
  if (present.has(OTHER_SET_ID)) result.push([OTHER_SET_ID, { label: 'Other' }]);
  return result;
}

// Every scan belonging to `setId`, in manifest order — one pose button per
// entry (ui.js). A set id nothing matches (or `undefined`) yields an empty
// list rather than the whole manifest.
export function scansForSet(manifest, setId) {
  return manifest.scans.filter((s) => setIdForScan(manifest, s) === setId);
}

// First scan belonging to `setId` (from URL ?set=), or null when `setId` is
// falsy or matches no scan — caller falls back to `findScan`'s default.
export function findScanInSet(manifest, setId) {
  if (!setId) return null;
  return manifest.scans.find((s) => setIdForScan(manifest, s) === setId) || null;
}

// Every geometry filename in the manifest carries a literal "{tier}"
// placeholder (docs/manifest.md: "static.{tier}.gltf") because the same
// .bin is shared across tiers — only the baked-in material/image refs
// inside the per-tier .gltf differ.
export function tierFilename(template, tier) {
  return template.replace('{tier}', tier);
}

function safeGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private browsing / storage disabled — tier just won't persist.
  }
}
