// Pure logic behind the viewer's "Scene" panel — site-wide, per-category
// visibility toggles (persisted across scans/sets/reloads, owner: "чтобы
// это касалось всего сайта, а не отдельных моделек на нем"), layered under
// a per-scan "hidden in this pose" profile override (`part.hidden`, e.g.
// Nat-90's eyelashes — see profiles.py's `hidden_parts`) with a session-
// only, non-persisted "show anyway" escape hatch. Split out of ui.js so
// it's testable with plain `node --test` (see lod-select.js/loader-util.js/
// xr-math.js for the same split) — no DOM/localStorage here, ui.js owns
// those.

// kind -> { id, label } for the Scene panel's rows. An unlisted kind (a
// future part type) still gets a row, auto-labeled "Props: <kind>" rather
// than being silently un-toggleable.
const KNOWN_CATEGORIES = {
  shadow: { id: 'shadow', label: 'Shadow' },
  sofa: { id: 'furniture', label: 'Furniture' },
  feet: { id: 'furniture', label: 'Furniture' },
  eyelashes: { id: 'eyelashes', label: 'Eyelashes' },
  // Eye spheres are a rough shader approximation, not scene dressing — the
  // toggle is named for what it visually stands in for, not the part kind
  // (design review 22.09 point 6: "включение назвать «блик роговицы»").
  eyes: { id: 'eyes', label: 'Corneal highlight' },
};

// Categories that start OFF when nothing's been persisted yet.
export const DEFAULT_OFF_CATEGORIES = new Set(['eyes']);

export function categoryFor(kind) {
  return KNOWN_CATEGORIES[kind] ?? { id: `props:${kind}`, label: `Props: ${kind}` };
}

// Every distinct category across the WHOLE manifest (every scan, every
// set), in label order — computed once at startup. Site-wide on purpose:
// switching scans/sets must not add, remove, or reset a toggle row. Only
// `hideable` parts contribute a category — e.g. "body" never is one, with
// no kind-name special-casing needed here.
export function collectSceneCategories(manifest) {
  const seen = new Map();
  for (const scan of manifest.scans) {
    for (const part of scan.parts) {
      if (!part.hideable) continue;
      const category = categoryFor(part.kind);
      if (!seen.has(category.id)) seen.set(category.id, category);
    }
  }
  return [...seen.values()].sort((a, b) => a.label.localeCompare(b.label));
}

// A (possibly partial/stale/corrupt-after-JSON.parse) persisted toggle map
// -> one explicit boolean per CURRENT category. `persisted` is whatever
// `JSON.parse(localStorage.getItem(...))` produced, or `null`/`undefined`
// on first run or a parse failure — catching that is ui.js's job, not this
// function's; this only has to tolerate the resulting value being junk.
export function resolveSceneToggles(categories, persisted) {
  const toggles = {};
  for (const category of categories) {
    const stored = persisted && typeof persisted === 'object' ? persisted[category.id] : undefined;
    toggles[category.id] = typeof stored === 'boolean' ? stored : !DEFAULT_OFF_CATEGORIES.has(category.id);
  }
  return toggles;
}

// Whether ONE part should currently be visible. Never gated for a part
// that isn't hideable at all (e.g. body). Otherwise: the category's global
// toggle first, then the profile's per-scan override (`part.hidden`)
// unless this exact part id is in `showAnywayPartIds`.
export function isPartVisible(part, globalToggles, showAnywayPartIds) {
  if (!part.hideable) return true;
  const category = categoryFor(part.kind);
  const categoryOn = globalToggles[category.id] ?? !DEFAULT_OFF_CATEGORIES.has(category.id);
  if (!categoryOn) return false;
  if (part.hidden && !(showAnywayPartIds && showAnywayPartIds.has(part.id))) return false;
  return true;
}

// `isPartVisible` applied to every part of one scan -> Map(partId ->
// visible), the exact shape `ScanInstance.partVisibility` needs (scan.js)
// — both for its initial state and to recompute after a toggle changes.
export function visibilityMapForScan(scanEntry, globalToggles, showAnywayPartIds) {
  return new Map(scanEntry.parts.map((p) => [p.id, isPartVisible(p, globalToggles, showAnywayPartIds)]));
}

// Which of a scan's parts are forced hidden in THIS pose regardless of the
// category toggle (`part.hidden`, profile-driven) — what ui.js shows the
// "hidden in this pose / show anyway" affordance for.
export function forcedHiddenParts(scanEntry) {
  return scanEntry.parts.filter((p) => p.hideable && p.hidden);
}
