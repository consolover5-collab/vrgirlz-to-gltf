// Ф1 auto-preset: navigator.xr + UA sniffing for "this is a Quest 2" is the
// only signal we have without an active XR session (deviceMemory/GPU string
// detection is Ф2 — see the plan's "Вьюер (Ф2)" note). Manual override in
// ui.js always wins; this is only the first guess before the user touches
// anything.

export function pickTier(manifest) {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : '';
  const hasXr = typeof navigator !== 'undefined' && 'xr' in navigator;
  const isQuest2 = hasXr && /Quest 2\b/i.test(ua);

  const wanted = isQuest2 ? '2k' : '4k';
  if (manifest.tiers[wanted]) return wanted;

  // Requested tier isn't in this package (e.g. sample only ships one) —
  // fall back to whatever tier is listed first.
  return Object.keys(manifest.tiers)[0];
}
