import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_OFF_CATEGORIES, categoryFor, collectSceneCategories, forcedHiddenParts, isPartVisible,
  resolveSceneToggles, visibilityMapForScan,
} from '../src/part-toggles.js';

function part(overrides) {
  return { id: 'p', kind: 'sofa', hideable: true, hidden: false, ...overrides };
}

test('categoryFor: sofa and feet share the "furniture" category', () => {
  assert.deepEqual(categoryFor('sofa'), { id: 'furniture', label: 'Furniture' });
  assert.deepEqual(categoryFor('feet'), { id: 'furniture', label: 'Furniture' });
});

test('categoryFor: eyes maps to the "Corneal highlight" label, not the raw kind', () => {
  assert.deepEqual(categoryFor('eyes'), { id: 'eyes', label: 'Corneal highlight' });
});

test('categoryFor: an unlisted kind gets an auto "Props: <kind>" row instead of being dropped', () => {
  assert.deepEqual(categoryFor('plant'), { id: 'props:plant', label: 'Props: plant' });
});

test('DEFAULT_OFF_CATEGORIES: only eyes starts off', () => {
  assert.deepEqual([...DEFAULT_OFF_CATEGORIES], ['eyes']);
});

test('collectSceneCategories: merges sofa+feet into one row, skips non-hideable parts, sorts by label', () => {
  const manifest = {
    scans: [
      {
        parts: [
          part({ id: 'body', kind: 'body', hideable: false }),
          part({ id: 'sofa-C1', kind: 'sofa' }),
          part({ id: 'feet', kind: 'feet' }),
          part({ id: 'shadow', kind: 'shadow' }),
          part({ id: 'eyeL', kind: 'eyes' }),
        ],
      },
    ],
  };
  const categories = collectSceneCategories(manifest);
  assert.deepEqual(categories.map((c) => c.id), ['eyes', 'furniture', 'shadow']);
});

test('collectSceneCategories: is the union across every scan, not just the first', () => {
  const manifest = {
    scans: [
      { parts: [part({ id: 'shadow', kind: 'shadow' })] },
      { parts: [part({ id: 'eyeL', kind: 'eyes' })] },
    ],
  };
  const categories = collectSceneCategories(manifest);
  assert.deepEqual(categories.map((c) => c.id).sort(), ['eyes', 'shadow']);
});

test('resolveSceneToggles: no persisted value -> defaults (only eyes off)', () => {
  const categories = [categoryFor('eyes'), categoryFor('shadow'), categoryFor('sofa')];
  const toggles = resolveSceneToggles(categories, null);
  assert.deepEqual(toggles, { eyes: false, shadow: true, furniture: true });
});

test('resolveSceneToggles: a persisted boolean overrides the default; junk/missing entries fall back', () => {
  const categories = [categoryFor('eyes'), categoryFor('shadow')];
  const toggles = resolveSceneToggles(categories, { eyes: true, shadow: 'nope' });
  assert.deepEqual(toggles, { eyes: true, shadow: true }); // shadow: non-boolean stored value ignored
});

test('isPartVisible: a non-hideable part is always visible, regardless of toggles', () => {
  const body = part({ kind: 'body', hideable: false, hidden: true });
  assert.equal(isPartVisible(body, {}, new Set()), true);
});

test('isPartVisible: hideable part follows its category toggle when not forced-hidden', () => {
  const shadow = part({ kind: 'shadow' });
  assert.equal(isPartVisible(shadow, { shadow: true }, new Set()), true);
  assert.equal(isPartVisible(shadow, { shadow: false }, new Set()), false);
});

test('isPartVisible: falls back to DEFAULT_OFF_CATEGORIES when the category is missing from globalToggles', () => {
  const eyeL = part({ id: 'eyeL', kind: 'eyes' });
  const shadow = part({ kind: 'shadow' });
  assert.equal(isPartVisible(eyeL, {}, new Set()), false); // eyes defaults off
  assert.equal(isPartVisible(shadow, {}, new Set()), true); // shadow defaults on
});

test('isPartVisible: part.hidden overrides an ON category toggle unless in showAnywayPartIds', () => {
  const lashes90 = part({ id: 'eyelashes', kind: 'eyelashes', hidden: true });
  assert.equal(isPartVisible(lashes90, { eyelashes: true }, new Set()), false);
  assert.equal(isPartVisible(lashes90, { eyelashes: true }, new Set(['eyelashes'])), true);
  // show-anyway for a DIFFERENT part id must not leak onto this one
  assert.equal(isPartVisible(lashes90, { eyelashes: true }, new Set(['some-other-part'])), false);
});

test('isPartVisible: category OFF wins even if showAnywayPartIds has the part', () => {
  const lashes90 = part({ id: 'eyelashes', kind: 'eyelashes', hidden: true });
  assert.equal(isPartVisible(lashes90, { eyelashes: false }, new Set(['eyelashes'])), false);
});

test('visibilityMapForScan: one entry per part, matching isPartVisible', () => {
  const scanEntry = {
    parts: [
      part({ id: 'body', kind: 'body', hideable: false }),
      part({ id: 'eyeL', kind: 'eyes' }),
      part({ id: 'eyelashes', kind: 'eyelashes', hidden: true }),
    ],
  };
  const map = visibilityMapForScan(scanEntry, { eyes: true, eyelashes: true }, new Set(['eyelashes']));
  assert.deepEqual(
    [...map.entries()],
    [['body', true], ['eyeL', true], ['eyelashes', true]],
  );
});

test('forcedHiddenParts: only hideable parts with hidden=true (e.g. Nat-90 eyelashes)', () => {
  const scanEntry = {
    parts: [
      part({ id: 'eyeL', kind: 'eyes', hidden: true }),
      part({ id: 'eyelashes', kind: 'eyelashes', hidden: false }),
      part({ id: 'body', kind: 'body', hideable: false, hidden: true }), // never hideable -> excluded
    ],
  };
  assert.deepEqual(forcedHiddenParts(scanEntry).map((p) => p.id), ['eyeL']);
});
