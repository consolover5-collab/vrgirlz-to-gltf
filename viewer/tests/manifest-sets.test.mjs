// `node --test viewer/tests/*.test.mjs` — the set -> pose-list logic behind
// the panel's Set buttons and Pose grid (ui.js's `_activateSet`), pure
// manifest juggling, no DOM.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { OTHER_SET_ID, scanSets, scansForSet, setIdForScan } from '../src/manifest.js';

const manifest = {
  sets: { 'vrgirlz-2016': { label: 'VRGirlz 2016' }, 'vrgirlz-2015': { label: 'VRGirlz 2015' } },
  scans: [
    { id: '09', set: 'vrgirlz-2015' },
    { id: '42', set: 'vrgirlz-2016' },
    { id: '45', set: 'vrgirlz-2016' },
    { id: '91', set: 'nowhere' }, // a set id `sets` doesn't know -> "Other"
  ],
};

test('one button per set actually present, ids sorted, Other last', () => {
  assert.deepEqual(
    scanSets(manifest).map(([id]) => id),
    ['vrgirlz-2015', 'vrgirlz-2016', OTHER_SET_ID],
  );
});

test('a set lists exactly its own poses, in manifest order', () => {
  assert.deepEqual(scansForSet(manifest, 'vrgirlz-2016').map((s) => s.id), ['42', '45']);
  assert.deepEqual(scansForSet(manifest, 'vrgirlz-2015').map((s) => s.id), ['09']);
});

test('a scan whose set is missing from `sets` lands in the Other bucket', () => {
  assert.equal(setIdForScan(manifest, manifest.scans[3]), OTHER_SET_ID);
  assert.deepEqual(scansForSet(manifest, OTHER_SET_ID).map((s) => s.id), ['91']);
});

test('every scan appears under exactly one set button', () => {
  const listed = scanSets(manifest).flatMap(([id]) => scansForSet(manifest, id).map((s) => s.id));
  assert.deepEqual(listed.sort(), manifest.scans.map((s) => s.id).sort());
});

test('an unknown set id yields no poses rather than the whole manifest', () => {
  assert.deepEqual(scansForSet(manifest, 'not-a-set'), []);
  assert.deepEqual(scansForSet(manifest, undefined), []);
});
