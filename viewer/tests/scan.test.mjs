// `node --test viewer/tests/*.test.mjs` — ScanInstance's async loads under
// races (design review 22.09 B2): init overtaken by a tier change, two quick
// tier changes, dispose mid-load, a late LOD after a wet toggle. Loads are
// ./fake-loader.mjs Promises settled by hand; `three` is the vendored copy
// the page's importmap points at.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const VENDOR_THREE = new URL('../vendor/three/build/three.module.js', import.meta.url).href;
const FAKE_LOADER = new URL('./fake-loader.mjs', import.meta.url).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'three') return { url: VENDOR_THREE, shortCircuit: true };
    if (specifier === './loader.js' && context.parentURL?.endsWith('/src/scan.js')) {
      return { url: FAKE_LOADER, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const { ScanInstance } = await import('../src/scan.js');
const fake = await import('./fake-loader.mjs');

const SCAN = {
  id: '91',
  pivot: [0, 0, 0],
  parts: [
    { id: 'body', material: 'body', lod: true },
    { id: 'sofa', material: 'sofa', lod: false },
  ],
  lods: [
    { level: 0, screenHeight: 0.5, gltf: 'lod0.gltf' },
    { level: 1, screenHeight: 0.1, gltf: 'lod1.gltf' },
  ],
  bodyBounds: { min: [0, 0, 0], max: [1, 1, 1] },
};
const TIERS = { '1k': { maxLod: 1 }, '2k': { maxLod: 0 }, '4k': { maxLod: 0 } };

function makeScan(tier = '2k') {
  fake.pending.length = 0;
  fake.builtSets.length = 0;
  return new ScanInstance(SCAN, tier, '', TIERS, null);
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

// Resolves the one pending load matching `where` and lets its continuation run.
async function land(where) {
  const i = fake.pending.findIndex((p) => Object.entries(where).every(([k, v]) => p[k] === v));
  assert.notEqual(i, -1, `no pending load ${JSON.stringify(where)}`);
  const [load] = fake.pending.splice(i, 1);
  load.resolve();
  await settle();
}

// Everything in the scene is exactly the current static group plus the
// current LOD groups, all live — no stale group, no second body.
function assertSceneIsCurrent(scan) {
  const expected = [scan.staticGroup, ...scan.lodGroups.values()].filter(Boolean);
  assert.deepEqual(new Set(scan.group.children), new Set(expected));
  for (const group of scan.group.children) {
    assert.equal(group.userData.disposed, false);
    assert.equal(group.userData.tier, scan.tier);
  }
  const bodies = scan.group.children.filter((g) => g.userData.kind === 'lod');
  assert.equal(new Set(bodies.map((g) => g.userData.level)).size, bodies.length);
}

test('a tier change during init: the older static lands late and is freed, off-scene', async () => {
  const scan = makeScan('2k');
  const init = scan.init();
  const setTier = scan.setTier('4k');
  await land({ kind: 'static', tier: '4k' });
  await land({ kind: 'lod', tier: '4k', level: 1 });
  await setTier;

  await land({ kind: 'static', tier: '2k' });
  await init;
  assert.equal(scan.staticGroup.userData.tier, '4k');
  assert.deepEqual(fake.pending, []); // the stale init started no LOD load
  assertSceneIsCurrent(scan);
  assert.equal(fake.builtSets.find((s) => s.tier === '2k').disposed, true);
  assert.equal(scan.materialSet.disposed, false);
});

test('a tier change during init: the older static landing first never reaches the scene', async () => {
  const scan = makeScan('2k');
  const init = scan.init();
  const setTier = scan.setTier('4k');
  await land({ kind: 'static', tier: '2k' });
  await init;
  assert.deepEqual(scan.group.children, []);

  await land({ kind: 'static', tier: '4k' });
  await land({ kind: 'lod', tier: '4k', level: 1 });
  await setTier;
  assertSceneIsCurrent(scan);
  assert.equal(scan.lodGroups.size, 1);
});

for (const order of [['4k', '1k'], ['1k', '4k']]) {
  test(`two quick tier changes, statics landing ${order.join(' then ')}: only the last tier is on screen`, async () => {
    const scan = makeScan('2k');
    const init = scan.init();
    await land({ kind: 'static', tier: '2k' });
    await land({ kind: 'lod', tier: '2k', level: 1 });
    await land({ kind: 'lod', tier: '2k', level: 0 });
    await init;
    const oldGroups = [...scan.group.children];
    assert.equal(oldGroups.length, 3);

    const first = scan.setTier('4k');
    const second = scan.setTier('1k');
    for (const tier of order) await land({ kind: 'static', tier });
    await land({ kind: 'lod', tier: '1k', level: 1 });
    await Promise.all([first, second]);

    assert.equal(scan.tier, '1k');
    assertSceneIsCurrent(scan);
    assert.equal(scan.lodGroups.size, 1);
    assert.ok(oldGroups.every((g) => g.userData.disposed && !scan.group.children.includes(g)));
    const sets = Object.fromEntries(fake.builtSets.map((s) => [s.tier, s]));
    assert.equal(sets['2k'].disposed, true);
    assert.equal(sets['4k'].disposed, true);
    assert.equal(sets['1k'].disposed, false);
    assert.deepEqual(fake.pending, []); // the superseded 4k never asked for a LOD
  });
}

test('dispose while the static load is in flight: the result is freed and nothing follows', async () => {
  const scan = makeScan('2k');
  const init = scan.init();
  scan.dispose();
  await land({ kind: 'static', tier: '2k' });
  await init;
  assert.deepEqual(scan.group.children, []);
  assert.deepEqual(fake.pending, []);
  assert.equal(scan.materialSet.disposed, true);
});

test('dispose while a LOD load is in flight: the late LOD never joins the scene', async () => {
  const scan = makeScan('2k');
  const init = scan.init();
  await land({ kind: 'static', tier: '2k' });
  scan.dispose();
  await land({ kind: 'lod', level: 1 });
  await init;
  assert.deepEqual(scan.group.children.map((g) => g.userData.kind), ['static']);
  assert.equal(scan.lodGroups.size, 0);
  assert.deepEqual(fake.pending, []);
});

test('a LOD requested dry that lands after the wet toggle shows wet', async () => {
  const scan = makeScan('2k');
  const init = scan.init();
  await land({ kind: 'static', tier: '2k' });
  assert.equal(fake.pending[0].wet, false);
  scan.setWet(true);
  await land({ kind: 'lod', level: 1 });
  const body = scan.lodGroups.get(1).getObjectByName('body');
  assert.deepEqual(body.material, { name: 'body', wet: true, tier: '2k' });
  await land({ kind: 'lod', level: 0 });
  await init;
  assertSceneIsCurrent(scan);
});

test('a LOD of the old tier landing after a tier change is freed, not shown', async () => {
  const scan = makeScan('2k');
  const init = scan.init();
  await land({ kind: 'static', tier: '2k' });
  const setTier = scan.setTier('4k');
  await land({ kind: 'lod', tier: '2k', level: 1 });
  await init;
  assert.equal(scan.lodGroups.size, 0);
  await land({ kind: 'static', tier: '4k' });
  await land({ kind: 'lod', tier: '4k', level: 1 });
  await setTier;
  assertSceneIsCurrent(scan);
});
