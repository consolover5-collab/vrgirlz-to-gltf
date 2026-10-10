import { test } from 'node:test';
import assert from 'node:assert/strict';

import { sameTextureRefs } from '../src/loader-util.js';

test('sameTextureRefs: true when every role resolves to the same texture key', () => {
  const dry = { baseColor: 'body_CLR', normal: 'body_NM', orm: 'body_ORM' };
  const wet = { baseColor: 'body_CLR', normal: 'body_NM', orm: 'body_ORM' };
  assert.equal(sameTextureRefs(dry, wet), true);
});

test('sameTextureRefs: false when a role actually differs (the normal case — wet ORM override)', () => {
  const dry = { baseColor: 'body_CLR', normal: 'body_NM', orm: 'body_ORM' };
  const wet = { baseColor: 'body_CLR', normal: 'body_NM', orm: 'body_ORM_wet' };
  assert.equal(sameTextureRefs(dry, wet), false);
});

test('sameTextureRefs: false when one side has an extra role the other lacks', () => {
  const dry = { baseColor: 'body_CLR' };
  const wet = { baseColor: 'body_CLR', orm: 'body_ORM_wet' };
  assert.equal(sameTextureRefs(dry, wet), false);
});

test('sameTextureRefs: true for two empty maps', () => {
  assert.equal(sameTextureRefs({}, {}), true);
});
