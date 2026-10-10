// `node --test viewer/tests/*.test.mjs` — ui.js's two layouts of the one
// panel (design review 22.09 point 3): the XR `‹ set ›` / `‹ pose n/N ›`
// steppers, driven through the same callbacks and state as the flat
// buttons. Node has no DOM, so the few element features ui.js touches are
// stubbed below — just enough to build the panel and click its buttons.

import { test } from 'node:test';
import assert from 'node:assert/strict';

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.style = {};
    this.disabled = false;
    this._text = '';
    this._classes = new Set();
    this._listeners = new Map();
    this._attributes = new Map();
    const classes = this._classes;
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !classes.has(name) : force;
        if (on) classes.add(name);
        else classes.delete(name);
        return on;
      },
    };
  }

  get className() { return [...this._classes].join(' '); }
  set className(value) {
    this._classes.clear();
    for (const name of value.split(/\s+/).filter(Boolean)) this._classes.add(name);
  }

  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  set textContent(value) { this.children = []; this._text = String(value); }

  set innerHTML(value) { assert.equal(value, ''); this.children = []; this._text = ''; }

  appendChild(child) {
    child.remove();
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  remove() {
    if (!this.parentElement) return;
    const siblings = this.parentElement.children;
    siblings.splice(siblings.indexOf(this), 1);
    this.parentElement = null;
  }

  setAttribute(name, value) { this._attributes.set(name, String(value)); }
  getAttribute(name) { return this._attributes.get(name) ?? null; }

  addEventListener(type, listener) {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type).push(listener);
  }

  // Like HTMLElement.click(): nothing happens on a disabled button.
  click() {
    if (this.disabled) return;
    for (const listener of this._listeners.get('click') ?? []) listener({ type: 'click', target: this });
  }
}

globalThis.document = { createElement: (tag) => new FakeElement(tag) };

const { Ui } = await import('../src/ui.js');

function scan(id, set) {
  return { id, set, tiers: ['2k', '4k'], parts: [], materials: {} };
}

const manifest = {
  tiers: { '2k': { label: '2K' }, '4k': { label: '4K' } },
  sets: { alpha: { label: 'Alpha' }, beta: { label: 'Beta' } },
  scans: [
    ...Array.from({ length: 10 }, (_, i) => scan(`a${i + 1}`, 'alpha')),
    // A set-prefixed id, like every real non-Nat-21 dataset's — the label
    // shows only its pose part.
    scan('beta-b1', 'beta'),
    ...Array.from({ length: 3 }, (_, i) => scan(`b${i + 2}`, 'beta')),
  ],
};

// A Ui plus a fake app: onScanChange does what app.js's setScan does
// (ui.setScan with the new entry), and every callback call is recorded.
function makeUi() {
  const calls = [];
  const root = new FakeElement('div');
  const ui = new Ui(root, manifest, {
    onScanChange: (id) => {
      calls.push(['scan', id]);
      ui.setScan(manifest.scans.find((s) => s.id === id), '2k');
    },
    onTierChange: (tier) => calls.push(['tier', tier]),
    onWetChange: () => calls.push(['wet']),
    onPartVisibleChange: () => {},
    onExitVr: () => calls.push(['exit']),
    onLayoutChange: () => calls.push(['layout']),
  });
  ui.setScan(manifest.scans[0], '2k');
  return { ui, calls };
}

// The `‹`/`›` buttons of the stepper whose label is `label`.
function arrows(label) {
  const row = label.parentElement;
  return { prev: row.children[0], next: row.children[2] };
}

const scanCalls = (calls) => calls.filter(([kind]) => kind === 'scan').map(([, id]) => id);

test('XR layout: pose stepper shows the shared selection as pose n/N', () => {
  const { ui } = makeUi();
  ui.setPresentation('xr');
  assert.ok(ui.panel.classList.contains('vrscan-xr'));
  assert.equal(ui.xrSetLabel.textContent, 'Alpha');
  assert.equal(ui.xrPoseLabel.textContent, 'a1  1/10');
});

test('one arrow click = one onScanChange, wrapping at the first and last pose', () => {
  const { ui, calls } = makeUi();
  ui.setPresentation('xr');
  const { prev, next } = arrows(ui.xrPoseLabel);

  prev.click(); // first -> last
  assert.deepEqual(scanCalls(calls), ['a10']);
  assert.equal(ui.xrPoseLabel.textContent, 'a10  10/10');

  next.click(); // last -> first
  assert.deepEqual(scanCalls(calls), ['a10', 'a1']);
  next.click();
  assert.deepEqual(scanCalls(calls), ['a10', 'a1', 'a2']);
  assert.equal(ui.xrPoseLabel.textContent, 'a2  2/10');
});

test('switching from a 10-pose set to a 4-pose set picks its first pose and counts /4', () => {
  const { ui, calls } = makeUi();
  ui.setPresentation('xr');
  arrows(ui.xrSetLabel).next.click();
  assert.deepEqual(scanCalls(calls), ['beta-b1']);
  assert.equal(ui.xrSetLabel.textContent, 'Beta');
  assert.equal(ui.xrPoseLabel.textContent, 'b1  1/4');
  arrows(ui.xrPoseLabel).prev.click();
  assert.equal(ui.xrPoseLabel.textContent, 'b4  4/4');
  // The flat grid was rebuilt for the new set behind the XR layout too.
  assert.deepEqual([...ui._poseButtons.keys()], ['beta-b1', 'b2', 'b3', 'b4']);
});

test('flat -> XR -> flat keeps the selection made in the headset', () => {
  const { ui } = makeUi();
  assert.equal(ui.exitBtn.style.display, 'none');
  ui.setPresentation('xr');
  assert.equal(ui.exitBtn.style.display, '');
  arrows(ui.xrSetLabel).next.click();
  arrows(ui.xrPoseLabel).next.click();
  ui.setPresentation('flat');

  assert.ok(!ui.panel.classList.contains('vrscan-xr'));
  assert.equal(ui.exitBtn.style.display, 'none');
  assert.equal(ui.activeSet, 'beta');
  assert.equal(ui.currentScanEntry.id, 'b2');
  assert.ok(ui._setButtons.get('beta').classList.contains('active'));
  assert.ok(ui._poseButtons.get('b2').classList.contains('active'));
  assert.ok(!ui._poseButtons.get('beta-b1').classList.contains('active'));
});

test('EXIT VR stays in the XR layout and fires onExitVr once per click', () => {
  const { ui, calls } = makeUi();
  ui.setPresentation('xr');
  arrows(ui.xrSetLabel).next.click();
  assert.equal(ui.exitBtn.parentElement, ui.panel);
  assert.equal(ui.exitBtn.style.display, '');
  ui.exitBtn.click();
  assert.deepEqual(calls.filter(([kind]) => kind === 'exit'), [['exit']]);
});

test('the set and pose grids are flat-only, the steppers XR-only', () => {
  const { ui } = makeUi();
  assert.ok(ui.setsRow.parentElement.classList.contains('vrscan-flat-only'));
  assert.ok(ui.posesGrid.parentElement.classList.contains('vrscan-flat-only'));
  assert.ok(ui.xrSetLabel.parentElement.parentElement.classList.contains('vrscan-xr-only'));
  assert.ok(ui.xrPoseLabel.parentElement.parentElement.classList.contains('vrscan-xr-only'));
});

test('layout changes are reported so the VR panel can re-measure', () => {
  const { ui, calls } = makeUi();
  calls.length = 0;
  ui.setPresentation('xr');
  assert.ok(calls.some(([kind]) => kind === 'layout'));
  calls.length = 0;
  arrows(ui.xrPoseLabel).next.click();
  assert.ok(calls.some(([kind]) => kind === 'layout'));
});
