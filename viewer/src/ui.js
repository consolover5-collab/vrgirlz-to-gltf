// The control panel — one DOM tree, two modes. On a flat screen it's the
// overlay you click with mouse/finger; in a headset vr-panel.js renders
// this very element into a texture and feeds the controller ray back in as
// DOM clicks, so there is exactly one set of widgets and one state. That's
// why everything here is a <button> (never a <select>, never a checkbox
// inside a <label>) and why nothing relies on CSS HTMLMesh can't paint —
// see index.html's style comment.
//
// app.js wires the callbacks to ScanInstance/manifest state and feeds live
// stats every frame; the stats block deliberately sits OUTSIDE the panel
// element, since HTMLMesh re-rasterises the whole panel on every DOM
// mutation and per-frame stats would do that 60 times a second.

import { scanSets, scansForSet, setIdForScan } from './manifest.js';
import {
  categoryFor, collectSceneCategories, forcedHiddenParts, resolveSceneToggles, visibilityMapForScan,
} from './part-toggles.js';

const SCENE_TOGGLES_STORAGE_KEY = 'vrscan.sceneToggles';

export class Ui {
  constructor(root, manifest, callbacks) {
    this.manifest = manifest;
    this.callbacks = callbacks;
    this.root = root;
    this.currentTier = null;
    // [[setId, setEntry], ...], one per dataset actually present among the
    // manifest's scans — see manifest.js's scanSets doc comment for order.
    this.sets = scanSets(manifest);
    this.activeSet = null;

    // Site-wide "Scene" toggles (part-toggles.js) — computed once from the
    // WHOLE manifest, not the current scan, and persisted across reloads;
    // switching scans must not add/remove/reset a row (owner: "чтобы это
    // касалось всего сайта, а не отдельных моделек на нем").
    this.sceneCategories = collectSceneCategories(manifest);
    this.sceneToggles = resolveSceneToggles(this.sceneCategories, this._loadPersistedToggles());
    // A part forced hidden in the CURRENT scan by its profile (`hidden:
    // true`) that the "show anyway" button overrode — session-only, reset
    // on every scan switch, never persisted (a one-off look, not a setting).
    this.showAnywayPartIds = new Set();
    this.currentScanEntry = null;
    this.wet = false;

    this._build();
  }

  _build() {
    this.root.innerHTML = '';

    // "☰" lives outside `.vrscan-body` so it stays reachable when the panel
    // is slid away (owner: the panel must not cover the model).
    this.collapseBtn = document.createElement('button');
    this.collapseBtn.type = 'button';
    this.collapseBtn.className = 'vrscan-collapse';
    this.collapseBtn.textContent = '☰';
    this.collapseBtn.title = 'Show/hide controls';
    this.collapseBtn.addEventListener('click', () => this.setCollapsed(!this.collapsed));
    this.collapsed = false;
    this.root.appendChild(this.collapseBtn);

    const body = document.createElement('div');
    body.className = 'vrscan-body';
    this.root.appendChild(body);

    // The element vr-panel.js rasterises. Everything the user has to reach
    // from inside the headset goes in here; anything else stays a sibling.
    this.panel = document.createElement('div');
    this.panel.className = 'vrscan-panel';
    body.appendChild(this.panel);

    this.setsRow = this._section('Set', 'vrscan-btn-row');
    this.setsRow.parentElement.classList.add('vrscan-flat-only');
    this._setButtons = new Map();
    for (const [setId, setEntry] of this.sets) {
      const btn = this._button(setEntry?.label ?? setId, () => this._activateSet(setId, { pickFirstScan: true }));
      this.setsRow.appendChild(btn);
      this._setButtons.set(setId, btn);
    }
    // A single dataset needs no picker — one button that does nothing would
    // just be noise (dev-sample, or a manifest converted from only one
    // dataset so far).
    if (this.sets.length <= 1) this.setsRow.parentElement.remove();

    this.posesGrid = this._section('Pose', 'vrscan-grid');
    this.posesGrid.parentElement.classList.add('vrscan-flat-only');
    this._poseButtons = new Map();

    // The XR presentation of the same two choices (design review 22.09 point
    // 3): `‹ set ›` and `‹ pose n/N ›` instead of a button per set and per
    // pose, so ten poses don't shrink the whole panel in the headset. The
    // arrows go through the very callbacks the flat buttons use, and what
    // they show is read from `activeSet`/`currentScanEntry` — no VR copy of
    // the state to drift.
    const xrSetRow = this._stepper('Set', () => this._stepSet(-1), () => this._stepSet(1));
    this.xrSetLabel = xrSetRow.label;
    this._xrSetSection = xrSetRow.section;
    if (this.sets.length <= 1) xrSetRow.section.remove();
    const xrPoseRow = this._stepper('Pose', () => this._stepPose(-1), () => this._stepPose(1));
    this.xrPoseLabel = xrPoseRow.label;
    this.presentation = 'flat';

    this.tiersRow = this._section('Quality', 'vrscan-btn-row');
    this._tierButtons = new Map();

    this.sceneBox = this._section('Scene', 'vrscan-scene');
    // Named for what it is: the wet look is a roughness approximation built
    // from WET_SPC, not the original runtime shader (design review 22.09).
    this.wetSwitch = this._switch('Wet sheen ≈', () => this.callbacks.onWetChange(!this.wet));
    this.wetSwitch.title = 'Wet sheen — approximation from the WET_SPC map, not the original shader';
    this.sceneBox.appendChild(this.wetSwitch);
    this._buildSceneToggles();

    this.statusEl = document.createElement('div');
    this.statusEl.className = 'vrscan-status';
    this.panel.appendChild(this.statusEl);

    // The way out of a session — in flat mode entering VR is VRButton's job
    // (below, outside the panel), so this one only exists in the XR layout
    // (`setPresentation`).
    this.exitBtn = document.createElement('button');
    this.exitBtn.type = 'button';
    this.exitBtn.className = 'vrscan-exit';
    this.exitBtn.textContent = 'EXIT VR';
    this.exitBtn.style.display = 'none';
    this.exitBtn.addEventListener('click', () => this.callbacks.onExitVr?.());
    this.panel.appendChild(this.exitBtn);

    this.vrSlot = document.createElement('div');
    this.vrSlot.className = 'vrscan-vr-slot';
    body.appendChild(this.vrSlot);

    this.statsEl = document.createElement('pre');
    this.statsEl.className = 'vrscan-stats';
    body.appendChild(this.statsEl);
  }

  // <section title> + the container its controls go into.
  _section(title, className) {
    const section = document.createElement('div');
    section.className = 'vrscan-section';
    const heading = document.createElement('div');
    heading.className = 'vrscan-section-title';
    heading.textContent = title;
    const box = document.createElement('div');
    box.className = className;
    section.appendChild(heading);
    section.appendChild(box);
    this.panel.appendChild(section);
    return box;
  }

  // An XR-only section: `‹` label `›`. Returns its parts for the caller.
  _stepper(title, onPrev, onNext) {
    const row = this._section(title, 'vrscan-stepper');
    const section = row.parentElement;
    section.classList.add('vrscan-xr-only');
    const prev = this._button('‹', onPrev);
    const label = document.createElement('div');
    label.className = 'vrscan-stepper-label';
    const next = this._button('›', onNext);
    row.appendChild(prev);
    row.appendChild(label);
    row.appendChild(next);
    return { section, label, prev, next };
  }

  // Neighbouring set/pose, wrapping at the ends — first ‹ goes to the last.
  _stepSet(delta) {
    const ids = this.sets.map(([id]) => id);
    const index = ids.indexOf(this.activeSet);
    if (index < 0 || ids.length < 2) return;
    this._activateSet(ids[(index + delta + ids.length) % ids.length], { pickFirstScan: true });
  }

  _stepPose(delta) {
    const scans = scansForSet(this.manifest, this.activeSet);
    const index = scans.findIndex((scan) => scan.id === this.currentScanEntry?.id);
    if (index < 0 || scans.length < 2) return;
    this.callbacks.onScanChange(scans[(index + delta + scans.length) % scans.length].id);
  }

  _syncXrLabels() {
    const setEntry = this.sets.find(([id]) => id === this.activeSet)?.[1];
    this.xrSetLabel.textContent = setEntry?.label ?? this.activeSet ?? '';
    const scans = scansForSet(this.manifest, this.activeSet);
    const index = scans.findIndex((scan) => scan.id === this.currentScanEntry?.id);
    // The set is on the row above, so a prefixed id ("vrgirlz-09-ariel-79")
    // only needs its pose part here.
    const id = scans[index]?.id ?? '';
    const poseId = id.startsWith(`${this.activeSet}-`) ? id.slice(this.activeSet.length + 1) : id;
    this.xrPoseLabel.textContent = index < 0 ? '—' : `${poseId}  ${index + 1}/${scans.length}`;
  }

  // 'flat' | 'xr' — which layout of the one panel is showing. Only the
  // layout changes: the selection lives in `activeSet`/`currentScanEntry`
  // either way, and both layouts are kept up to date, so leaving XR shows
  // the flat grid on whatever was picked inside the headset. EXIT VR is
  // part of the XR layout — the headset has no other way out.
  setPresentation(mode) {
    this.presentation = mode;
    this.panel.classList.toggle('vrscan-xr', mode === 'xr');
    this.exitBtn.style.display = mode === 'xr' ? '' : 'none';
    this._syncXrLabels();
    this.callbacks.onLayoutChange?.();
  }

  _button(label, onClick) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'vrscan-btn';
    btn.textContent = label;
    btn.addEventListener('click', onClick);
    return btn;
  }

  // A switch is a button with `aria-checked` (not a checkbox): one element
  // under the VR ray, one click, and the on/off state is a colour the
  // HTMLMesh rasteriser actually paints.
  _switch(label, onClick) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'vrscan-switch';
    btn.setAttribute('role', 'switch');
    btn.setAttribute('aria-checked', 'false');
    const text = document.createElement('span');
    text.className = 'vrscan-switch-label';
    text.textContent = label;
    const track = document.createElement('span');
    track.className = 'vrscan-switch-track';
    const knob = document.createElement('span');
    knob.className = 'vrscan-switch-knob';
    track.appendChild(knob);
    btn.appendChild(text);
    btn.appendChild(track);
    btn.addEventListener('click', () => {
      if (!btn.disabled) onClick();
    });
    return btn;
  }

  _setSwitch(btn, on) {
    btn.setAttribute('aria-checked', on ? 'true' : 'false');
  }

  setCollapsed(collapsed) {
    this.collapsed = collapsed;
    this.root.classList.toggle('vrscan-collapsed', collapsed);
  }

  // Built once (site-wide, see constructor) — one switch per category, plus
  // a per-scan "hidden in this pose / show anyway" button that `setScan`
  // shows/hides without rebuilding the row.
  _buildSceneToggles() {
    this._categoryNotes = new Map(); // category id -> note <button>
    for (const category of this.sceneCategories) {
      const toggle = this._switch(category.label, () => {
        this.sceneToggles[category.id] = !this.sceneToggles[category.id];
        this._setSwitch(toggle, this.sceneToggles[category.id]);
        this._savePersistedToggles();
        this._syncVisibility();
      });
      this._setSwitch(toggle, this.sceneToggles[category.id]);
      this.sceneBox.appendChild(toggle);

      const note = this._button('hidden in this pose — show anyway', () => this._showAnyway(category.id));
      note.className = 'vrscan-note-btn';
      note.style.display = 'none';
      this._categoryNotes.set(category.id, note);
      this.sceneBox.appendChild(note);
    }
  }

  _loadPersistedToggles() {
    try {
      const raw = localStorage.getItem(SCENE_TOGGLES_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null; // storage disabled/unavailable, or a corrupt value — resolveSceneToggles falls back to defaults
    }
  }

  _savePersistedToggles() {
    try {
      localStorage.setItem(SCENE_TOGGLES_STORAGE_KEY, JSON.stringify(this.sceneToggles));
    } catch {
      // private browsing / storage full/disabled — the toggle itself must still work this session
    }
  }

  // A part this scan's profile forces hidden (`hidden: true`) just got its
  // "show anyway" button clicked — only ever affects `this.currentScanEntry`,
  // never persisted.
  _showAnyway(categoryId) {
    if (!this.currentScanEntry) return;
    for (const part of forcedHiddenParts(this.currentScanEntry)) {
      if (categoryFor(part.kind).id === categoryId) this.showAnywayPartIds.add(part.id);
    }
    const note = this._categoryNotes.get(categoryId);
    if (note) note.style.display = 'none';
    this._syncVisibility();
    this.callbacks.onLayoutChange?.();
  }

  // Recomputes every part's visibility for the current scan and pushes it
  // through the existing `onPartVisibleChange` callback (app.js forwards it
  // to `ScanInstance.setPartVisible`) — used after a Scene toggle or "show
  // anyway" changes state for an already-loaded scan.
  _syncVisibility() {
    if (!this.currentScanEntry) return;
    for (const [partId, visible] of this.visibilityMapFor(this.currentScanEntry)) {
      this.callbacks.onPartVisibleChange(partId, visible);
    }
  }

  // app.js calls this BEFORE constructing a new `ScanInstance`, once
  // `setScan` has updated `currentScanEntry`/`showAnywayPartIds` for the
  // scan being switched to — gives `ScanInstance` a visibility map that
  // already reflects the Scene toggles instead of just `!hidden`.
  visibilityMapFor(scanEntry) {
    return visibilityMapForScan(scanEntry, this.sceneToggles, this.showAnywayPartIds);
  }

  attachVrButton(buttonEl) {
    this.vrSlot.appendChild(buttonEl);
  }

  // Switches the active set: highlights its button and rebuilds the pose
  // grid to only list its poses. `pickFirstScan` (set button clicks only)
  // additionally jumps to that set's first pose via the normal onScanChange
  // callback — setScan (driven by app.js after the scan actually changed)
  // never does, to avoid a loop.
  _activateSet(setId, { pickFirstScan = false } = {}) {
    this.activeSet = setId;
    for (const [id, btn] of this._setButtons) btn.classList.toggle('active', id === setId);

    this.posesGrid.innerHTML = '';
    this._poseButtons = new Map();
    const scans = scansForSet(this.manifest, setId);
    for (const scan of scans) {
      // No thumbnails in the manifest yet (docs/manifest.md) — the pose id
      // is all we can put on the button.
      const btn = this._button(scan.id, () => this.callbacks.onScanChange(scan.id));
      this.posesGrid.appendChild(btn);
      this._poseButtons.set(scan.id, btn);
    }
    this._writeUrlSet(setId);
    this._syncXrLabels();
    if (pickFirstScan && scans[0]) this.callbacks.onScanChange(scans[0].id);
  }

  // ?set=<id> — only meaningful (and only written) once there's more than
  // one set to choose between.
  _writeUrlSet(setId) {
    try {
      const url = new URL(location.href);
      if (this.sets.length > 1) url.searchParams.set('set', setId);
      else url.searchParams.delete('set');
      history.replaceState(null, '', url);
    } catch {
      // URL/history unavailable (e.g. sandboxed iframe) — set just won't persist across reloads.
    }
  }

  // Called whenever the active scan or tier changes — rebuilds the parts
  // of the UI that depend on which scan/tier is selected.
  setScan(scanEntry, tier) {
    this.currentTier = tier;
    const setId = setIdForScan(this.manifest, scanEntry);
    if (setId !== this.activeSet) this._activateSet(setId);
    for (const [id, btn] of this._poseButtons) btn.classList.toggle('active', id === scanEntry.id);

    this.tiersRow.innerHTML = '';
    this._tierButtons = new Map();
    for (const tierId of scanEntry.tiers) {
      const btn = this._button(tierId, () => this.callbacks.onTierChange(tierId));
      btn.title = this.manifest.tiers[tierId]?.label ?? tierId;
      btn.classList.toggle('active', tierId === tier);
      this.tiersRow.appendChild(btn);
      this._tierButtons.set(tierId, btn);
    }

    const hasWet = scanEntry.parts.some((p) => scanEntry.materials[p.material]?.variants?.wet);
    this.wetSwitch.disabled = !hasWet;
    // A new scan is loaded dry (ScanInstance starts with `wet = false`).
    this.wet = false;
    this._setSwitch(this.wetSwitch, false);

    // The Scene panel itself is built once (site-wide) — only its per-scan
    // "hidden in this pose" affordance and the "show anyway" session state
    // reset here.
    this.currentScanEntry = scanEntry;
    this.showAnywayPartIds = new Set();
    const forcedCategoryIds = new Set(forcedHiddenParts(scanEntry).map((p) => categoryFor(p.kind).id));
    for (const [categoryId, note] of this._categoryNotes) {
      note.style.display = forcedCategoryIds.has(categoryId) ? '' : 'none';
    }
    this._syncXrLabels();
    // Tier count and "hidden in this pose" notes can change the panel's
    // height — the VR panel has to re-measure it (vr-panel.js).
    this.callbacks.onLayoutChange?.();
  }

  // app.js echoes the wet state back here after ScanInstance took it, so
  // the switch reflects the scene rather than the click.
  setWet(wet) {
    this.wet = wet;
    this._setSwitch(this.wetSwitch, wet);
  }

  // Shown while a set/pose/tier switch is loading its LOD chain — the same
  // line the VR panel shows, since it's the same element.
  setLoading(what) {
    this.statusEl.textContent = what ? `loading ${what}…` : '';
  }

  // stats: { fps, lodLevel, rendererInfo (renderer.info), scanEntry, tier }
  updateStats(stats) {
    const budget = stats.scanEntry.budget?.[stats.tier];
    const gpuBytes = budget
      ? budget.gpuBytesTextures
      : Object.values(stats.scanEntry.textures[stats.tier] || {}).reduce((sum, t) => sum + t.gpuBytes, 0);
    const info = stats.rendererInfo;
    this.statsEl.textContent = [
      `FPS: ${stats.fps.toFixed(1)}`,
      `LOD: ${stats.lodLevel ?? '—'}`,
      `draw calls: ${info.render.calls}  triangles: ${info.render.triangles}`,
      `geometries: ${info.memory.geometries}  textures: ${info.memory.textures}`,
      `Σ texture bytes (${stats.tier}): ${(gpuBytes / 1024 / 1024).toFixed(1)} MiB`,
    ].join('\n');
  }
}
