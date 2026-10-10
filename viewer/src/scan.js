// ScanInstance: one loaded scan (static parts + whichever LOD levels are
// currently resident) as a single THREE.Group, plus the state the UI and
// controls touch — wet toggle, hideable parts, pivot for orbit/grab.

import * as THREE from 'three';
import { MaterialSet, loadStatic, loadLod, disposeGroup } from './loader.js';
import { LodSelector, boundingSphere } from './lod.js';

const MAX_RESIDENT_LODS = 2;

export class ScanInstance {
  // tierDefs: the manifest's top-level `tiers` object (tier id -> {maxLod, ...}) —
  // needed for the LOD floor clamp, which lives on the tier, not the scan.
  // initialVisibility: Map(partId -> boolean), from ui.js's site-wide Scene
  // panel state (part-toggles.js's `visibilityMapForScan`) — falls back to
  // `!p.hidden` (ignoring any Scene toggle) only for a caller that doesn't
  // have one yet.
  constructor(scanEntry, tier, basePath, tierDefs, initialVisibility) {
    this.scanEntry = scanEntry;
    this.tier = tier;
    this.basePath = basePath;
    this.tierDefs = tierDefs;

    this.group = new THREE.Group();
    this.group.name = `scan-${scanEntry.id}`;
    this.pivot = new THREE.Vector3(...scanEntry.pivot);

    this.materialSet = MaterialSet.build(scanEntry, tier, basePath);
    this.wet = false;
    this.partVisibility = initialVisibility
      ? new Map(initialVisibility)
      : new Map(scanEntry.parts.map((p) => [p.id, !p.hidden]));
    this.staticGroup = null;
    this.lodGroups = new Map(); // level -> THREE.Group, currently in the scene
    this.lodSelector = new LodSelector(scanEntry.lods, tierDefs[tier].maxLod);
    // Bounding sphere of the body, in scan-local space — app.js projects it
    // through `group.matrixWorld` every frame to get the screen height LOD
    // selection needs (see lod-math.js).
    this.bodySphere = boundingSphere(scanEntry.bodyBounds.min, scanEntry.bodyBounds.max);
    this.currentLoadedLevel = null; // finest level actually resident and visible
    // Every async load remembers the counters it started under and, when it
    // lands, is shown only if they still hold; otherwise its group is freed
    // without touching the scene (design review 22.09 B2).
    this._generation = 0; // bumped whenever a newer LOD load supersedes an in-flight one
    this._tierEpoch = 0; // bumped by setTier: a static load of an older tier is stale
    this._disposed = false;
    this._loadingLevel = null;
    // What an older tier still has on screen until the current tier's own
    // static group lands: LOD groups and the MaterialSets they draw with.
    this._retiredLodGroups = [];
    this._retiredMaterialSets = [];
  }

  // Loads static parts once, then progressively loads LOD levels coarsest
  // -> finest (show each as it lands) so there's always something on
  // screen quickly instead of a blank scan while LOD0 downloads.
  async init() {
    // The render loop's `update` runs while this awaits (from the very
    // first frame, static parts included); once it has asked for a level of
    // its own — a new generation — the progressive pass stops. Carrying on
    // under the new generation used to load a level `update` was already
    // loading a second time: both copies went into the scene, the first one
    // orphaned — never evicted, never switched to wet, drawn over the other.
    const generation = this._generation;
    if (!(await this._loadStatic())) return;

    const levels = [...this.scanEntry.lods].map((l) => l.level).sort((a, b) => b - a); // coarsest first
    for (const level of levels) {
      if (level < this.lodSelector.tierMaxLod) continue; // tier never shows anything finer than its cap
      if (this._disposed || this._generation !== generation) break;
      await this._loadAndShow(level, generation);
    }
  }

  // Loads the current tier's static group and shows it in place of
  // whatever an older tier left on screen. Tier, MaterialSet and epoch are
  // taken before the await; a result that lands after a newer setTier (or
  // dispose) is freed and false returned.
  async _loadStatic() {
    const { tier, materialSet } = this;
    const epoch = this._tierEpoch;
    const group = await loadStatic(this.basePath, this.scanEntry, tier, materialSet, this.wet);
    if (this._disposed || epoch !== this._tierEpoch) {
      disposeGroup(group);
      return false;
    }
    this._applyVisibility(group);
    this._applyWet(group);
    this.group.add(group);
    if (this.staticGroup) {
      this.group.remove(this.staticGroup);
      disposeGroup(this.staticGroup);
    }
    this.staticGroup = group;
    this._dropRetiredLodGroups();
    for (const set of this._retiredMaterialSets) set.dispose();
    this._retiredMaterialSets = [];
    return true;
  }

  _dropRetiredLodGroups() {
    for (const group of this._retiredLodGroups) {
      this.group.remove(group);
      disposeGroup(group);
    }
    this._retiredLodGroups = [];
  }

  // Called every frame by app.js with the projected screen-height fraction;
  // triggers a load if the desired level changed.
  update(projectedHeight) {
    const desired = this.lodSelector.select(projectedHeight);
    if (desired !== this.currentLoadedLevel && desired !== this._loadingLevel) {
      this._generation += 1;
      this._loadAndShow(desired, this._generation);
    }
  }

  async _loadAndShow(level, generation) {
    this._loadingLevel = level;
    const { tier, materialSet } = this;
    let group;
    try {
      group = await loadLod(this.basePath, this.scanEntry, tier, level, materialSet, this.wet);
    } catch (err) {
      console.error(`vrscan: failed to load lod${level} for scan "${this.scanEntry.id}"`, err);
      if (this._loadingLevel === level) this._loadingLevel = null;
      return;
    }
    if (this._disposed || generation !== this._generation) {
      // A newer request (or tier, or dispose) has since been issued —
      // discard this stale one.
      disposeGroup(group);
      return;
    }
    // Shown with the wet state of now, not of when it was requested; and
    // an older tier's body goes the moment this tier has one of its own.
    this._applyVisibility(group);
    this._applyWet(group);
    this._dropRetiredLodGroups();
    this.group.add(group);
    this.lodGroups.set(level, group);
    this.currentLoadedLevel = level;
    this._loadingLevel = null;
    this._evictExcessLevels(level);
  }

  _evictExcessLevels(justShownLevel) {
    if (this.lodGroups.size <= MAX_RESIDENT_LODS) return;
    // Keep the just-shown level; drop others, farthest-from-it first.
    const levels = [...this.lodGroups.keys()]
      .filter((l) => l !== justShownLevel)
      .sort((a, b) => Math.abs(b - justShownLevel) - Math.abs(a - justShownLevel));
    while (this.lodGroups.size > MAX_RESIDENT_LODS && levels.length) {
      const evict = levels.pop();
      const group = this.lodGroups.get(evict);
      this.group.remove(group);
      disposeGroup(group);
      this.lodGroups.delete(evict);
    }
  }

  setWet(wet) {
    this.wet = wet;
    for (const part of this.scanEntry.parts) {
      if (!this.materialSet.hasWet(part.material)) continue;
      const material = this.materialSet.get(part.material, wet);
      this._forEachPartMesh(part.id, (mesh) => {
        mesh.material = material;
      });
    }
  }

  setPartVisible(partId, visible) {
    this.partVisibility.set(partId, visible);
    this._forEachPartMesh(partId, (mesh) => {
      mesh.visible = visible;
    });
  }

  // Rebuilds the MaterialSet and reloads static+LOD geometry for a new
  // tier. Async — callers that care about completion should await it. The
  // old tier stays on screen until the new one's groups land; a newer
  // setTier before then makes this one's loads stale.
  async setTier(tier) {
    if (tier === this.tier) return;
    this._retiredLodGroups.push(...this.lodGroups.values());
    this._retiredMaterialSets.push(this.materialSet);

    this.tier = tier;
    this.materialSet = MaterialSet.build(this.scanEntry, tier, this.basePath);
    this.lodGroups = new Map();
    this.currentLoadedLevel = null;
    this._loadingLevel = null;
    this.lodSelector.setTierMaxLod(this.tierDefs[tier].maxLod);
    this.lodSelector.currentLevel = null; // force a fresh progressive load below
    this._tierEpoch += 1;
    this._generation += 1;
    const generation = this._generation;

    if (!(await this._loadStatic())) return;
    const coarsest = Math.max(...this.scanEntry.lods.map((l) => l.level));
    await this._loadAndShow(coarsest, generation);
  }

  _applyVisibility(group) {
    group.traverse((node) => {
      if (node.isMesh && this.partVisibility.has(node.userData.partId)) {
        node.visible = this.partVisibility.get(node.userData.partId);
      }
    });
  }

  // A level requested before a wet toggle arrives with that moment's
  // materials — bring it in line with the current state before showing it.
  _applyWet(group) {
    group.traverse((node) => {
      const part = node.isMesh ? this.scanEntry.parts.find((p) => p.id === node.userData.partId) : null;
      if (part && this.materialSet.hasWet(part.material)) node.material = this.materialSet.get(part.material, this.wet);
    });
  }

  _forEachPartMesh(partId, fn) {
    if (this.staticGroup) {
      const node = this.staticGroup.getObjectByName(partId);
      if (node && node.isMesh) fn(node);
    }
    for (const group of this.lodGroups.values()) {
      const node = group.getObjectByName(partId);
      if (node && node.isMesh) fn(node);
    }
  }

  // Loads still in flight free their own groups when they land.
  dispose() {
    this._disposed = true;
    if (this.staticGroup) disposeGroup(this.staticGroup);
    for (const group of this.lodGroups.values()) disposeGroup(group);
    for (const group of this._retiredLodGroups) disposeGroup(group);
    this.materialSet.dispose();
    for (const set of this._retiredMaterialSets) set.dispose();
  }
}
