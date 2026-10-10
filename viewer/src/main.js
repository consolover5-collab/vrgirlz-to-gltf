// Bootstrap: read URL params, fetch the manifest, resolve scan/tier, start
// the app. ?manifest= (default /out/manifest.json), ?scan=, ?tier=.
//
// ?autotest=1 drives a scripted 5s run (wet toggle, hide a part, tier
// switch) and writes a pass/fail summary into #autotest — for headless
// verification (no human, no headset) per the brief's smoke-test step.
//
// ?wet=0|1 starts on the dry or wet variant and lists the roughness maps
// the scene actually binds in a hidden #wet-debug — a dry/wet A/B pair of
// screenshots from two otherwise identical URLs (design review 22.09 point 2).

import { loadManifest, resolveTier, findScan, findScanInSet } from './manifest.js';
import { App } from './app.js';

const errors = [];
window.addEventListener('error', (e) => errors.push(e.message || String(e.error)));
window.addEventListener('unhandledrejection', (e) => errors.push(String(e.reason)));

async function main() {
  const params = new URLSearchParams(location.search);
  const manifestUrl = params.get('manifest') || '/out/manifest.json';

  const manifest = await loadManifest(manifestUrl);
  // ?scan= wins outright (picks its set as a side effect, in Ui.setScan);
  // ?set= alone picks that set's first scan; neither -> first scan overall.
  const scanEntry = params.get('scan')
    ? findScan(manifest, params.get('scan'))
    : findScanInSet(manifest, params.get('set')) || findScan(manifest, null);
  const tier = resolveTier(manifest, params.get('tier'));

  const app = new App(document.body, manifest);
  await app.start(scanEntry, tier);

  if (params.has('wet')) {
    runWetAb(app, params.get('wet') === '1');
  }
  if (params.get('autotest') === '1') {
    runAutotest(app);
  }
  if (params.get('vrpanel') === '1') {
    runVrPanelDebug(app);
  }
}

// ?vrpanel=1: hang the in-scene VR panel (vr-panel.js, HTMLMesh) in front
// of the flat camera, EXIT VR included, and report any error it raised —
// the only way to look at what the headset would show without a headset.
async function runVrPanelDebug(app) {
  const overlay = document.createElement('pre');
  overlay.id = 'vrpanel-debug';
  overlay.style.cssText = 'position:fixed;bottom:0;left:0;background:#000;color:#0f0;padding:8px;margin:0;font:12px monospace;z-index:1000;';
  document.body.appendChild(overlay);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  await sleep(1500);
  app.ui.setPresentation('xr');
  app.ui.setCollapsed(true);
  let thrown = null;
  try {
    app.xrControls._showPanel();
  } catch (err) {
    thrown = err;
    errors.push(`vrpanel: ${err.message || err}`);
  }
  await sleep(1500);
  const panel = app.vrPanel;
  // Smallest button actually shown, in millimetres at the panel's scale
  // (HTMLMesh: 1 CSS px = 1 mm before scaling) — the design review's
  // yardstick for a ray-clickable target is >= 25 mm.
  const scale = panel.mesh ? panel.mesh.scale.x : 0;
  const shown = [...app.ui.panel.querySelectorAll('button')]
    .map((b) => b.getBoundingClientRect().height)
    .filter((h) => h > 0);
  const result = {
    ok: errors.length === 0,
    errors,
    thrown: thrown ? String(thrown.stack || thrown) : null,
    visible: panel.visible,
    mesh: panel.mesh ? { size: panel._meshSize, scale: panel.mesh.scale.x, texture: panel.mesh.material.map?.image ? [panel.mesh.material.map.image.width, panel.mesh.material.map.image.height] : null } : null,
    position: panel.group.position.toArray().map((v) => +v.toFixed(2)),
    yaw: +panel.group.rotation.y.toFixed(2),
    minHitTargetMm: shown.length ? +(Math.min(...shown) * scale).toFixed(1) : null,
    panelHeightMm: panel._meshSize ? +(panel._meshSize.height * scale).toFixed(0) : null,
  };
  overlay.textContent = `vrpanel: ${result.ok ? 'PASS' : 'FAIL'}\n${JSON.stringify(result, null, 2)}`;
}

async function runWetAb(app, wet) {
  app.ui.callbacks.onWetChange(wet);
  const report = document.createElement('pre');
  report.id = 'wet-debug';
  report.style.display = 'none';
  document.body.appendChild(report);
  await new Promise((resolve) => setTimeout(resolve, 4000)); // textures load after the switch
  const maps = new Set();
  const meshesPerPart = {}; // > 1 = the same part drawn twice (a duplicate LOD copy)
  app.scan.group.traverse((node) => {
    if (!node.isMesh || !node.visible) return;
    if (node.userData.partId) meshesPerPart[node.userData.partId] = (meshesPerPart[node.userData.partId] ?? 0) + 1;
    const src = node.material?.roughnessMap?.image?.src;
    if (src) maps.add(new URL(src).pathname);
  });
  report.textContent = JSON.stringify({
    scan: app.scan.scanEntry.id, tier: app.scan.tier, wet: app.scan.wet, roughnessMaps: [...maps].sort(), meshesPerPart,
  });
}

async function runAutotest(app) {
  const overlay = document.createElement('pre');
  overlay.id = 'autotest';
  overlay.style.cssText = 'position:fixed;bottom:0;left:0;background:#000;color:#0f0;padding:8px;margin:0;font:12px monospace;z-index:1000;';
  overlay.textContent = 'autotest: running…';
  document.body.appendChild(overlay);

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const scanEntry = app.scan.scanEntry;
  const hasWet = Object.values(scanEntry.materials).some((m) => m.variants?.wet);
  const hideablePart = scanEntry.parts.find((p) => p.hideable);
  const otherTier = scanEntry.tiers.find((t) => t !== app.scan.tier);

  await sleep(1000);
  if (hasWet) app.scan.setWet(true);

  await sleep(1000);
  if (hideablePart) app.scan.setPartVisible(hideablePart.id, false);

  await sleep(1000);
  if (otherTier) await app.setTier(otherTier);

  await sleep(2000);

  const result = {
    ok: errors.length === 0,
    errors,
    scan: scanEntry.id,
    tier: app.scan.tier,
    wetApplied: hasWet ? app.scan.wet === true : 'n/a (no wet variant)',
    hiddenPart: hideablePart ? `${hideablePart.id}=${app.scan.partVisibility.get(hideablePart.id)}` : 'n/a',
    lodLevel: app.scan.currentLoadedLevel,
    fps: app._fps,
    drawCalls: app.renderer.info.render.calls,
  };
  overlay.textContent = `autotest: ${result.ok ? 'PASS' : 'FAIL'}\n${JSON.stringify(result, null, 2)}`;
}

main().catch((err) => {
  console.error('vrscan: fatal error during startup', err);
  document.body.textContent = `vrscan viewer failed to start: ${err.message}`;
});
