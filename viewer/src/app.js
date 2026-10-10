// Bootstrap target: renderer, scene, camera+rig, lighting, controls, UI,
// and the render loop that drives the LOD selector every frame. One scene
// graph for both flat and VR — sessionstart/sessionend only toggle which
// controls are active (see controls-xr.js).

import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FlatControls } from './controls-flat.js';
import { XrControls } from './controls-xr.js';
import { ScanInstance } from './scan.js';
import { Ui } from './ui.js';
import { VrPanel } from './vr-panel.js';
import { projectedSphereHeight } from './lod.js';

export class App {
  constructor(container, manifest) {
    this.manifest = manifest;
    this.container = container;
    this.scan = null;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1f);

    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.03, 50);
    this.camera.position.set(0, 1.6, 2.5);

    // Permanent rig: camera is always its child, in flat mode as well as
    // XR. XrControls only moves `rig` during a session and resets it to
    // identity on sessionend, so OrbitControls' world-space assumptions
    // about camera.position hold whenever it's actually driving the camera.
    this.rig = new THREE.Group();
    this.rig.name = 'player-rig';
    this.rig.add(this.camera);
    this.scene.add(this.rig);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.xr.enabled = true;
    container.appendChild(this.renderer.domElement);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    const grid = new THREE.GridHelper(10, 20, 0x888888, 0x333338);
    this.scene.add(grid);

    this.flatControls = new FlatControls(this.camera, this.renderer.domElement, new THREE.Vector3());
    this.xrControls = new XrControls(this.renderer, this.scene, this.camera, this.rig, this.flatControls);
    this.renderer.xr.addEventListener('sessionstart', () => {
      this.flatControls.setEnabled(false);
      this.ui.setPresentation('xr');
    });
    this.renderer.xr.addEventListener('sessionend', () => {
      this.flatControls.setEnabled(true);
      this.ui.setPresentation('flat');
    });

    this.uiRoot = document.createElement('div');
    this.uiRoot.id = 'ui';
    container.appendChild(this.uiRoot);
    this.ui = new Ui(this.uiRoot, manifest, {
      onScanChange: (scanId) => this.setScan(scanId),
      onTierChange: (tier) => this.setTier(tier),
      onWetChange: (wet) => {
        this.scan?.setWet(wet);
        this.ui.setWet(this.scan?.wet ?? false);
      },
      onPartVisibleChange: (partId, visible) => this.scan?.setPartVisible(partId, visible),
      onExitVr: () => this.renderer.xr.getSession()?.end(),
      onLayoutChange: () => this.vrPanel?.refreshLayout(),
    });
    this.ui.attachVrButton(this.xrControls.vrButton);

    // Same panel, same callbacks, rendered into the scene for the headset.
    this.vrPanel = new VrPanel(this.ui.panel, this.scene);
    this.xrControls.setPanel(this.vrPanel);

    this._clock = new THREE.Clock();
    this._fps = 0;
    this._fpsAccum = 0;
    this._fpsFrames = 0;

    window.addEventListener('resize', () => this._onResize());
    this.renderer.setAnimationLoop(() => this._tick());
  }

  async start(scanEntry, tier) {
    await this.setScan(scanEntry.id, tier);
  }

  async setScan(scanId, tier) {
    const scanEntry = this.manifest.scans.find((s) => s.id === scanId) || this.manifest.scans[0];
    this.ui.setLoading(scanEntry.id);
    const useTier = tier && this.manifest.tiers[tier] ? tier : (this.scan?.tier ?? scanEntry.tiers[0]);

    if (this.scan) {
      this.scene.remove(this.scan.group);
      this.scan.dispose();
    }
    const basePath = `/out/${scanEntry.dir}/`;
    // ui.setScan first: it resets the per-scan "show anyway" state (see
    // ui.js) that visibilityMapFor needs to already be current for THIS
    // scan before ScanInstance's initial partVisibility is computed from it.
    this.ui.setScan(scanEntry, useTier);
    const initialVisibility = this.ui.visibilityMapFor(scanEntry);
    this.scan = new ScanInstance(scanEntry, useTier, basePath, this.manifest.tiers, initialVisibility);
    this.scene.add(this.scan.group);
    this.flatControls.setTarget(this.scan.pivot);
    this.xrControls.setPivot(this.scan.pivot);
    this.xrControls.setGrabbable(this.scan.group);
    await this.scan.init();
    this.ui.setLoading(null);
  }

  async setTier(tier) {
    if (!this.scan) return;
    this.ui.setLoading(tier);
    await this.scan.setTier(tier);
    this.ui.setScan(this.scan.scanEntry, tier);
    this.ui.setWet(this.scan.wet);
    this.ui.setLoading(null);
  }

  _onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  _tick() {
    const delta = this._clock.getDelta();
    this.flatControls.update(delta);
    this.xrControls.update(delta);

    if (this.scan) {
      // Both the scan (grabbable in VR) and the camera can have moved
      // since the last render, so refresh their world matrices before
      // projecting the body sphere through them.
      this.scan.group.updateMatrixWorld();
      this.camera.updateMatrixWorld();
      const projected = projectedSphereHeight(
        this.renderer, this.camera, this.scan.bodySphere, this.scan.group.matrixWorld,
      );
      this.scan.update(projected);
    }

    this.renderer.render(this.scene, this.camera);

    this._fpsFrames += 1;
    this._fpsAccum += delta;
    if (this._fpsAccum >= 0.5) {
      this._fps = this._fpsFrames / this._fpsAccum;
      this._fpsFrames = 0;
      this._fpsAccum = 0;
    }
    if (this.scan) {
      this.ui.updateStats({
        fps: this._fps,
        lodLevel: this.scan.currentLoadedLevel,
        rendererInfo: this.renderer.info,
        scanEntry: this.scan.scanEntry,
        tier: this.scan.tier,
      });
    }
  }
}
