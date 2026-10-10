// The control panel inside the headset. It is not a second UI: three's
// HTMLMesh rasterises the very DOM element ui.js builds into a texture, and
// InteractiveGroup turns a controller ray hit into a DOM mouse event on
// whatever button sits under it — one markup, one state, both modes
// (owner: "кнопки смены нужны и в vr-среде, а не только в браузере").
//
// HTMLMesh caveats this module has to work around:
//  * the canvas is sized once from the element's bounding rect, so a panel
//    that grew (the XR layout replacing the flat one, a scan with another
//    tier count or a "hidden in this pose" note) needs the mesh rebuilt —
//    `_ensureMesh` compares sizes on every opening and, while open, on
//    `refreshLayout`;
//  * the source element must stay laid out (a real rect): ui.js "collapses"
//    the flat panel by translating it off-screen, never `display: none`;
//  * it re-rasterises on every DOM mutation, which is why per-frame stats
//    live outside the panel element.

import * as THREE from 'three';
import { HTMLMesh } from 'three/addons/interactive/HTMLMesh.js';
import { InteractiveGroup } from 'three/addons/interactive/InteractiveGroup.js';
import { panelPlacement } from './xr-math.js';

const PANEL_DISTANCE = 0.7; // metres in front of the user
const PANEL_FLOOR_HEIGHT = 1.2; // metres, panel centre
// HTMLMesh maps 1 CSS pixel to 1 mm, so a 320 px panel would be 32 cm wide
// at arm's length — readable, but small. Scaled up to a comfortable size,
// capped in height so a long pose list can't become a wall.
const PANEL_WIDTH_M = 0.45;
const PANEL_MAX_HEIGHT_M = 0.9;

export class VrPanel {
  constructor(element, scene) {
    this.element = element;
    this.scene = scene;
    this.mesh = null;
    this._meshSize = null; // { width, height } in CSS px the current mesh was built from
    this._relayoutTimer = null;

    this.group = new InteractiveGroup();
    this.group.name = 'vr-panel';
    this.group.visible = false;
    scene.add(this.group);

    // Where the ray lands on the panel — without it, aiming at a 44 px
    // button from 70 cm away is guesswork.
    this.cursor = new THREE.Mesh(
      new THREE.SphereGeometry(0.006, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffe08a, depthTest: false }),
    );
    this.cursor.name = 'vr-panel-cursor';
    this.cursor.renderOrder = 10;
    this.cursor.visible = false;
    scene.add(this.cursor);
  }

  listenToController(controller) {
    this.group.listenToXRControllerEvents(controller);
  }

  get visible() {
    return this.group.visible;
  }

  // Places the panel in front of the camera's current pose and shows it.
  // cameraPosition/cameraYaw are world-space (the rig's transform included).
  show(cameraPosition, cameraYaw) {
    this._ensureMesh();
    const place = panelPlacement(cameraPosition.x, cameraPosition.z, cameraYaw, PANEL_DISTANCE, PANEL_FLOOR_HEIGHT);
    this.group.position.set(place.x, place.y, place.z);
    this.group.rotation.set(0, place.yaw, 0);
    this.group.visible = true;
  }

  hide() {
    this.group.visible = false;
    this.cursor.visible = false;
  }

  toggle(cameraPosition, cameraYaw) {
    if (this.group.visible) this.hide();
    else this.show(cameraPosition, cameraYaw);
  }

  // Nearest intersection of `raycaster` with the panel, or null — used both
  // for the cursor and to keep the model-grab off while the ray is on the
  // panel (controls-xr.js).
  intersect(raycaster) {
    if (!this.group.visible || !this.mesh) return null;
    return raycaster.intersectObject(this.mesh, false)[0] ?? null;
  }

  setCursor(point) {
    if (point) {
      this.cursor.position.copy(point);
      this.cursor.visible = true;
    } else {
      this.cursor.visible = false;
    }
  }

  // ui.js reports (through app.js) every change that can resize the panel.
  // An open panel re-measures one macrotask later, once the batch of DOM
  // mutations that triggered it is done and the layout is final —
  // `window.requestAnimationFrame` would be the obvious hook, but it does
  // not run during an immersive session. A closed panel re-measures on
  // `show` anyway.
  refreshLayout() {
    if (this._relayoutTimer !== null) return;
    this._relayoutTimer = setTimeout(() => {
      this._relayoutTimer = null;
      if (this.group.visible) this._ensureMesh();
    }, 0);
  }

  // HTMLMesh bakes the element's size in at construction, so rebuild
  // whenever it no longer matches. The old mesh's `dispose` drops its own
  // event listeners and the MutationObserver on the element; the
  // controller registration stays on `this.group` (InteractiveGroup
  // raycasts its current children), so the new mesh is interactive as soon
  // as it is added and the old one stops receiving rays once removed.
  _ensureMesh() {
    const rect = this.element.getBoundingClientRect();
    const width = Math.round(rect.width);
    const height = Math.round(rect.height);
    if (!width || !height) return; // element not laid out (yet) — nothing to rasterise
    if (this.mesh && this._meshSize.width === width && this._meshSize.height === height) return;

    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.dispose();
    }
    this.mesh = new HTMLMesh(this.element);
    this.mesh.name = 'vr-panel-mesh';
    const scale = Math.min(PANEL_WIDTH_M / (width * 0.001), PANEL_MAX_HEIGHT_M / (height * 0.001));
    this.mesh.scale.setScalar(scale);
    this._meshSize = { width, height };
    this.group.add(this.mesh);
  }

  dispose() {
    clearTimeout(this._relayoutTimer);
    this._relayoutTimer = null;
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = null;
    }
    this.group.disconnect();
    this.scene.remove(this.group);
    this.scene.remove(this.cursor);
    this.cursor.geometry.dispose();
    this.cursor.material.dispose();
  }
}
