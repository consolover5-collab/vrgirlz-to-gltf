// VR input: VRButton to enter/exit, rig placement on session start, a
// target-ray raycast grab (any scan mesh -> whole scan.group follows the
// controller), thumbstick snap-turn, and the in-scene control panel
// (vr-panel.js) — the only chrome a headset has, which is why EXIT VR
// lives on it (see the plan: "из VR нельзя выйти [без этого]").

import * as THREE from 'three';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { yawTowardsXZ } from './xr-math.js';

const SNAP_TURN_DEGREES = 30;
const SNAP_TURN_COOLDOWN = 0.4; // seconds, prevents repeat-fire while the stick is held over
const STICK_DEADZONE = 0.7;
const START_DISTANCE = 1.5; // metres in front of the pivot, per the plan
const RAY_LENGTH = 3; // metres, the pointer line when it hits nothing
// xr-standard gamepad mapping (WebXR Device API, "xr-standard" button
// order): 0 trigger, 1 squeeze, 2 touchpad, 3 thumbstick press, 4 A/X,
// 5 B/Y. B (right) / Y (left) opens and closes the control panel — the
// trigger is taken by select (grab/click) and A/X is the one people press
// by accident while gripping.
const PANEL_BUTTON_INDEX = 5;

const _headMatrix = new THREE.Matrix4();
const _headScale = new THREE.Vector3();

export class XrControls {
  // camera must already be a permanent child of `rig` (added once in
  // app.js) — the plan's "one scene graph, only VRButton switches": flat
  // mode works unmodified as long as `rig` sits at identity whenever
  // OrbitControls is driving the camera. `flatControls` is only used to
  // save/restore its OrbitControls' position/target across a VR session
  // (see `_handleSessionStart`/`_handleSessionEnd`).
  constructor(renderer, scene, camera, rig, flatControls) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.rig = rig;
    this.flatControls = flatControls;
    this.pivot = new THREE.Vector3();
    this.grabbable = null;
    this.grabbed = null; // { controller, object } — `object` is what's actually attached to the grip
    this.diagnostics = { selectstart: 0, selectend: 0, squeezestart: 0, squeezeend: 0 };
    this._snapTurnCooldown = 0;
    this._savedCameraPosition = new THREE.Vector3();
    this._savedOrbitTarget = new THREE.Vector3();

    this.vrButton = VRButton.createButton(renderer);

    const modelFactory = new XRControllerModelFactory();
    // Vendored via scripts/vendor-three.sh (@webxr-input-profiles/assets,
    // dist/profiles/ only) — otherwise XRControllerModelFactory falls back
    // to a jsdelivr CDN, which is offline-LAN-hostile (README: "офлайн-LAN
    // с первого релиза").
    modelFactory.setPath('./vendor/webxr-input-profiles/profiles/');

    this._raycaster = new THREE.Raycaster();
    this._tmpMatrix = new THREE.Matrix4();

    // Set by app.js once ui.js has built the DOM the panel renders.
    this.panel = null;
    this._pendingPanelShow = false;
    this._panelButtonDown = new Map(); // handedness -> was B/Y down last frame

    this.controllers = [0, 1].map((i) => this._buildController(i));
    this.grips = [0, 1].map((i) => this._buildGrip(i, modelFactory));

    this._onSessionStart = () => this._handleSessionStart();
    this._onSessionEnd = () => this._handleSessionEnd();
    renderer.xr.addEventListener('sessionstart', this._onSessionStart);
    renderer.xr.addEventListener('sessionend', this._onSessionEnd);
  }

  // The shared control panel (vr-panel.js). Its group is raycast before the
  // model on every select, so clicking a button never grabs the scan.
  setPanel(panel) {
    this.panel = panel;
    for (const controller of this.controllers) panel.listenToController(controller);
  }

  setGrabbable(object3D) {
    // A scan switch mid-grab must drop whatever the grip is actually
    // holding (`this.grabbed.object`) first — swapping `this.grabbable`
    // out from under an active grab left the old mesh parented to a grip
    // that's about to be torn down, with no release ever firing for it
    // (design review 22.09).
    if (this.grabbed) this._release();
    this.grabbable = object3D;
  }

  setPivot(vector3) {
    this.pivot.copy(vector3);
  }

  update(deltaSeconds) {
    if (!this.renderer.xr.isPresenting) return;
    this._snapTurnCooldown = Math.max(0, this._snapTurnCooldown - deltaSeconds);
    const session = this.renderer.xr.getSession();
    if (!session) return;

    // Deferred from sessionstart: there is no headset pose until the first
    // presented frame, and the panel has to be placed in front of where
    // the user actually is (`_headWorldPose` — not `this.camera`).
    if (this._pendingPanelShow) {
      this._pendingPanelShow = false;
      this._showPanel();
    }

    for (const source of session.inputSources) {
      const gp = source.gamepad;
      if (!gp) continue;
      this._pollPanelButton(source, gp);
      if (gp.axes.length < 4 || this._snapTurnCooldown > 0) continue;
      const stickX = gp.axes[2]; // xr-standard mapping: axes[2..3] = thumbstick
      if (Math.abs(stickX) > STICK_DEADZONE) {
        this._snapTurn(stickX > 0 ? -1 : 1);
        this._snapTurnCooldown = SNAP_TURN_COOLDOWN;
      }
    }

    this._updatePointers();
  }

  dispose() {
    this.renderer.xr.removeEventListener('sessionstart', this._onSessionStart);
    this.renderer.xr.removeEventListener('sessionend', this._onSessionEnd);
  }

  _buildController(index) {
    const controller = this.renderer.xr.getController(index);
    this.rig.add(controller);
    const ray = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]),
      new THREE.LineBasicMaterial({ color: 0xffffff }),
    );
    ray.name = 'ray';
    ray.scale.z = 3;
    controller.add(ray);
    controller.addEventListener('selectstart', () => this._onSelectStart(controller));
    controller.addEventListener('selectend', () => this._onSelectEnd(controller));
    controller.addEventListener('squeezestart', () => {
      this.diagnostics.squeezestart += 1;
    });
    controller.addEventListener('squeezeend', () => {
      this.diagnostics.squeezeend += 1;
    });
    // The controller that's holding something can disconnect (battery
    // died, put down) without ever firing `selectend` — release the grab
    // instead of leaving the mesh attached to a grip nothing tracks
    // anymore (design review 22.09).
    controller.addEventListener('disconnected', () => {
      if (this.grabbed && this.grabbed.controller === controller) this._release();
    });
    return controller;
  }

  _buildGrip(index, modelFactory) {
    const grip = this.renderer.xr.getControllerGrip(index);
    grip.add(modelFactory.createControllerModel(grip));
    this.rig.add(grip);
    return grip;
  }

  _onSelectStart(controller) {
    this.diagnostics.selectstart += 1;
    // A click on the panel is InteractiveGroup's business (it dispatches it
    // into the DOM) — the model must not jump into the hand at the same
    // time.
    if (this.panel?.intersect(this._rayFromController(controller))) return;
    if (this.grabbable && !this.grabbed && this._raycastHit(controller, [this.grabbable])) {
      const grip = this.grips[this.controllers.indexOf(controller)];
      grip.attach(this.grabbable);
      this.grabbed = { controller, object: this.grabbable };
    }
  }

  _onSelectEnd(controller) {
    this.diagnostics.selectend += 1;
    if (this.grabbed && this.grabbed.controller === controller) this._release();
  }

  // Reattaches whatever's actually grabbed (`this.grabbed.object`, not
  // `this.grabbable` — the two can differ after a scan switch mid-grab,
  // see `setGrabbable`) back to the scene and clears the grab state. Shared
  // by selectend, controller disconnect, a scan switch, and sessionend.
  _release() {
    if (!this.grabbed) return;
    this.scene.attach(this.grabbed.object);
    this.grabbed = null;
  }

  _raycastHit(controller, objects) {
    return this._rayFromController(controller).intersectObjects(objects, true).length > 0;
  }

  // The shared raycaster, aimed down the controller's target ray.
  _rayFromController(controller) {
    this._tmpMatrix.identity().extractRotation(controller.matrixWorld);
    this._raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
    this._raycaster.ray.direction.set(0, 0, -1).applyMatrix4(this._tmpMatrix);
    return this._raycaster;
  }

  // Pointer feedback: the ray stops at the panel and a dot marks where it
  // lands, so a 44 px button is actually aimable from 70 cm away.
  _updatePointers() {
    let nearest = null;
    for (const controller of this.controllers) {
      const hit = this.panel ? this.panel.intersect(this._rayFromController(controller)) : null;
      const ray = controller.getObjectByName('ray');
      if (ray) ray.scale.z = hit ? hit.distance : RAY_LENGTH;
      if (hit && (!nearest || hit.distance < nearest.distance)) nearest = hit;
    }
    this.panel?.setCursor(nearest ? nearest.point : null);
  }

  _pollPanelButton(source, gamepad) {
    const pressed = gamepad.buttons[PANEL_BUTTON_INDEX]?.pressed === true;
    const key = source.handedness || 'none';
    const wasPressed = this._panelButtonDown.get(key) === true;
    this._panelButtonDown.set(key, pressed);
    if (pressed && !wasPressed) {
      if (this.panel?.visible) this.panel.hide();
      else this._showPanel();
    }
  }

  // Opens the panel in front of wherever the user is looking right now.
  _showPanel() {
    if (!this.panel) return;
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    this._headWorldPose(position, quaternion);
    // YXZ order isolates the yaw of a head that is also pitched/rolled —
    // the panel stays upright regardless of how the user is looking.
    const yaw = new THREE.Euler().setFromQuaternion(quaternion, 'YXZ').y;
    this.panel.show(position, yaw);
  }

  // World-space head pose. While presenting, `this.camera` is NOT it:
  // WebXRManager copies the headset pose into the user camera only inside
  // `renderer.render()` (updateUserCamera), i.e. after the animation
  // callback — so on the first presented frame `this.camera` still holds
  // the flat OrbitControls pose, and a panel placed from it landed ~2 m
  // behind the user, facing away (invisible: the plane is single-sided).
  // The XR camera (`renderer.xr.getCamera()`) gets the pose in
  // onAnimationFrame, before our callback, in reference-space (= rig)
  // coordinates — so head world = rig.matrixWorld * cameraXR.matrix.
  _headWorldPose(position, quaternion) {
    if (this.renderer.xr.isPresenting) {
      this.rig.updateMatrixWorld();
      _headMatrix.multiplyMatrices(this.rig.matrixWorld, this.renderer.xr.getCamera().matrix);
      _headMatrix.decompose(position, quaternion, _headScale);
      return;
    }
    this.camera.getWorldPosition(position);
    this.camera.getWorldQuaternion(quaternion);
  }

  _handleSessionStart() {
    // WebXRManager drives camera.position/.quaternion directly from the
    // headset pose (relative to rig, its parent) for as long as the
    // session presents, and leaves them at that last pose once it ends —
    // save the flat camera's position + OrbitControls' target now so
    // `_handleSessionEnd` can put flat mode back where the user left it
    // instead of wherever VR happened to end (design review 22.09).
    this._savedCameraPosition.copy(this.camera.position);
    this._savedOrbitTarget.copy(this.flatControls.orbit.target);

    this.rig.position.set(this.pivot.x, 0, this.pivot.z + START_DISTANCE);
    // `rig.lookAt()` points a plain Object3D's own +Z at the target, but
    // the camera riding inside `rig` looks down -Z — using it directly
    // aimed the initial view 180° away from the pivot (P1, design review
    // 22.09; see xr-math.js for the exact three.js source citation).
    this.rig.rotation.set(
      0, yawTowardsXZ(this.rig.position.x, this.rig.position.z, this.pivot.x, this.pivot.z), 0,
    );
    // The panel needs a real headset pose — placed on the first presented
    // frame instead (see `update`). It starts open: it carries EXIT VR.
    this._pendingPanelShow = true;
    this._panelButtonDown.clear();
  }

  _handleSessionEnd() {
    this._release();
    this._pendingPanelShow = false;
    this.panel?.hide();
    // Flat mode expects the rig at identity — OrbitControls treats
    // camera.position/.quaternion as world space (see controls-flat.js).
    this.rig.position.set(0, 0, 0);
    this.rig.quaternion.identity();
    // Restore the flat camera + OrbitControls target saved on session
    // start, then let OrbitControls itself recompute the camera's
    // orientation from them (its own doc comment: call update() after any
    // manual change to the camera's transform).
    this.camera.position.copy(this._savedCameraPosition);
    this.flatControls.orbit.target.copy(this._savedOrbitTarget);
    this.flatControls.orbit.update();
  }

  _snapTurn(direction) {
    const angle = direction * THREE.MathUtils.degToRad(SNAP_TURN_DEGREES);
    const camPos = new THREE.Vector3();
    this.camera.getWorldPosition(camPos);
    this.rig.position.sub(camPos).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle).add(camPos);
    this.rig.rotateY(angle);
  }
}

