// Flat-screen navigation: OrbitControls around the scan's pivot, plus
// WASD/arrow-key walking (moves the orbit target + camera together so
// orbiting still feels centred on the scan after walking around it).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const WALK_SPEED = 1.6; // m/s
const KEY_TO_AXIS = {
  KeyW: [0, -1], ArrowUp: [0, -1],
  KeyS: [0, 1], ArrowDown: [0, 1],
  KeyA: [-1, 0], ArrowLeft: [-1, 0],
  KeyD: [1, 0], ArrowRight: [1, 0],
};

export class FlatControls {
  constructor(camera, domElement, pivot) {
    this.camera = camera;
    this.orbit = new OrbitControls(camera, domElement);
    this.orbit.target.copy(pivot);
    this.orbit.minDistance = 0.3;
    this.orbit.maxDistance = 8;
    this.orbit.maxPolarAngle = Math.PI * 0.97;
    this.orbit.update();

    this._pressed = new Set();
    this._onKeyDown = (e) => {
      if (e.code in KEY_TO_AXIS) this._pressed.add(e.code);
    };
    this._onKeyUp = (e) => this._pressed.delete(e.code);
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
  }

  setTarget(pivot) {
    this.orbit.target.copy(pivot);
  }

  setEnabled(enabled) {
    this.orbit.enabled = enabled;
  }

  update(deltaSeconds) {
    if (!this.orbit.enabled) return;
    let dx = 0, dz = 0;
    for (const code of this._pressed) {
      const [ax, az] = KEY_TO_AXIS[code];
      dx += ax;
      dz += az;
    }
    if (dx !== 0 || dz !== 0) {
      const len = Math.hypot(dx, dz);
      dx /= len;
      dz /= len;
      // Move along the camera's current facing (flattened to the floor
      // plane), not world axes, so "forward" always means "where you're looking".
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      const right = new THREE.Vector3().crossVectors(forward, this.camera.up).normalize();
      const move = new THREE.Vector3()
        .addScaledVector(right, dx)
        .addScaledVector(forward, -dz)
        .multiplyScalar(WALK_SPEED * deltaSeconds);
      this.camera.position.add(move);
      this.orbit.target.add(move);
    }
    this.orbit.update();
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    this.orbit.dispose();
  }
}
