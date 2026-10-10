# vrscan viewer

Static three.js viewer, no build step (ES modules + importmap).

## 1. Get the dependencies (vendor three.js + WebXR controller models)

`viewer/vendor/` is gitignored — pull it locally before running anything:

```
../scripts/vendor-three.sh
```

Idempotent: safe to re-run, skips each download if the vendored version
already matches.

- `three@0.180.0` -> `viewer/vendor/three/` (`build/` + `examples/jsm/`,
  `VERSION` file).
- `@webxr-input-profiles/assets` -> `viewer/vendor/webxr-input-profiles/profiles/`
  (controller glTF models + `profilesList.json`; only `dist/profiles/` from
  the npm package, ~80 MB — every headset's controller model, not just
  Quest's). `XRControllerModelFactory.setPath()` in `controls-xr.js` points
  here so controller models load from the LAN, not a CDN (offline-first per
  the plan).

## 2. Structure

```
viewer/
  index.html        # the real viewer — importmap, mounts src/main.js
  smoke.html         # Ф0 VR smoke test (see below), unrelated to src/
  src/
    main.js          # bootstrap: URL params, manifest fetch, ?autotest=1
    app.js            # renderer/scene/camera/rig, render loop, LOD tick
    manifest.js        # fetch+validate manifest.json, tier resolution
    quality.js          # Ф1 auto tier guess (navigator.xr + UA)
    loader.js            # MaterialSet + GLTFLoader (static once, lod<K> on demand)
    lod.js                 # own LOD selector (projected screenHeight, hysteresis, tier floor)
    scan.js                 # ScanInstance: group, wet toggle, part visibility, tier switch
    controls-flat.js         # OrbitControls + WASD
    controls-xr.js            # VRButton, rig placement, grab, snap-turn, exit panel
    ui.js                      # DOM side panel (scan/tier/wet/parts/stats)
  dev-sample/
    make-sample.py              # generates a synthetic package (see below)
    manifest.json, scans/s1/…    # generated output, committed (~210 KiB)
  vendor/                          # gitignored, see step 1
```

## 3. Run against the synthetic dev-sample (no real converter/Unity data needed)

`viewer/dev-sample/` is a synthetic package matching `docs/manifest.md`
exactly (validated against `vrscan.manifest.Manifest`) — one scan `s1`, a
low-poly humanoid silhouette + a cube "sofa" + an alpha-blend "shadow"
plane, two tiers. Regenerate it any time with:

```
uv run python viewer/dev-sample/make-sample.py
```

Then, from the repo root:

```
uv run vrscan serve viewer/dev-sample --port 8099
```

This serves `viewer/` at `/` and `viewer/dev-sample/` at `/out/` (matching
what `vrscan convert`'s real output directory would look like). Open
`http://localhost:8099/?tier=4k` — `?scan=`, `?tier=` (`2k`/`4k`), and
`?manifest=` (default `/out/manifest.json`) all override the picked
defaults. `?autotest=1` runs a scripted 5 s pass (wet on, hide a part,
switch tier) and writes a PASS/FAIL summary into `#autotest` — used for
headless verification (`chromium --headless=new --enable-unsafe-swiftshader`),
no human or headset required.

## 4. Run against a real converted package

```
uv run vrscan serve <out-dir> --https --port 8443
```

then open the printed LAN/tailnet URL (`?tier=` etc. still apply). `--https`
is required for WebXR on anything but `localhost` — see step 5.

## 5. What `smoke.html` should show

Ф0 smoke test proving the "HTTPS server on a real-data machine -> Quest 3 Browser ->
immersive-vr + controllers" path works, independent of the real viewer.

On a desktop browser: a grey floor grid and a small rotating cyan cube ~1.3 m
up, 1.5 m away, orbit-controllable with the mouse, plus a diagnostics overlay
in the top-left corner (WebXR support, WebGL capabilities, GPU, FPS).

On Quest 3 Browser: tap "ENTER VR" — the scene should appear in the headset,
both controllers should show as rays/models, and pulling the trigger while
pointing at the cube should let you pick it up and carry it (release the
trigger to drop it back in place). The overlay should log `sessionstart` /
`sessionend` events.

## 6. Quest 3 checklist for the real viewer (index.html)

- Open the `https://` LAN/tailnet URL in Quest Browser, "Advanced -> Proceed"
  past the self-signed cert warning.
- Flat screen first: scan loads, LOD upgrades as you dolly the camera in
  (orbit + WASD), wet toggle changes the skin's specular look, unchecking a
  part (sofa/shadow) hides it.
- Tap "Enter VR" (bottom of the side panel) — session starts with you
  standing ~1.5 m in front of the scan, facing it.
- Point a controller's ray at any part of the scan and pull the trigger:
  the whole scan should attach to that controller and follow it; release to
  drop it back in the world.
- Push a thumbstick left/right past the deadzone: view should snap-turn 30°
  per push, not continuously.
- Point at the red "EXIT VR" panel near you and pull the trigger — session
  should end cleanly, control returns to OrbitControls on the flat screen.
- No dropped frames/stutter for ~5 minutes on the `4k` tier (Ф1 acceptance
  in the plan; this is also where a real Quest 2 on `2k` — LOD floor
  `maxLod: 1` — should be checked against the tier's texture budget).
