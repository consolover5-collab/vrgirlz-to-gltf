# Launcher (Ф1, `src/vrscan/launcher/`)

A console-free tkinter window that wraps `vrscan scan` + `convert` + `serve`
so the product's entry point is "pick a folder, get a URL/QR", not a
terminal. Rationale and the Ф1→Ф3 sequencing are in the plan (owner decision
22.09: Python launcher + system browser, no Tauri).

## Run it

```sh
vrscan gui                       # real scan/convert/serve
vrscan gui --fake                # fake poses + fake convert (fake_convert.py); the server is real
vrscan gui --out /path/to/out    # start with this output folder instead of <data_dir>-vrscan-out
python -m vrscan.launcher --fake # same thing, without going through the vrscan CLI (dev convenience)
```

## What it does

Every step that reads data or serves it runs in a worker process of its own
(`vrscan scan`/`convert`/`serve`); the window reads their stdout (JSONL, see
`docs/manifest.md`) on a background thread and redraws on its own thread.

1. **Choose folder** — `filedialog.askdirectory`, then `vrscan scan --json
   <folder>`: the reader's own discovery decides whether it is a dataset, the
   launcher has no file list of its own. A refusal is one line, shown as is:
   not a `*_Data` folder (no `level0`), a Unity release no profile covers
   ("not supported"), a file the set needs missing — named, e.g. a
   `.resS`/`.resource`. There is no fallback pose list: poses of another set
   are never offered.
2. **Choose poses, quality, output folder** — checkboxes from the scan (the
   pose number, without the set prefix), a note when the release is only
   "recognised tentatively" or some poses can't be converted, 2k ("Quest 2")
   / 4k ("Quest 3 / PC VR"). The output folder is `<data_dir>-vrscan-out` next
   to the set by default; **Change…** picks any other (writing next to the
   set can be refused). One output folder per set — a library of several sets
   in one folder through the GUI is Ф2.
3. **Convert** — `vrscan convert … --progress jsonl`; a progress bar (by
   finished poses) and the current step. The converter tries writing to the
   output folder before it reads anything, so a folder it can't write comes
   back within a second, on this screen. **Cancel** stops the convert's whole
   process tree and returns to step 2; whatever the stopped process still
   said is ignored. A partial success (`done` with `failed > 0`) goes on to
   step 4 and says "Converted N of M poses. Not converted: …"; nothing
   converted returns to step 2 with the reason.
4. **Serve + share** — `vrscan serve <out> --https --port 8443 --progress
   jsonl`. The server binds by itself: 8443, or the next free port if
   another program has it — never sharing it (on Windows `SO_EXCLUSIVEADDRUSE`:
   plain `SO_REUSEADDR` there would let it bind a port in use) — and
   reports the port it got in a `ready` event. Only then the window shows
   - the screen URL (`https://localhost:<port>/`) with **Open on this
     screen** (`webbrowser.open`);
   - the headset URL, from the addresses in the `ready` event, preferring
     `192.168.*`/`10.*`/`172.16-31.*` over the tailnet (`100.64.0.0/10`)
     over loopback (never) — plus its **QR code** (`segno` PNG straight into
     a `tkinter.PhotoImage`; no Pillow in the GUI path).

   A server that exits before `ready` returns to step 2 with its reason.
   **Convert another folder** stops this set's server and goes back to
   step 1; the next set gets its own output folder and server.
5. **Exit** — closing the window (and `atexit`) stops every process the
   launcher started that is still running, each with its whole tree
   (`taskkill /T` on Windows; elsewhere a worker leads its own process
   group). Nothing is ever stopped by port or by name: a server on 8443 the
   launcher didn't start is left alone.

## Logs

Per user, never next to the data or the output (either can be read-only):
`%LOCALAPPDATA%\VRScan\logs\` on Windows, `~/.local/state/vrscan/logs/`
elsewhere.

| File | What |
|---|---|
| `launcher.log` | the window: folders chosen, commands started, errors, any traceback of the window itself (appended) |
| `scan.log`, `convert.log`, `serve.log` | the stderr of that worker's last run — human progress, tracebacks |

An error on screen ends with the path of the log that has the details.

## Module layout

| File | Purpose |
|---|---|
| `flow.py` | No tkinter. Commands, `parse_scan_report`, `convert_outcome`, `pick_headset_address`, `Worker` (a process, a reader thread, its log), `terminate_process_tree`, and `LauncherFlow` — the steps above, drawn through a `View` protocol. |
| `app.py` | `LauncherApp`, the tkinter `View`, and `run()`. |
| `fake_convert.py` | Dev/test-only stand-in: `--list` prints a `scan --json` report of fake poses; otherwise ~20 `step` events over ~6 s, then `scanDone` + `done` (writes an empty `manifest.json`); `--fail-at PCT` emits `error` and exits 1 instead. |
| `__main__.py` | `python -m vrscan.launcher` — direct dev entry point. |

`vrscan serve --progress jsonl` (the `ready`/`error` events) and the port
fallback live in `cli.py` (`_bind_server`).

## Testing

`tests/test_launcher.py` drives `LauncherFlow` with fake workers and a
recording view: screens in order, the URL and QR only after `ready`, partial
success, a failed convert, a refused folder, cancel, a server that doesn't
start, a second set with its own output and server, close stopping only the
launcher's own processes. Real processes cover `Worker` (events, stderr to
its log), the tree kill (POSIX), `serve` binding next to a foreign listener
without taking its port, its `ready` and `error` events, and `fake_convert`.
A `Tk()` window opens under `xvfb-run` when it is installed (else skipped).

Not covered here: the Windows tree kill and exclusive bind (checked on a real-data machine
with the exe, see below), and a person clicking through the window — the
owner's end-to-end test.

Live check on a machine with a display:

```sh
xvfb-run -a uv run vrscan gui --fake --out /tmp/vrscan-gui-out
```

## Packaged exe (Windows)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build.ps1
```

builds `dist\VRScan\VRScan.exe`: PyInstaller onedir, windowed — the exe and
its `_internal\` folder, shipped together as the `dist\VRScan` folder. It
builds in a venv of its own, `.venv-build` (the project's `.venv` can be held
by a running `vrscan serve`; one under `%TEMP%` has been eaten by Defender),
with versions from `uv.lock` and PyInstaller pinned in the `build`
dependency group. `packaging/vrscan.spec` refuses to build without
`viewer/vendor` (`scripts/vendor-three.sh`).

- **Double-click** — the launcher window, no console.
- **`VRScan.exe --worker <vrscan command …>`** — that one `vrscan` command
  (`src/vrscan/frozen.py`). This is how the window starts its scan, convert
  and serve workers: in the exe `sys.executable` is the exe, there is no
  `-m vrscan.cli`. A worker's stdout (the JSONL) and stderr are the pipe and
  the log the window hands it; one started without them (no console, no
  redirection) drops stdout and appends stderr to `worker.log` in the log
  folder. A worker that crashes leaves a traceback in its log, never
  PyInstaller's error dialog.
- **The bundle** carries, besides the code (the profiles and normal-map pins
  are code): the viewer as served — `index.html`, `src/`, `vendor/`
  (three.js, controller profiles) — in `_internal\viewer`; UnityPy's type
  trees; and two files PyInstaller's analysis can't see: FMOD's library,
  which UnityPy's texture decoding loads by path, and archspec's CPU tables,
  which `astc_encoder` reads on import.
- **Firewall** — the first time the server listens, Windows asks whether
  VRScan may accept connections: allow it on the network the headset is on
  (Private for a home network). The screen URL (`localhost`) works either
  way.

Checked on a real-data machine (23.09) without the window — `LauncherFlow` driven by a
script, with the exe as its worker: scan and convert of Saph (10 poses, 4k)
into a temporary folder, over the pipes; serve next to a listener already on
its port — it took the next one, and the listener kept its connections; the
manifest, a glTF, the page and three.js fetched over HTTPS; cancel stopped
the convert within a second; nothing left running. The window itself is the
owner's double-click test.

## Not yet

This is the Ф1 minimum: one plain tkinter window, good enough to not need a
terminal. Ф2: one library of several sets through the GUI (rules for update,
delete, collisions, concurrent writes). Ф3 ("polish", owner decision 22.09)
replaces the UI with `pywebview` (HTML/CSS inside a native window, same
Python backend); `flow.py` is meant to survive that rewrite unchanged, only
the tkinter widget tree gets replaced.
