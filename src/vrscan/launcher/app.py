"""`vrscan gui` — a console-free tkinter window: pick a VRGirlz data folder,
pick poses, quality and output folder, convert with a progress bar, then
serve the result and show a screen URL plus a QR code for the headset.

The window only draws. What happens and when — which worker process runs,
what its events mean, which processes get stopped — is
`flow.LauncherFlow`, testable without a display. Written for `pythonw` and
the windowed exe: nothing here assumes a visible terminal, and anything a
user can't see on screen goes to `launcher.log` in `flow.log_dir()`.
"""

from __future__ import annotations

import atexit
import io
import logging
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import segno

from vrscan.launcher.flow import LauncherFlow, ScanList, log_dir, logger

WRAP = 480  # label wrap width, px
POSES_PER_ROW = 6
NOTE_COLOUR = "#a15c00"  # an assumed release, poses left out, a partial success


class LauncherApp:
    def __init__(self, root: tk.Tk, *, fake: bool = False, out_override: Path | None = None) -> None:
        self.root = root
        self.flow = LauncherFlow(self, fake=fake, out_override=out_override)

        self.scan_vars: dict[str, tk.BooleanVar] = {}
        self.tier_var = tk.StringVar(value="4k")
        self.status_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self._qr_photo: tk.PhotoImage | None = None  # keep a ref — tkinter drops GC'd images

        root.title("VRScan")
        root.geometry("540x520")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.report_callback_exception = self._on_callback_error
        atexit.register(self.flow.close)

        self.show_choose()
        self._pump()

    # ---- window plumbing ----

    def _pump(self) -> None:
        self.flow.pump()
        self.root.after(100, self._pump)

    def _on_close(self) -> None:
        self.flow.close()
        self.root.destroy()

    def _on_callback_error(self, exc_type, exc, tb) -> None:
        logger.error("unexpected error in the window", exc_info=(exc_type, exc, tb))
        messagebox.showerror("VRScan", f"Something went wrong: {exc}\n\nLog: {log_dir() / 'launcher.log'}")

    def _screen(self, title: str) -> ttk.Frame:
        for child in self.root.winfo_children():
            child.destroy()
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=title, font=("", 12, "bold"), wraplength=WRAP).pack(anchor="w")
        return frame

    @staticmethod
    def _text(frame: ttk.Frame, text: str, colour: str | None = None, pady: tuple[int, int] = (4, 0)) -> None:
        if text:
            ttk.Label(frame, text=text, foreground=colour, wraplength=WRAP, justify="left").pack(anchor="w", pady=pady)

    # ---- step 1: choose a folder ----

    def show_choose(self, error: str = "") -> None:
        frame = self._screen("Choose your VRGirlz data folder")
        self._text(frame, "The *_Data folder next to the game's .exe (it contains level0).", pady=(4, 8))
        self._text(frame, error, "red")
        ttk.Button(frame, text="Choose folder…", command=self._choose_folder).pack(anchor="w", pady=12)

    def _choose_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose your VRGirlz data folder", mustexist=True)
        if chosen:
            self.flow.choose_folder(Path(chosen))

    def show_loading(self, data_dir: Path) -> None:
        self._screen(f"Reading {data_dir.name}…")

    # ---- step 2: poses, quality, output folder ----

    def show_options(self, scans: ScanList, out_dir: Path, error: str = "") -> None:
        frame = self._screen("Choose poses")
        poses = ttk.Frame(frame)
        poses.pack(anchor="w", fill="x", pady=(4, 0))
        self.scan_vars = {scan_id: self.scan_vars.get(scan_id) or tk.BooleanVar(value=True) for scan_id in scans.ids}
        for index, scan_id in enumerate(scans.ids):
            ttk.Checkbutton(poses, text=scans.label(scan_id), variable=self.scan_vars[scan_id]).grid(
                row=index // POSES_PER_ROW, column=index % POSES_PER_ROW, sticky="w", padx=(0, 8)
            )
        self._text(frame, scans.note, NOTE_COLOUR)
        if scans.skipped:
            self._text(frame, f"Can't be converted ({len(scans.skipped)}): " + "; ".join(scans.skipped), NOTE_COLOUR)

        ttk.Label(frame, text="Quality", font=("", 12, "bold")).pack(anchor="w", pady=(12, 0))
        for value, label in (("2k", "Quest 2 (2k)"), ("4k", "Quest 3 / PC VR (4k)")):
            ttk.Radiobutton(frame, text=label, value=value, variable=self.tier_var).pack(anchor="w")

        ttk.Label(frame, text="Output folder", font=("", 12, "bold")).pack(anchor="w", pady=(12, 0))
        self._text(frame, str(out_dir))
        ttk.Button(frame, text="Change…", command=self._change_out_dir).pack(anchor="w", pady=(4, 0))

        self._text(frame, error, "red", pady=(8, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="w", pady=12)
        ttk.Button(buttons, text="Convert", command=self._convert).pack(side="left")
        ttk.Button(buttons, text="Back", command=self.flow.back).pack(side="left", padx=(8, 0))

    def _change_out_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Choose the output folder", mustexist=False)
        if chosen:
            self.flow.change_out_dir(Path(chosen))

    def _convert(self) -> None:
        self.flow.convert([scan_id for scan_id, var in self.scan_vars.items() if var.get()], self.tier_var.get())

    # ---- step 3: convert ----

    def show_converting(self) -> None:
        frame = self._screen("Converting…")
        self.progress_var.set(0.0)
        ttk.Progressbar(frame, variable=self.progress_var, maximum=100).pack(fill="x", pady=12)
        self.status_var.set("Reading the data folder…")
        ttk.Label(frame, textvariable=self.status_var, wraplength=WRAP).pack(anchor="w")
        ttk.Button(frame, text="Cancel", command=self.flow.cancel).pack(anchor="w", pady=12)

    def show_progress(self, pct: float | None, status: str) -> None:
        if pct is not None:
            self.progress_var.set(pct)
        self.status_var.set(status)

    # ---- step 4: serve, URLs and QR ----

    def show_starting(self) -> None:
        self._screen("Starting the viewer…")

    def show_ready(self, screen_url: str, headset_url: str | None, warning: str) -> None:
        frame = self._screen("Ready")
        self._text(frame, warning, NOTE_COLOUR)

        ttk.Label(frame, text=screen_url).pack(anchor="w", pady=(8, 0))
        ttk.Button(frame, text="Open on this screen", command=lambda: webbrowser.open(screen_url)).pack(anchor="w", pady=8)

        if headset_url:
            ttk.Label(frame, text=f"Headset: {headset_url}").pack(anchor="w")
            ttk.Label(frame, image=self._make_qr_photo(headset_url)).pack(anchor="w", pady=8)
        else:
            self._text(frame, "No LAN address found — connect the headset to the same network as this PC.", "red")
        ttk.Button(frame, text="Convert another folder", command=self.flow.back).pack(anchor="w", pady=(4, 0))

    def _make_qr_photo(self, url: str) -> tk.PhotoImage:
        qr = segno.make(url, error="m")
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=5, border=2)
        self._qr_photo = tk.PhotoImage(data=buf.getvalue())
        return self._qr_photo


def _start_logging() -> None:
    try:
        logs = log_dir()
        logs.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(logs / "launcher.log", encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run(*, fake: bool = False, out_override: str | None = None) -> int:
    _start_logging()
    logger.info("launcher started (fake=%s)", fake)
    root = tk.Tk()
    LauncherApp(root, fake=fake, out_override=Path(out_override) if out_override else None)
    root.mainloop()
    return 0
