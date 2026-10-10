"""`vrscan` command line: `scan`, `doctor`, `serve`, `convert`.

`scan` is read-only inspection; `convert` (Ф1) is the actual glTF exporter
(see `gltf/writer.py`, `docs/manifest.md`); `serve` just serves whatever is
already in an output directory plus the (possibly still-empty) `viewer/`
package.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from vrscan import __version__
from vrscan.assembler import PoseResult, assemble_poses
from vrscan.gltf.materials import MaterialPlan, material_key_for_part
from vrscan.gltf.writer import ScanWriteResult, write_scan
from vrscan.ir import Scan
from vrscan.manifest import (
    Bounds,
    EnvironmentEntry,
    Generator,
    GeometryFile,
    Manifest,
    MaterialEntry,
    MaterialProvenance,
    MaterialVariant,
    NormalMapProvenance,
    Provenance,
    ScanEntry,
    ScanLod,
    ScanPart,
    ScanTexture,
    SetEntry,
    Source,
    SourceFile,
    Tier,
    TierBudget,
)
from vrscan.normal_pins import verified_rules
from vrscan.presets import PRESETS, TIERS
from vrscan.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    DatasetError,
    DatasetProfile,
    NormalRule,
    NormalRules,
)
from vrscan.reader.backend import ReaderBackend, TextureId, node_id_str
from vrscan.reader.unitypy_backend import UnityPyBackend
from vrscan.textures import TextureCache

REPO_ROOT = Path(__file__).resolve().parents[2]

# Textures whose brightness we histogram in `scan`: NN_SPC / NNWET_SPC.
# Small (1024²) and specifically what the roughness/wet heuristic in Ф1
# will be based on — decoding them here (unlike every other texture) is the
# plan's explicit exception to "no pixel decode in Ф0".
import re as _re

SPC_TEXTURE_RE = _re.compile(r"^\d{2}(WET)?_SPC$")


# --------------------------------------------------------------------------
# peak RSS, portable (no psutil dependency — resource is POSIX-only, the target machine is
# Windows, so this also covers GetProcessMemoryInfo via ctypes).
# --------------------------------------------------------------------------


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # ctypes defaults to a 32-bit `int` restype/argtypes, which silently
        # mangles the HANDLE on 64-bit Windows (found live: GetProcessMemoryInfo
        # returned success but a zeroed struct) — declare the real signatures.
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else 0

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------


def load_and_assemble(backend: ReaderBackend, data_dir: Path, scan_id_prefix: str = "") -> list[PoseResult]:
    """Backend-agnostic: `data_dir` is read through `backend`, so this works
    identically for `UnityPyBackend` (real data) and `FixtureBackend` (tests).
    `backend.profile` is set by `load()` (auto-detected, or the explicit
    `--profile` override baked into the backend by `cmd_scan`/`cmd_convert`)."""

    scene = backend.load(data_dir)
    return assemble_poses(
        scene, getattr(backend, "profile", None) or DEFAULT_PROFILE, scan_id_prefix,
        fingerprint=backend.get_texture_fingerprint,
    )


def pose_error_event(set_id: str, result: PoseResult) -> dict[str, Any]:
    """The one report of a pose that failed assembly, for `scan` and
    `convert` alike: the JSONL `error` event, whose `message` is also the
    human line — set, pose, part and LOD levels, no traceback (the error is
    an expected one, `PartMaterialError`)."""

    exc = result.error
    return {
        "event": "error", "set": set_id, "scan": result.scan_id, "part": exc.part_id, "levels": exc.levels,
        "message": f"set {set_id!r} scan {result.scan_id!r}: {exc}",
    }


def _spc_histogram(backend: ReaderBackend, texture_id: TextureId) -> list[int] | None:
    image = backend.get_texture_image(texture_id)
    if image is None:
        return None
    hist256 = image.convert("L").histogram()
    return [sum(hist256[i : i + 16]) for i in range(0, 256, 16)]


def _histogram_summary(hist: list[int]) -> str:
    total = sum(hist) or 1
    dark = sum(hist[:5]) / total
    mid = sum(hist[5:11]) / total
    bright = sum(hist[11:]) / total
    peak = max(("dark", dark), ("mid", mid), ("bright", bright), key=lambda kv: kv[1])[0]
    return f"peak in {peak} bins (dark={dark:.0%} mid={mid:.0%} bright={bright:.0%})"


def _texture_report_entry(backend: ReaderBackend, tex: Any, role_override: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role_override or tex.role,
        "id": node_id_str(tex.id),  # the name is only a label: two textures can share one
        "name": tex.name,
        "width": tex.width,
        "height": tex.height,
        "format": tex.format,
        "bytes": tex.byte_size,
    }
    if SPC_TEXTURE_RE.match(tex.name):
        hist = _spc_histogram(backend, tex.id)
        if hist is not None:
            entry["histogram16"] = hist
            entry["histogramSummary"] = _histogram_summary(hist)
    return entry


def _collect_report(scans: list[Scan], backend: ReaderBackend) -> dict[str, Any]:
    report_scans = []
    for scan in scans:
        parts_out = []
        for part in scan.parts:
            lods_out = [
                {
                    "level": lm.level,
                    "vertices": lm.vertex_count,
                    "triangles": lm.triangle_count,
                    "meshes": list(lm.mesh_names),
                }
                for lm in part.lods
            ]
            textures_out = []
            if part.material is not None:
                textures_out.extend(_texture_report_entry(backend, tex) for tex in part.material.textures)
            if part.wet_material is not None:
                # The standalone "<id>Wet" material's own property is
                # `_SpecTex` (already role="specular" via
                # TEXTURE_ROLE_BY_PROPERTY) — relabel to "specularWet" here
                # so it's distinguishable from the dry material's specular.
                textures_out.extend(
                    _texture_report_entry(backend, tex, role_override="specularWet")
                    for tex in part.wet_material.textures if tex.role == "specular"
                )
            parts_out.append(
                {
                    "id": part.id,
                    "kind": part.kind,
                    "hideable": part.hideable,
                    "hidden": part.hidden,
                    "hasLodChain": part.has_lod_chain,
                    "material": part.material.name if part.material else None,
                    "shader": part.material.shader if part.material else None,
                    "lods": lods_out,
                    "textures": textures_out,
                }
            )
        report_scans.append(
            {
                "id": scan.id,
                "rootName": scan.root_name,
                "pivot": list(scan.pivot),
                "lods": [dataclasses.asdict(lvl) for lvl in scan.lods],
                "parts": parts_out,
            }
        )
    return {"scans": report_scans}


def _print_report(report: dict[str, Any]) -> None:
    scans = report["scans"]
    print(f"Poses found: {len(scans)}")
    for scan in scans:
        px, py, pz = scan["pivot"]
        print(f"\n[{scan['id']}] root={scan['rootName']}  pivot=({px:.3f}, {py:.3f}, {pz:.3f})")
        if scan["lods"]:
            print("  LOD levels:")
            for lvl in scan["lods"]:
                print(f"    L{lvl['level']}  screenHeight={lvl['screen_height']:.3f}  triangles={lvl['triangle_count']}")
        print("  parts:")
        for part in scan["parts"]:
            flags = []
            if part["hideable"]:
                flags.append("hideable")
            if part["hidden"]:
                flags.append("hidden by default")
            if not part["hasLodChain"]:
                flags.append("fixed mesh")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"    {part['id']:<10} kind={part['kind']:<10}{flag_str}")
            for lod in part["lods"]:
                print(
                    f"      L{lod['level']}  vertices={lod['vertices']:<8} triangles={lod['triangles']:<8} "
                    f"meshes={','.join(lod['meshes'])}"
                )
            for tex in part["textures"]:
                line = (
                    f"      tex {tex['role']:<14} {tex['name']:<16} "
                    f"{tex['width']}x{tex['height']} {tex['format']:<6} {tex['bytes']} B"
                )
                if "histogramSummary" in tex:
                    line += f"  hist16={tex['histogram16']}  ({tex['histogramSummary']})"
                print(line)


def _resolve_profile_arg(name: str | None) -> DatasetProfile | None:
    """`--profile` CLI value -> `None` (auto-detect in `backend.load()`) or
    the explicitly named `DatasetProfile`."""

    return None if name is None or name == "auto" else PROFILES[name]


def run_scan(backend: ReaderBackend, data_dir: Path, as_json: bool) -> int:
    """Backend-agnostic core of `vrscan scan`. A pose that fails assembly is
    reported (`pose_error_event`) and the others are still shown; the exit
    code is non-zero if any pose failed or none was found."""

    t0 = time.time()
    set_id = _dataset_set_id(data_dir)
    try:
        # The ids `convert --scans` takes: a profile that prefixes scan ids
        # with the set's gets the same prefix here (the launcher offers these).
        poses = load_and_assemble(backend, data_dir, scan_id_prefix=f"{set_id}-")
    except DatasetError as exc:
        if as_json:
            print(json.dumps({"error": str(exc), "scans": []}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    scans = [p.scan for p in poses if p.scan is not None]
    errors = [pose_error_event(set_id, p) for p in poses if p.error is not None]
    report = _collect_report(scans, backend)
    elapsed = time.time() - t0
    peak_rss = peak_rss_bytes()

    if as_json:
        report["dataset"] = _dataset_info(backend, set_id)
        report["errors"] = errors
        report["summary"] = {"ok": len(scans), "failed": len(errors)}
        report["elapsedSeconds"] = elapsed
        report["peakRssBytes"] = peak_rss
        # ASCII-only: a pipe on Windows is encoded in the ANSI code page,
        # which can't hold every name or path (a Cyrillic one on cp1252).
        print(json.dumps(report, indent=2))
    else:
        print(f"vrscan scan: {data_dir}")
        if getattr(backend, "support", None) != "supported":
            print(f"warning: {getattr(backend, 'support_note', '')}")
        _print_report(report)
        if errors:
            print()
        for event in errors:
            print(f"[{event['scan']}] ERROR: {event['message']}")
        print(f"\nPoses: {len(scans)} ok, {len(errors)} failed")
        print(f"Elapsed: {elapsed:.1f}s   Peak RSS: {peak_rss / (1024 * 1024):.0f} MiB")

    return 0 if scans and not errors else 1


def cmd_scan(args: argparse.Namespace) -> int:
    backend = UnityPyBackend(profile=_resolve_profile_arg(args.profile))
    return run_scan(backend, Path(args.data_dir), args.json)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def cmd_doctor(_args: argparse.Namespace) -> int:
    import UnityPy

    print(f"python:  {sys.version.split()[0]} ({sys.platform})")
    print(f"UnityPy: {UnityPy.__version__}")
    try:
        import pydantic

        print(f"pydantic: {pydantic.VERSION}")
    except ImportError:
        print("pydantic: NOT INSTALLED")
    toktx = shutil.which("toktx")
    print(f"toktx:   {toktx or 'not found on PATH (needed from Ф2 for KTX2 export)'}")
    return 0


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


def _local_ipv4_addresses() -> list[str]:
    addrs: set[str] = {"127.0.0.1"}
    try:
        addrs.update(ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127."))
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(addrs)


def _cert_paths(out_dir: Path) -> tuple[Path, Path]:
    cert_dir = out_dir / ".vrscan"
    return cert_dir, cert_dir / "cert.pem"


def _get_or_create_cert(out_dir: Path, hosts: list[str]) -> Path:
    cert_dir, cert_path = _cert_paths(out_dir)
    if cert_path.exists():
        return cert_path

    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vrscan-local")])

    san: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for host in hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            san.append(x509.DNSName(host))

    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_dir.mkdir(parents=True, exist_ok=True)
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    return cert_path


STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".gltf": "model/gltf+json",
    ".glb": "model/gltf-binary",
    ".bin": "application/octet-stream",
    ".ktx2": "image/ktx2",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".wasm": "application/wasm",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _make_handler(viewer_root: Path, out_root: Path):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import unquote, urlparse

    viewer_root = viewer_root.resolve()
    out_root = out_root.resolve()

    def resolve(url_path: str) -> Path | None:
        """Map a URL path to a whitelisted file under viewer_root or
        out_root, or None if it's not servable (unknown extension, escapes
        the root, or missing)."""

        path = unquote(urlparse(url_path).path)
        if path.startswith("/out/"):
            base, rel = out_root, path[len("/out/") :]
        else:
            base, rel = viewer_root, path.lstrip("/")
        if not rel:
            rel = "index.html"
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None  # path traversal attempt
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if candidate.suffix.lower() not in STATIC_CONTENT_TYPES:
            return None
        if not candidate.is_file():
            return None
        return candidate

    class Handler(BaseHTTPRequestHandler):
        server_version = "vrscan/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep stdout quiet
            pass

        def _serve_file(self) -> None:
            path = resolve(self.path)
            if path is None:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(b"not found")
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", STATIC_CONTENT_TYPES[path.suffix.lower()])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:
            self._serve_file()

        def do_HEAD(self) -> None:
            self._serve_file()

    return Handler


def _viewer_dir() -> Path:
    """The viewer's files: `viewer/` in the repo, or the copy the packaged
    exe carries next to its modules (`packaging/vrscan.spec`)."""

    bundle = getattr(sys, "_MEIPASS", None)
    return Path(bundle) / "viewer" if bundle else REPO_ROOT / "viewer"


class _ExclusiveServer(ThreadingHTTPServer):
    """A port another program listens on makes the bind fail instead of
    being shared. `HTTPServer` sets SO_REUSEADDR, which on Windows lets a
    second socket bind a port that is in use — a foreign server on 8443
    would quietly lose connections to this one; SO_EXCLUSIVEADDRUSE there.
    Elsewhere SO_REUSEADDR only skips TIME_WAIT, and stays."""

    daemon_threads = True

    if sys.platform == "win32":
        allow_reuse_address = False

        def server_bind(self) -> None:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()


SERVE_PORT_ATTEMPTS = 10  # the port asked for and the ones after it, then any free one


def _bind_server(host: str, port: int, handler_cls: type) -> ThreadingHTTPServer:
    """A server bound to `port`, or to the first free one after it, or to
    one the OS picks. Binding is how a port is found free, so there is no
    gap between finding it and taking it; the caller reports the port it
    got."""

    candidates = [p for p in range(port, port + SERVE_PORT_ATTEMPTS) if p <= 65535] + [0]
    for candidate in candidates:
        try:
            return _ExclusiveServer((host, candidate), handler_cls)
        except OSError as exc:
            error = exc
    raise error


def cmd_serve(args: argparse.Namespace) -> int:
    jsonl = args.progress == "jsonl"
    # With --progress jsonl stdout carries only events (ASCII, like convert's).
    human_out = sys.stderr if jsonl else sys.stdout

    def say(msg: str) -> None:
        print(msg, file=human_out)

    def fail(message: str) -> int:
        if jsonl:
            print(json.dumps({"event": "error", "message": message}), flush=True)
        print(f"error: {message}", file=sys.stderr)
        return 1

    out_dir = Path(args.dir).resolve()
    viewer_dir = _viewer_dir()
    if not out_dir.is_dir():
        return fail(f"{out_dir} does not exist")
    if not viewer_dir.exists():
        say(f"warning: {viewer_dir} does not exist yet — / will 404 until the viewer is built")

    handler_cls = _make_handler(viewer_dir, out_dir)
    try:
        server = _bind_server(args.host, args.port, handler_cls)
    except OSError as exc:
        return fail(f"can't listen on {args.host} port {args.port} or any other ({exc.strerror or exc})")

    scheme = "http"
    if args.https:
        import ssl

        hosts = _local_ipv4_addresses()
        try:
            cert_path = _get_or_create_cert(out_dir, hosts)
        except OSError as exc:
            server.server_close()
            return fail(f"can't write the certificate into {out_dir} ({exc.strerror or exc})")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"

    port = server.server_address[1]
    addresses = _local_ipv4_addresses() if args.host in ("", "0.0.0.0") else [args.host]
    if jsonl:
        print(json.dumps({"event": "ready", "scheme": scheme, "port": port, "addresses": addresses}), flush=True)
    say(f"vrscan serve — viewer={viewer_dir}  out={out_dir}")
    if args.port and port != args.port:
        say(f"port {args.port} is taken — serving on {port}")
    say("URLs:")
    for host in addresses:
        say(f"  {scheme}://{host}:{port}/")
    if args.https:
        say("  Quest Browser: open the tailnet/LAN URL above, tap Advanced -> Proceed on the certificate warning.")
    say("Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_app_name(data_dir: Path) -> str:
    return data_dir.name.removesuffix("_Data")


def _dataset_set_id(data_dir: Path) -> str:
    """Manifest `sets` key / `ScanEntry.set` value / scan id prefix root for
    this run's dataset — a slug of `_infer_app_name(data_dir)`, same regex as
    `DatasetProfile.local_id`'s fallback slug. Distinct per real-world
    product even when two products share one `DatasetProfile` (Lucid Dreams
    II and Ariel are both Unity 5.x / `vrgirlz-2016`) — the profile id alone
    can't tell them apart."""

    name = _infer_app_name(data_dir)
    slug = _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if slug:
        return slug
    # A name with no Latin letter or digit at all (a Cyrillic one) slugs to
    # nothing; a short hash of it keeps the id non-empty and stable.
    return "set-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]


def _dataset_info(backend: ReaderBackend, set_id: str) -> dict[str, Any]:
    """Which release and profile a dataset was read as, and how established
    that is (`profiles.match_profile`) — `scan --json`'s `dataset` and
    `convert`'s `dataset` event."""

    profile = getattr(backend, "profile", None)
    return {
        "set": set_id, "unity": getattr(backend, "unity_version", ""), "profile": profile.id if profile else None,
        "support": getattr(backend, "support", None), "note": getattr(backend, "support_note", ""),
    }


def _output_dir_error(out_dir: Path) -> str | None:
    """None if `out_dir` can be created and written to, else why not —
    found out by writing, since permission bits don't say (ACLs, a
    read-only share, Windows' controlled folder access)."""

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / f".vrscan-write-test-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        return f"can't write to the output folder {out_dir} ({exc.strerror or exc}) — choose another folder"
    return None


def _build_source(data_dir: Path, source_files: list[Path], unity_version: str) -> Source:
    """`source_files`/`unity_version` come from `backend.source_files`/
    `backend.unity_version`, set by `load()` — whichever files+profile were
    actually used for this run (Nat: `level0`+`sharedassets1.assets`, as
    before; a `level*`-glob dataset lists whatever matched, per profile)."""

    files = [SourceFile(name=p.name, bytes=p.stat().st_size, sha256=_sha256_file(p)) for p in source_files]
    return Source(app=_infer_app_name(data_dir), unity=unity_version, files=files)


def _approximations(plans: dict[str, MaterialPlan]) -> list[str]:
    approx = []
    if any(p.recipe.use_orm for p in plans.values()):
        approx.append("roughness = clamp(1 - luma(SPC), 0.25, 0.95)")
    if any(p.wet_specular is not None for p in plans.values()):
        approx.append("wet sheen = WET_SPC roughness (approximation, not the source shader)")
    if "body" in plans:
        approx.append("no SSS")
        approx.append("no micro-detail normals")
    return approx


def _build_scan_entry(scan: Scan, result: ScanWriteResult, source: Source, set_id: str) -> ScanEntry:
    parts = [
        ScanPart(
            id=p.id, kind=p.kind, material=material_key_for_part(p), lod=p.has_lod_chain,
            hideable=p.hideable, hidden=p.hidden,
        )
        for p in scan.parts
    ]

    materials: dict[str, MaterialEntry] = {}
    provenance_materials: dict[str, MaterialProvenance] = {}
    for key, plan in result.material_plans.items():
        recipe = plan.recipe
        textures_map: dict[str, str] = {"baseColor": f"{plan.prefix}_CLR"}
        # A normal map the run skipped (no rule for it) has no file to point at.
        if recipe.use_normal and any(f"{plan.prefix}_NM" in t for t in result.textures_by_tier.values()):
            textures_map["normal"] = f"{plan.prefix}_NM"
        if recipe.use_orm and (plan.texture("occlusion") is not None or plan.texture("specular") is not None):
            textures_map["orm"] = f"{plan.prefix}_ORM"
        variants: dict[str, MaterialVariant] = {}
        if plan.wet_specular is not None:
            variants["wet"] = MaterialVariant(textures={"orm": f"{plan.prefix}_ORM_wet"})
        materials[key] = MaterialEntry(
            type=recipe.type,
            alphaMode=recipe.alpha_mode,
            alphaCutoff=recipe.alpha_cutoff,
            doubleSided=recipe.double_sided,
            textures=textures_map,
            variants=variants,
            roughnessFactor=recipe.roughness_factor,
            baseColorFactor=plan.base_color_factor(),
            blend=recipe.blend,
        )
        provenance_materials[key] = MaterialProvenance(
            unity=plan.source.name, shader=plan.source.shader, unmapped=plan.unmapped_properties()
        )

    textures_by_tier: dict[str, dict[str, ScanTexture]] = {}
    provenance_textures: dict[str, str] = {}
    for tier, texmap in result.textures_by_tier.items():
        textures_by_tier[tier] = {
            key: ScanTexture(file=texp.file, size=texp.size, channels=texp.channels, gpuBytes=texp.gpu_bytes)
            for key, texp in texmap.items()
        }
        for key, texp in texmap.items():
            provenance_textures.setdefault(key, texp.source_name)

    budget: dict[str, TierBudget] = {}
    for tier, texmap in result.textures_by_tier.items():
        gpu_textures = sum(t.gpu_bytes for t in texmap.values())
        lod0 = next((r for level, _sh, r in result.lods if level == 0), None)
        gpu_lod0 = lod0.bytes if lod0 is not None else result.static.bytes
        budget[tier] = TierBudget(gpuBytesTextures=gpu_textures, gpuBytesLod0=gpu_lod0)

    all_mesh_names = sorted({name for part in scan.parts for lm in part.lods for name in lm.mesh_names})
    provenance = Provenance(
        sceneRoot=scan.root_name,
        meshes=all_mesh_names,
        materials=provenance_materials,
        textures=provenance_textures,
        normalMaps={
            key: NormalMapProvenance(
                source=note.source, fingerprint=note.fingerprint,
                rule=note.decision.rule.key() if note.decision.rule is not None else None,
                basis=note.decision.basis, status=note.decision.status, evidence=note.evidence,
            )
            for key, note in result.normal_maps.items()
        },
        approximations=_approximations(result.material_plans),
    )

    return ScanEntry(
        id=scan.id,
        dir=result.dir_name,
        pivot=result.pivot,
        bounds=Bounds(min=result.bounds_min, max=result.bounds_max),
        bodyBounds=Bounds(min=result.body_bounds_min, max=result.body_bounds_max),
        heightM=result.height_m,
        tiers=result.tiers,
        static=GeometryFile(
            gltf="static.{tier}.gltf", bin="geometry.static.bin", bytes=result.static.bytes,
            triangles=result.static.triangles,
        ),
        lods=[
            ScanLod(
                gltf=f"lod{level}.{{tier}}.gltf", bin=f"geometry.lod{level}.bin",
                bytes=r.bytes, triangles=r.triangles, level=level, screenHeight=sh,
            )
            for level, sh, r in result.lods
        ],
        parts=parts, materials=materials, textures=textures_by_tier, budget=budget, provenance=provenance,
        source=source, set=set_id,
    )


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def _load_existing_manifest(out_dir: Path) -> Manifest | None:
    """`None` means only "no manifest.json yet" (first run in this
    `out_dir`) — a manifest.json that exists but fails to read/parse/
    validate is a bug or disk corruption, not "no manifest", and must not
    be treated as one: `_merge_manifest` would then write a brand-new
    manifest containing only THIS run's scans, silently dropping every
    scan the real (unreadable) file listed. Found live 22.09 15:38 UTC: a
    BOM-prefixed manifest.json was read as "no manifest" this way, and
    converting one dataset's scan dropped 18 scans from two others.
    `utf-8-sig` tolerates a BOM if present (transparent for plain UTF-8
    too) — everything else that can go wrong (missing/unreadable file
    content once it's known to exist, invalid JSON, schema mismatch)
    raises instead of swallowing the problem."""

    path = _manifest_path(out_dir)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
        return Manifest.model_validate_json(text)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{path} exists but can't be read as a vrscan manifest ({exc}) — fix or remove it "
            "manually before converting. vrscan will not silently overwrite a manifest.json it "
            "can't first load and merge into."
        ) from exc


def _check_no_set_collision(scans_by_id: dict[str, ScanEntry], new_scan_sets: dict[str, str | None]) -> None:
    """Scan ids only need to be unique within their own dataset (`set`) —
    `assemble(..., scan_id_prefix=f"{set_id}-")` is what actually makes that
    true in practice. If two DIFFERENT datasets ever produced the same scan
    id (a broken prefix, a hand-edited manifest, ...), blindly `.update()`-ing
    `scans_by_id` would silently replace one dataset's scan with another's
    under a shared id, corrupting the (set, scanId) pairing that `--scans`
    filtering and the viewer's scan picker both rely on. A prior manifest
    entry with `set=None` predates the `set` field and isn't treated as a
    real collision (only re-converts, never blocks upgrading old data).

    `run_convert` runs this before writing anything as well as in the merge:
    a scan's files go to `scans/<id>` before the manifest is merged, so a
    check at merge time alone would already have replaced the other
    dataset's scan directory under the entry it leaves in place. Two
    profiles keep bare ids (Nat-21 and Saph-11 are both `vrgirlz-2015`),
    which is what makes such a collision possible at all."""

    for scan_id, new_set in new_scan_sets.items():
        prior = scans_by_id.get(scan_id)
        if prior is not None and prior.set is not None and new_set is not None and prior.set != new_set:
            raise ValueError(
                f"scan id {scan_id!r} already exists in manifest.json under set {prior.set!r}; this run "
                f"would overwrite it under set {new_set!r}. Scan ids must be unique per (set, scan "
                "id), not just globally — rename or remove the stale entry in manifest.json if this "
                "collision is intentional."
            )


def _missing_scan_files(entry: ScanEntry, out_dir: Path) -> list[str]:
    """Every file path `entry` carries (relative to `out_dir / entry.dir`)
    that isn't actually present on disk — checked right before writing
    manifest.json so a reference to a file that was never written (or got
    deleted/moved out from under an untouched carried-over scan) is caught
    as a build error instead of shipping a manifest the viewer can 404 on.
    Doesn't check `scan.gltf` — every `write_scan()` run always produces it
    regardless, it isn't a field the schema itself tracks."""

    scan_dir = out_dir / entry.dir
    missing: list[str] = []

    def check(rel: str) -> None:
        if not (scan_dir / rel).is_file():
            missing.append(f"{entry.dir}/{rel}")

    for tier in entry.tiers:
        check(entry.static.gltf.format(tier=tier))
    check(entry.static.bin)
    for lod in entry.lods:
        check(lod.bin)
        for tier in entry.tiers:
            check(lod.gltf.format(tier=tier))
    for texmap in entry.textures.values():
        for texture in texmap.values():
            check(texture.file)
    return missing


def _merge_manifest(
    out_dir: Path, source: Source, tiers: list[str], new_scans: dict[str, ScanEntry], set_id: str, new_set: SetEntry
) -> None:
    """Rebuild `manifest.json`, keeping every previously-converted scan not
    touched by this run — so `vrscan convert --scans 91` after an earlier
    `--scans 12` doesn't drop pose 12. Written atomically (tmp + replace),
    so a crash anywhere before the final `tmp.replace(path)` (including one
    raised by the checks below) leaves the PREVIOUS manifest.json exactly as
    it was — found live 22.09 that this matters: a re-convert interrupted
    mid-merge must not leave a half-written or missing manifest behind.
    `new_set` upserts into `sets[set_id]` (`set_id` is `_dataset_set_id`'s
    slug, NOT `new_set.profile` — two datasets can share a profile), leaving
    every other dataset's entry untouched."""

    existing = _load_existing_manifest(out_dir)
    tier_defs = {t: Tier(**TIERS[t]) for t in tiers}
    scans_by_id: dict[str, ScanEntry] = {}
    merged_tiers = dict(tier_defs)
    merged_sets: dict[str, SetEntry] = {}
    environments = [EnvironmentEntry(id="room", type="builtin")]
    if existing is not None:
        scans_by_id.update({s.id: s for s in existing.scans})
        merged_tiers = {**existing.tiers, **tier_defs}
        merged_sets = dict(existing.sets)
        if existing.environments:
            environments = existing.environments
    _check_no_set_collision(scans_by_id, {scan_id: entry.set for scan_id, entry in new_scans.items()})
    scans_by_id.update(new_scans)
    merged_sets[set_id] = new_set

    manifest = Manifest(
        schema_="vrscan-manifest/1",
        generator=Generator(name="vrscan", version=__version__),
        source=source,
        tiers=merged_tiers,
        # Only presets whose tier is in this manifest (design review 22.09).
        presets={name: tier for name, tier in PRESETS.items() if tier in merged_tiers},
        scans=list(scans_by_id.values()),
        environments=environments,
        sets=merged_sets,
    )

    # Only the scans THIS run just wrote — not every carried-over entry the
    # existing manifest already had. Checking carried-over entries too would
    # make one stale/hand-edited entry (leftover from an unrelated run) block
    # every future convert; this check exists to catch OUR OWN bugs (writing
    # an entry for a tier/file that didn't actually get produced), not to
    # audit the whole manifest on every run.
    missing = [f for entry in new_scans.values() for f in _missing_scan_files(entry, out_dir)]
    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(
            f"refusing to write manifest.json: {len(missing)} referenced file(s) missing under "
            f"{out_dir}: {preview}{more}"
        )

    path = _manifest_path(out_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True, exclude_none=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _resolve_normal_rules(requested: str, profile: DatasetProfile) -> NormalRules:
    """The rules this run decides every normal map by (`NormalRules`): an
    explicit non-"auto" `--normal-swizzle` for every map; otherwise the
    audited fingerprints, then the profile's rule as "unverified". Nothing
    here depends on the dataset's folder name — `writer` warns per map that
    has no audited fingerprint (design review 22.09 A1)."""

    if requested != "auto":
        return NormalRules(override=NormalRule.for_swizzle(requested), profile_id=profile.id)
    return NormalRules(verified=verified_rules(), profile_rule=profile.normal_rule, profile_id=profile.id)


def run_convert(
    backend: ReaderBackend, data_dir: Path, out_dir: Path, tiers: list[str], wanted: set[str] | None,
    requested_normal_swizzle: str, progress_mode: str | None,
) -> int:
    """Backend-agnostic core of `vrscan convert` — `cmd_convert` supplies
    `UnityPyBackend` (there is no vendor `.assets` file on the VM); tests use
    `FixtureBackend`, same pattern as `load_and_assemble` for `scan`."""

    def emit(event: dict[str, Any]) -> None:
        if progress_mode == "jsonl":
            # ASCII-only, like `scan --json`: see there.
            print(json.dumps(event))
            sys.stdout.flush()

    def human(msg: str) -> None:
        print(msg, file=sys.stderr)

    def fail(message: str, detail: str | None = None) -> int:
        emit({"event": "error", "message": message, **({"detail": detail} if detail else {})})
        human(f"vrscan convert: error: {message}")
        return 1

    t0 = time.time()
    emit({"event": "start", "scans": sorted(wanted) if wanted else [], "tiers": tiers, "out": str(out_dir)})
    human(f"vrscan convert: {data_dir} -> {out_dir}  tiers={tiers}")

    # Before the (long) read: an output folder that can't be written is the
    # user's to fix, and says so instead of failing after the whole load.
    out_error = _output_dir_error(out_dir)
    if out_error:
        return fail(out_error)

    set_id = _dataset_set_id(data_dir)
    try:
        scene = backend.load(data_dir)
        profile = getattr(backend, "profile", None) or DEFAULT_PROFILE
        # A pose that fails assembly is one scan's error, reported like a
        # conversion failure below — not a traceback that takes the whole set
        # down with it (independent review 22.09, P2).
        poses = [
            p for p in assemble_poses(
                scene, profile, scan_id_prefix=f"{set_id}-", fingerprint=backend.get_texture_fingerprint
            )
            if wanted is None or p.scan_id in wanted
        ]
    except DatasetError as exc:
        return fail(str(exc))
    except Exception as exc:  # noqa: BLE001 -- reported as the run's error, with the traceback as detail
        detail = traceback.format_exc()
        human(detail)
        return fail(f"can't read {data_dir.name}: {exc}", detail)
    emit({"event": "dataset", **_dataset_info(backend, set_id)})
    if getattr(backend, "support", None) != "supported":
        human(f"  warning: {getattr(backend, 'support_note', '')}")
    normal_rules = _resolve_normal_rules(requested_normal_swizzle, profile)
    scans = [p.scan for p in poses if p.scan is not None]
    assembly_errors = [p for p in poses if p.error is not None]
    if wanted is not None:
        missing = wanted - {p.scan_id for p in poses}
        if missing:
            human(f"warning: requested scans not found: {sorted(missing)}")
    for pose in assembly_errors:
        event = pose_error_event(set_id, pose)
        emit(event)
        human(f"  [{pose.scan_id}] ERROR: {event['message']}")
    if not scans:
        if not assembly_errors:
            emit({"event": "error", "message": "no matching scans found in source data"})
        return 1

    existing = _load_existing_manifest(out_dir)
    if existing is not None:
        try:
            _check_no_set_collision({s.id: s for s in existing.scans}, {s.id: set_id for s in scans})
        except ValueError as exc:
            emit({"event": "error", "message": str(exc)})
            human(f"vrscan convert: {exc}")
            return 1

    cache = TextureCache(out_dir)
    source = _build_source(
        data_dir, getattr(backend, "source_files", []) or [], getattr(backend, "unity_version", "") or ""
    )
    human(f"  profile: {profile.id}  unity: {source.unity}  files: {[f.name for f in source.files]}")

    # One `sets` entry per dataset (docs/manifest.md), keyed by `set_id`
    # (a slug of the app name, not `profile.id` — two datasets can share a
    # profile, e.g. Lucid Dreams II and Ariel are both `vrgirlz-2016`).
    set_entry = SetEntry(label=source.app or profile.label, profile=profile.id, app=source.app or None)

    new_entries: dict[str, ScanEntry] = {}
    had_error = bool(assembly_errors)
    for index, scan in enumerate(scans):
        scan_t0 = time.time()
        # Progress over the whole run, by finished scans — a scan's own
        # steps don't move it.
        pct = round(100 * index / len(scans), 1)

        def log_step(msg: str, scan_id: str = scan.id, pct: float = pct) -> None:
            human(f"  [{scan_id}] {msg}")
            emit({"event": "step", "scan": scan_id, "step": msg, "pct": pct, "message": f"{scan_id}: {msg}"})

        try:
            result = write_scan(backend, scene, scan, out_dir, tiers, normal_rules, cache, log_step)
            entry = _build_scan_entry(scan, result, source, set_id)
            tmp_dir = out_dir / f"scans/{scan.id}.tmp"
            final_dir = out_dir / result.dir_name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.replace(final_dir)
            new_entries[scan.id] = entry
            elapsed = time.time() - scan_t0
            emit({"event": "scanDone", "scan": scan.id, "dir": result.dir_name, "seconds": elapsed,
                  "bytes": result.static.bytes, "pct": round(100 * (index + 1) / len(scans), 1)})
            human(f"  [{scan.id}] done in {elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001 -- one scan's failure must not kill the whole run
            had_error = True
            emit({"event": "error", "scan": scan.id, "message": str(exc), "detail": traceback.format_exc()})
            human(f"  [{scan.id}] ERROR: {exc}")

    cache.save()
    if new_entries:
        _merge_manifest(out_dir, source, tiers, new_entries, set_id, set_entry)

    total_elapsed = time.time() - t0
    failed = len(assembly_errors) + len(scans) - len(new_entries)
    emit({"event": "done", "seconds": total_elapsed, "manifest": "manifest.json", "ok": len(new_entries),
          "failed": failed})
    human(f"vrscan convert: {len(new_entries)} ok, {failed} failed; done in {total_elapsed:.1f}s, "
          f"peak RSS {peak_rss_bytes() / (1024 * 1024):.0f} MiB")
    return 1 if had_error else 0


def cmd_convert(args: argparse.Namespace) -> int:
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown_tiers = [t for t in tiers if t not in TIERS]
    if unknown_tiers:
        print(f"error: unknown tier(s) {unknown_tiers} (known: {sorted(TIERS)})", file=sys.stderr)
        return 1
    wanted = {s.strip() for s in args.scans.split(",") if s.strip()} if args.scans else None
    backend = UnityPyBackend(profile=_resolve_profile_arg(args.profile))
    return run_convert(backend, Path(args.data_dir), Path(args.out), tiers, wanted, args.normal_swizzle, args.progress)


# --------------------------------------------------------------------------
# gui
# --------------------------------------------------------------------------


def cmd_gui(args: argparse.Namespace) -> int:
    from vrscan.launcher.app import run  # lazy: tkinter isn't needed for any other command

    return run(fake=args.fake, out_override=args.out)


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vrscan", description="VRGirlz data folder inspector / converter / viewer server")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="inspect a VRGirlz *_Data folder, read-only")
    p_scan.add_argument("data_dir", help="path to the *_Data folder (contains level0, sharedassets1.assets)")
    p_scan.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a table")
    p_scan.add_argument(
        "--profile", default="auto", choices=["auto", *sorted(PROFILES)],
        help="dataset naming profile (default: auto-detect from the source's Unity version)",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_doctor = sub.add_parser("doctor", help="print environment/dependency versions")
    p_doctor.set_defaults(func=cmd_doctor)

    p_serve = sub.add_parser("serve", help="serve the viewer + an output directory over HTTP(S)")
    p_serve.add_argument("dir", help="output directory to serve as /out/ (e.g. vrscan convert's output)")
    p_serve.add_argument("--https", action="store_true", help="serve HTTPS with a self-signed cert (required for WebXR)")
    p_serve.add_argument(
        "--port", type=int, default=None,
        help="default: 8443 with --https, else 8000; if another program has it, the next free one is used",
    )
    p_serve.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    p_serve.add_argument(
        "--progress", default=None, choices=["jsonl"],
        help="emit a JSONL `ready` event (actual port, addresses) or `error` event on stdout; human output to stderr",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_convert = sub.add_parser("convert", help="convert a VRGirlz *_Data folder into a glTF package (docs/manifest.md)")
    p_convert.add_argument("data_dir", help="path to the *_Data folder (contains level0, sharedassets1.assets)")
    p_convert.add_argument("--out", required=True, help="output directory (package root, e.g. out/)")
    p_convert.add_argument("--scans", default=None, help="comma-separated pose ids, e.g. 91,12 (default: all found)")
    p_convert.add_argument("--tiers", default="2k,4k", help="comma-separated tier ids, e.g. 2k,4k (default: 2k,4k)")
    p_convert.add_argument(
        "--normal-swizzle", default="auto", choices=["auto", "ag", "rg", "none"],
        help="normal map channel layout for every map (default: the audited rule of each map's content "
        "fingerprint, else the profile's rule marked unverified, with a warning)",
    )
    p_convert.add_argument(
        "--progress", default=None, choices=["jsonl"],
        help="emit JSONL progress events on stdout (docs/manifest.md); human progress always goes to stderr",
    )
    p_convert.add_argument(
        "--profile", default="auto", choices=["auto", *sorted(PROFILES)],
        help="dataset naming profile (default: auto-detect from the source's Unity version)",
    )
    p_convert.set_defaults(func=cmd_convert)

    p_gui = sub.add_parser("gui", help="console-free window: pick a folder, convert, open on screen/headset")
    p_gui.add_argument("--fake", action="store_true", help="drive a fake convert process, for UI testing/demoing")
    p_gui.add_argument("--out", default=None, help="output directory (default: <data_dir>-vrscan-out)")
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve" and args.port is None:
        args.port = 8443 if args.https else 8000
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
