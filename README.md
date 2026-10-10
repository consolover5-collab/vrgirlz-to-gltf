# vrscan

An unofficial converter and web viewer for `VRGirlz`-style Unity scan data —
turns a purchased data folder into glTF you can view on a flat screen or in
VR (WebXR), with a quality switch for different headsets.

> **Disclaimer:** this is not an official tool. It only converts asset files
> you already own; the repository ships no vendor assets. See
> [`DISCLAIMER.md`](DISCLAIMER.md) for the full legal notice.

## Status

Phase 0 (this checkout): repo skeleton, an intermediate representation (IR)
for scanned scenes, a UnityPy-backed reader, a scene assembler, the
`manifest.json` schema, and a `vrscan scan` CLI command that inspects a data
folder and prints what it found (poses, parts, LOD levels, textures) without
writing anything out. No glTF export yet — that's Phase 1.

## Quickstart

```bash
uv sync
uv run vrscan doctor
uv run vrscan scan "/path/to/VRGirlz-21-Nat_Data"
```

`vrscan scan` is read-only: it opens `level0` and `sharedassets1.assets`
inside the data folder, walks the scene, and prints a table of poses, parts,
LOD levels, and texture metadata, plus peak memory (RSS) and wall time.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and
[`docs/DEV.md`](docs/DEV.md) for development notes.

## License

MIT, see [`LICENSE`](LICENSE).
