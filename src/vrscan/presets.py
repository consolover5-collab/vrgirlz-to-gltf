"""Quality tiers and headset presets — one source of truth, shared by the
Ф1 converter (which resolutions to encode) and `manifest.tiers`/`presets`.
`manifest.Tier` mirrors this shape field-for-field.

Ф1 note: tier `2k` ships from the start (design review finding #2 — a real
small tier is needed to check the Quest 2 `gpuBytes` budget, not just a
future promise). `8k` and KTX2 encoding (replacing `jpeg`) land in Ф2.
"""

from __future__ import annotations

TIERS: dict[str, dict[str, object]] = {
    "2k": {
        "label": "Quest 2",
        "baseColor": 2048,
        "normal": 2048,
        "orm": 1024,
        "format": "jpeg",
        "maxLod": 1,
    },
    "4k": {
        "label": "Quest 3 / PC VR",
        "baseColor": 4096,
        "normal": 4096,
        "orm": 2048,
        "format": "jpeg",
        "maxLod": 0,
    },
    "8k": {
        "label": "Ultra (PC)",
        "baseColor": 8192,
        "normal": 4096,
        "orm": 4096,
        "format": "jpeg",
        "maxLod": 0,
    },
}

# Headset/use case -> default tier id (key into TIERS). Quest 3 is the only
# headset validated so far (Ф0 VR-smoke); the rest are informed defaults.
# No preset for `8k` until Ф2 ships that tier — a preset must name a tier the
# manifest actually has (`cli._merge_manifest` drops any that doesn't).
PRESETS: dict[str, str] = {
    "quest2": "2k",
    "quest3": "4k",
    "pcvr": "4k",
}
