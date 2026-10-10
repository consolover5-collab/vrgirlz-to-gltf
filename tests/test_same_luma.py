"""`writer._same_luma`: dry and wet specular compared by (width, height,
sha256 of the luma), one decoded image at a time (design review 22.09 B3)."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vrscan.gltf.writer import _same_luma
from vrscan.ir import TextureRef
from vrscan.reader.unitypy_backend import UnityPyBackend

# Texture ids of the two specular maps — what the backend is asked by; the
# names only label the log lines.
IDS = {"dry": (0, 1), "wet": (0, 2)}


def _rgba(gray: int, alpha: int, size=(8, 8)) -> Image.Image:
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[..., :3] = gray
    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")


class _Backend:
    """Hands out a fresh image per call, like `UnityPyBackend`, and records
    which earlier images were still alive when the next one was asked for."""

    def __init__(self, images: dict[str, Image.Image]):
        self._images = {IDS[name]: img for name, img in images.items()}
        self._handed_out: list[weakref.ref] = []
        self.alive_at_request: list[int] = []

    def get_texture_image(self, texture_id):
        self.alive_at_request.append(sum(ref() is not None for ref in self._handed_out))
        if texture_id not in self._images:
            return None
        img = self._images[texture_id].copy()
        self._handed_out.append(weakref.ref(img))
        return img


def _ref(name: str) -> TextureRef:
    return TextureRef(role="specular", id=IDS[name], name=name, width=8, height=8, format="DXT5", byte_size=0)


def _compare(images):
    backend, log = _Backend(images), []
    return _same_luma(backend, _ref("dry"), _ref("wet"), log.append), backend, log


def test_same_luma_under_a_different_alpha_is_the_same():
    same, backend, log = _compare({"dry": _rgba(120, 204), "wet": _rgba(120, 23)})
    assert same is True
    assert log == []
    assert backend.alive_at_request == [0, 0]  # the dry image was gone before the wet one was decoded


def test_different_size_or_luma_is_not_the_same():
    assert _compare({"dry": _rgba(120, 255), "wet": _rgba(121, 255)})[0] is False
    assert _compare({"dry": _rgba(120, 255), "wet": _rgba(120, 255, size=(8, 4))})[0] is False
    # same luma bytes, other shape
    assert _compare({"dry": _rgba(120, 255, size=(4, 8)), "wet": _rgba(120, 255, size=(8, 4))})[0] is False


def test_a_missing_image_is_a_warning_and_false():
    for images, missing in (({"wet": _rgba(1, 1)}, "dry"), ({"dry": _rgba(1, 1)}, "wet")):
        same, _backend, log = _compare(images)
        assert same is False
        (line,) = log
        assert line.startswith("warning:") and f"specular {missing} has no readable image" in line
        assert "wet variant kept" in line


def test_unitypy_backend_keeps_no_decoded_image():
    """The sequence above only helps if the backend itself holds no copy:
    `UnityPyBackend.get_texture_image` parses and decodes afresh per call."""

    backend = UnityPyBackend()
    backend._texture_readers[IDS["dry"]] = SimpleNamespace(read=lambda: SimpleNamespace(image=_rgba(120, 204)))
    first = backend.get_texture_image(IDS["dry"])
    ref = weakref.ref(first)
    del first
    assert ref() is None
    assert backend.get_texture_image(IDS["dry"]) is not backend.get_texture_image(IDS["dry"])
