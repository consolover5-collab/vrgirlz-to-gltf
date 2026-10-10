"""The standalone-material pass wraps an ObjectReader in `_FakePPtr` so the
same resolver code path (keyed by the dereferenced reader's global id) can
process it — it only has to look like a PPtr: `.path_id` + `.deref()`."""

from types import SimpleNamespace

from vrscan.reader.unitypy_backend import _FakePPtr


def test_fake_pptr_derefs_to_the_wrapped_reader() -> None:
    obj = SimpleNamespace(path_id=42)
    ptr = _FakePPtr(obj)
    assert ptr.path_id == 42
    assert ptr.deref() is obj
