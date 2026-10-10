"""`UnityPyBackend.load()` keys the scene graph by a *global* object id
(`_gid`: the `(file index, path_id)` pair), not by bare path_id. A path_id is unique only
within one serialized file, and Lucid-style datasets load several — found
live 22.09: pose 41's `Shad` node (one file) overwrote pose 21's LOD
renderer GameObject (another file) at `game_objects[21]`, so lucid-21's
shadow part merged '21shad' with '41shad'.

The fake env below is the smallest reproduction: two files, each with a
GameObject at path_id 1 and a Transform at path_id 2, plus in file A a
LODGroup on GameObject 1 whose renderer (path_id 3) belongs to GameObject 4
— while file B also has a GameObject 4.

The same fakes cover Texture2D ids and what `load()` refuses up front: an
unknown Unity release, a missing streaming companion.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from vrscan.profiles import PROFILES, DatasetError
from vrscan.reader import unitypy_backend
from vrscan.reader.unitypy_backend import UnityPyBackend, _gid


class _File:
    def __init__(self, name: str) -> None:
        self.name = name
        self.unity_version = "5.3.4f1"
        self.readers: dict[int, _Reader] = {}


class _Reader:
    def __init__(self, file: _File, path_id: int, type_name: str, data: Any) -> None:
        self.assets_file = file
        self.path_id = path_id
        self.type = SimpleNamespace(name=type_name)
        self.byte_size = 0
        self._data = data
        file.readers[path_id] = self

    def read(self) -> Any:
        return self._data


class _Ptr:
    """A PPtr that resolves inside `file` (m_FileID == 0 semantics)."""

    def __init__(self, file: _File | None, path_id: int) -> None:
        self._file = file
        self.path_id = path_id

    def deref(self) -> _Reader:
        if self._file is None or self.path_id not in self._file.readers:
            raise FileNotFoundError("dangling")
        return self._file.readers[self.path_id]


def _vec(*xyz: float) -> SimpleNamespace:
    return SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2], w=xyz[3] if len(xyz) > 3 else None)


def _game_object(file: _File, path_id: int, name: str) -> _Reader:
    return _Reader(file, path_id, "GameObject", SimpleNamespace(m_Name=name))


def _transform(file: _File, path_id: int, go: int, father: _Ptr | None, children: list[_Ptr]) -> _Reader:
    data = SimpleNamespace(
        m_GameObject=_Ptr(file, go),
        m_LocalPosition=_vec(0.0, 0.0, 0.0),
        m_LocalRotation=_vec(0.0, 0.0, 0.0, 1.0),
        m_LocalScale=_vec(1.0, 1.0, 1.0),
        m_Father=father,
        m_Children=children,
    )
    return _Reader(file, path_id, "Transform", data)


def _build_env() -> SimpleNamespace:
    a, b = _File("level1"), _File("sharedassets1.assets")
    # File A: GameObject 1 "A-root" (transform 2) with a LODGroup whose
    # renderer 3 sits on GameObject 4 "A-part" (transform 5, child of root).
    _game_object(a, 1, "A-root")
    _transform(a, 2, 1, None, [_Ptr(a, 5)])
    _game_object(a, 4, "A-part")
    _transform(a, 5, 4, _Ptr(a, 2), [])
    _Reader(a, 3, "MeshRenderer", SimpleNamespace(m_GameObject=_Ptr(a, 4), m_Materials=[]))
    _Reader(
        a, 6, "LODGroup",
        SimpleNamespace(
            m_GameObject=_Ptr(a, 1),
            m_LODs=[SimpleNamespace(screenRelativeHeight=0.25, renderers=[SimpleNamespace(renderer=_Ptr(a, 3))])],
        ),
    )
    # File B: the same path_ids, different objects.
    _game_object(b, 1, "B-root")
    _transform(b, 2, 1, None, [_Ptr(b, 5)])
    _game_object(b, 4, "B-part")
    _transform(b, 5, 4, _Ptr(b, 2), [])
    objects = [r for f in (a, b) for r in f.readers.values()]
    return SimpleNamespace(objects=objects, files={a.name: a, b.name: b}, _a=a, _b=b)


@pytest.fixture
def fake_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("level0", "level1", "sharedassets1.assets"):
        (tmp_path / name).write_bytes(b"")
    env = _build_env()
    monkeypatch.setattr(unitypy_backend.UnityPy, "load", lambda *paths: env)
    return tmp_path


def test_same_path_id_in_two_files_stays_two_game_objects(fake_data_dir: Path) -> None:
    scene = UnityPyBackend(profile=PROFILES["vrgirlz-2016"]).load(fake_data_dir)
    names = sorted(go.name for go in scene.game_objects.values())
    assert names == ["A-part", "A-root", "B-part", "B-root"]
    # Hierarchy wired within each file, never across.
    by_name = {go.name: go for go in scene.game_objects.values()}
    assert by_name["A-part"].parent == by_name["A-root"].node_id
    assert by_name["B-part"].parent == by_name["B-root"].node_id
    assert by_name["A-root"].children == [by_name["A-part"].node_id]
    assert by_name["A-root"].node_id != by_name["B-root"].node_id


def test_lod_group_renderer_resolves_to_its_own_files_game_object(fake_data_dir: Path) -> None:
    scene = UnityPyBackend(profile=PROFILES["vrgirlz-2016"]).load(fake_data_dir)
    by_name = {go.name: go for go in scene.game_objects.values()}
    lod = by_name["A-root"].lod_group
    assert lod is not None
    (renderer_go,) = lod.levels[0].renderer_game_objects
    assert scene.game_objects[renderer_go].name == "A-part"


def test_gid_keeps_file_and_path_id_apart_for_any_int64_path_id() -> None:
    """The id is the `(file index, pathID)` pair itself, so nothing about it
    depends on a pathID's magnitude or sign — Unity pathIDs are signed 64-bit
    and an asset bundle can carry negative or >40-bit ones (design review 22.09
    point 6), which the previous `file_index << 40 | path_id` packing could
    only answer with a ValueError."""

    assert _gid(0, 7) == (0, 7)
    assert _gid(1, 7) == (1, 7)
    assert _gid(0, 1 << 40) != _gid(1, 0)
    assert _gid(0, -6203845182340987654) == (0, -6203845182340987654)


# --------------------------------------------------------------------------
# Texture2D ids: two files, each with a material whose `_MainTex` is a
# Texture2D called "Skin_CLR" — same name, same path_id, different pixels.
# Keyed by name, whichever was resolved last stood in for both.
# --------------------------------------------------------------------------


def _texture2d(file: _File, path_id: int, name: str, payload: bytes, color: tuple[int, ...],
               stream: SimpleNamespace | None = None) -> _Reader:
    data = SimpleNamespace(
        m_Name=name, m_Width=1, m_Height=1, m_TextureFormat=4,  # RGBA32
        m_MipCount=1, m_ImageCount=1, m_PlatformBlob=None, m_StreamData=stream,
        object_reader=SimpleNamespace(platform=5, version=(5, 3, 4, 1)),
        get_image_data=lambda: payload, image=Image.new("RGBA", (1, 1), color),
    )
    return _Reader(file, path_id, "Texture2D", data)


def _skinned_part(file: _File, material: str, payload: bytes, color: tuple[int, ...],
                  stream: SimpleNamespace | None = None) -> None:
    _game_object(file, 1, f"{file.name}-part")
    _transform(file, 2, 1, None, [])
    _Reader(file, 3, "MeshRenderer", SimpleNamespace(m_GameObject=_Ptr(file, 1), m_Materials=[_Ptr(file, 4)]))
    properties = SimpleNamespace(
        m_TexEnvs=[("_MainTex", SimpleNamespace(m_Texture=_Ptr(file, 5)))], m_Floats=[], m_Colors=[]
    )
    _Reader(file, 4, "Material", SimpleNamespace(m_Name=material, m_Shader=None, m_SavedProperties=properties))
    _texture2d(file, 5, "Skin_CLR", payload, color, stream)


@pytest.fixture
def same_named_textures_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("level0", "level1", "sharedassets1.assets"):
        (tmp_path / name).write_bytes(b"")
    a, b = _File("level1"), _File("sharedassets1.assets")
    _skinned_part(a, "A-skin", b"AAAA", (255, 0, 0, 255))
    _skinned_part(b, "B-skin", b"BBBB", (0, 0, 255, 255))
    env = SimpleNamespace(objects=[r for f in (a, b) for r in f.readers.values()])
    monkeypatch.setattr(unitypy_backend.UnityPy, "load", lambda *paths: env)
    return tmp_path


def test_same_named_textures_stay_two_resources_each_with_its_own_pixels(same_named_textures_dir: Path) -> None:
    backend = UnityPyBackend(profile=PROFILES["vrgirlz-2016"])
    scene = backend.load(same_named_textures_dir)
    assert sorted(t.name for t in scene.textures.values()) == ["Skin_CLR", "Skin_CLR"]
    a_id = scene.materials["A-skin"].textures["_MainTex"]
    b_id = scene.materials["B-skin"].textures["_MainTex"]
    assert a_id != b_id and {a_id, b_id} == set(scene.textures)
    assert scene.textures[a_id].id == a_id
    assert backend.get_texture_fingerprint(a_id) != backend.get_texture_fingerprint(b_id)
    assert backend.get_texture_image(a_id).getpixel((0, 0)) == (255, 0, 0, 255)
    assert backend.get_texture_image(b_id).getpixel((0, 0)) == (0, 0, 255, 255)


# --------------------------------------------------------------------------
# what `load()` refuses before converting anything
# --------------------------------------------------------------------------


def _one_part_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, unity_version: str = "5.3.4p1",
                  stream: SimpleNamespace | None = None) -> Path:
    for name in ("level0", "level1", "sharedassets1.assets"):
        (tmp_path / name).write_bytes(b"")
    a = _File("sharedassets1.assets")
    a.unity_version = unity_version
    _skinned_part(a, "A-skin", b"AAAA", (255, 0, 0, 255), stream)
    env = SimpleNamespace(objects=list(a.readers.values()))
    monkeypatch.setattr(unitypy_backend.UnityPy, "load", lambda *paths: env)
    return tmp_path


def test_an_unknown_unity_release_is_not_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = _one_part_dir(tmp_path, monkeypatch, unity_version="2019.4.1f1")
    with pytest.raises(DatasetError, match=r"Unity 2019\.4\.1f1: not supported"):
        UnityPyBackend().load(data_dir)


def test_a_release_only_the_prefix_matches_is_read_as_assumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = UnityPyBackend()
    backend.load(_one_part_dir(tmp_path, monkeypatch, unity_version="5.6.0f3"))
    assert (backend.profile, backend.support) == (PROFILES["vrgirlz-2016"], "assumed")
    assert "tentatively" in backend.support_note


def test_a_missing_streaming_companion_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = SimpleNamespace(path="sharedassets1.assets.resS", offset=0, size=4)
    data_dir = _one_part_dir(tmp_path, monkeypatch, stream=stream)
    with pytest.raises(DatasetError, match=r"missing: sharedassets1\.assets\.resS"):
        UnityPyBackend().load(data_dir)
    (data_dir / "sharedassets1.assets.resS").write_bytes(b"AAAA")
    assert UnityPyBackend().load(data_dir).textures


def test_a_folder_without_level0_is_not_a_data_folder(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="has no level0"):
        UnityPyBackend().load(tmp_path)
    with pytest.raises(DatasetError, match="is not a folder"):
        UnityPyBackend().load(tmp_path / "nowhere")
