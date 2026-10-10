"""Unity TRS composition — the one place that turns a `SceneRec`'s local
transforms into the matrix that maps a node's local vertices into pose-root
space.

Lives outside `gltf/writer.py` because `assembler.py` needs the same
composition for the pose's `Pivot` node (a local `translation` is only the
right answer for a Pivot that is a direct child of the pose root — the reviewer
review 22.09, "Дефект: `_find_pivot` возвращает локальный pivot"), and the
writer's module pulls in pygltflib/textures that the assembler has no
business importing.

Nothing here flips X — Unity->glTF handedness is applied exactly once, in
`gltf/writer.py`, to both baked vertices and the manifest's pivot.
"""

from __future__ import annotations

import numpy as np

from vrscan.ir import Transform
from vrscan.reader.backend import NodeId, SceneRec, node_id_str


def quat_to_rotation_matrix(rotation: tuple[float, float, float, float], node: str = "") -> np.ndarray:
    """Rotation matrix of a (x, y, z, w) quaternion, normalized first.

    Unity stores quaternions at float precision and real scenes carry
    rounded values (0.70710678..., sometimes just 0.707) — building the
    matrix from an unnormalized quaternion leaves it slightly non-orthogonal
    and that error compounds over a chain (design review 22.09). A zero-norm
    or non-finite quaternion is not recoverable and is an explicit error,
    not a silently-identity rotation."""

    q = np.asarray(rotation, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        where = f" of node {node!r}" if node else ""
        raise ValueError(f"quat_to_rotation_matrix: degenerate rotation quaternion{where}: {tuple(rotation)}")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def trs_matrix(t: Transform, node: str = "") -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quat_to_rotation_matrix(t.rotation, node) @ np.diag(t.scale)
    m[:3, 3] = t.translation
    return m


def _node_name(scene: SceneRec, node_id: NodeId) -> str:
    go = scene.game_objects.get(node_id)
    return go.name if go is not None else node_id_str(node_id)


def _walk_up(
    scene: SceneRec, go_node_id: NodeId, stop_node_id: NodeId | None
) -> tuple[list[Transform], bool]:
    """Every local transform from `go_node_id` up to (not including)
    `stop_node_id`, plus whether the walk actually got there. Stops at a
    scene root (`parent is None`) otherwise. Raises on a missing parent or a
    cycle — both mean the source hierarchy is broken in a way no matrix can
    paper over."""

    chain: list[Transform] = []
    pid: NodeId | None = go_node_id
    seen: set[NodeId] = set()
    while pid is not None and pid != stop_node_id:
        if pid in seen:
            raise ValueError(
                f"compose_world_matrix: cycle in ancestor chain of node "
                f"node {node_id_str(go_node_id)} — revisited {_node_name(scene, pid)!r} ({node_id_str(pid)})"
            )
        seen.add(pid)
        go = scene.game_objects.get(pid)
        if go is None:
            raise ValueError(
                f"compose_world_matrix: missing parent GameObject {node_id_str(pid)} "
                f"in ancestor chain of node {node_id_str(go_node_id)}"
            )
        chain.append(go.transform)
        pid = go.parent
    return chain, pid == stop_node_id


def _product(scene: SceneRec, go_node_id: NodeId, chain: list[Transform]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    for t in reversed(chain):
        result = result @ trs_matrix(t, _node_name(scene, go_node_id))
    return result


def _scene_world_matrix(scene: SceneRec, go_node_id: NodeId) -> tuple[np.ndarray, NodeId]:
    """(matrix into scene-root space, node id of that scene root)."""

    chain, _ = _walk_up(scene, go_node_id, None)
    pid: NodeId = go_node_id
    while (go := scene.game_objects.get(pid)) is not None and go.parent is not None:
        pid = go.parent
    return _product(scene, go_node_id, chain), pid


def compose_world_matrix(scene: SceneRec, go_node_id: NodeId, root_node_id: NodeId) -> np.ndarray:
    """4x4 matrix mapping `go_node_id`'s local vertices into `root_node_id`'s
    space, by composing every ancestor's local TRS up to (not including) the
    pose root — established fact: mesh renderers are usually direct children
    of the root, but sofa pieces sit one level deeper under a `NNsofa` group,
    so this can't assume a single hop.

    A node whose parent chain never passes through the pose root is NOT an
    error by itself: real Unity 5.3 scenes hand a pose's own renderer to a
    node parented elsewhere (found live on lucid-21/ariel-45, whose shadow
    node `Shad` is a sibling of the pose root rather than a descendant).
    Such a node is resolved through the common scene root instead —
    `inv(W_root) @ W_node`, both chains walked to their own scene root,
    which must be the same one (design review 22.09). Genuinely broken
    hierarchies — a missing parent, a cycle, two disjoint scene roots, or a
    pose root whose own transform can't be inverted — still raise, naming
    the node: silently returning a partial matrix bakes the mesh into the
    wrong place with no visible error."""

    chain, reached_root = _walk_up(scene, go_node_id, root_node_id)
    if reached_root:
        return _product(scene, go_node_id, chain)

    world_node, node_scene_root = _scene_world_matrix(scene, go_node_id)
    if root_node_id not in scene.game_objects:
        raise ValueError(
            f"compose_world_matrix: node {_node_name(scene, go_node_id)!r} ({node_id_str(go_node_id)}) "
            f"is not a descendant of pose root {node_id_str(root_node_id)}, which is itself missing from the scene"
        )
    world_root, root_scene_root = _scene_world_matrix(scene, root_node_id)
    if node_scene_root != root_scene_root:
        raise ValueError(
            f"compose_world_matrix: node {_node_name(scene, go_node_id)!r} ({node_id_str(go_node_id)}) and pose "
            f"root {_node_name(scene, root_node_id)!r} ({node_id_str(root_node_id)}) live under different scene roots "
            f"({_node_name(scene, node_scene_root)!r} vs {_node_name(scene, root_scene_root)!r}) — "
            f"no common space to express one in the other"
        )
    return np.linalg.inv(check_invertible(world_root, _node_name(scene, root_node_id))) @ world_node


def check_invertible(matrix: np.ndarray, node: str) -> np.ndarray:
    """Return `matrix` unchanged, or raise if its 3x3 linear part can't be
    inverted meaningfully. Degeneracy is judged by the ratio of the smallest
    to the largest singular value, not by `det` against a fixed threshold —
    a uniformly tiny (e.g. millimetre-unit) transform has a minuscule
    determinant while being perfectly well-conditioned (design review
    22.09)."""

    linear = matrix[:3, :3]
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"transform of node {node!r} is not finite: {matrix[:3].tolist()}")
    singular = np.linalg.svd(linear, compute_uv=False)
    if singular[0] <= 0.0 or singular[-1] / singular[0] < 1e-8:
        raise ValueError(
            f"transform of node {node!r} is degenerate (singular values {singular.tolist()}) — "
            f"a flattened or zero-scale node can't be inverted for normals"
        )
    return matrix
