"""Vertex-coloured meshes for the part overlays (R1b): the heatmap and the station areas.

Viser 1.1.1 has no per-vertex colour on ``add_mesh_simple``; its GLB node does (the client's
GLTFLoader turns a ``COLOR_0`` attribute into ``material.vertexColors``). trimesh's own GLB
export writes no material, and the loader's default material is fully metallic and one-sided
(nearly black without an environment map), so the GLB is written here with a material that
matches ``add_mesh_simple`` (MeshStandardMaterial: metalness 0, roughness 1, double sided).

``NodeCache`` keeps the last few drawn overlays as hidden scene nodes, keyed by what they
show, so switching back to one only toggles visibility instead of re-uploading the mesh.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import OrderedDict
from collections.abc import Callable, Hashable

import numpy as np

from ui.display_lod import overlay_lift


def srgb_to_linear_u8(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..255 colours -> linear 0..255 (glTF vertex colours are linear; three.js converts
    ``add_mesh_simple``'s sRGB colour the same way, so both draw the same shade)."""
    c = np.asarray(rgb, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.clip(np.round(lin * 255), 0, 255).astype(np.uint8)


def _pad(b: bytes, fill: bytes = b"\0") -> bytes:
    return b + fill * (-len(b) % 4)


def glb(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, opacity: float = 1.0) -> bytes:
    """A one-mesh GLB: float32 positions, uint32 triangles, per-vertex sRGB ``colors`` (n, 3)
    uint8, stored as linear RGBA uint8. ``opacity`` < 1 blends (alphaMode BLEND)."""
    V = np.ascontiguousarray(vertices, np.float32)
    F = np.ascontiguousarray(faces, np.uint32).reshape(-1)
    C = np.empty((len(V), 4), np.uint8)
    C[:, :3] = srgb_to_linear_u8(colors)
    C[:, 3] = 255
    parts = [F.tobytes(), V.tobytes(), C.tobytes()]
    offs, blob = [], b""
    for p in parts:
        offs.append(len(blob))
        blob += _pad(p)
    lo, hi = (V.min(0), V.max(0)) if len(V) else (np.zeros(3), np.zeros(3))
    mat = {
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, float(opacity)],
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": True,
    }
    if opacity < 1.0:
        mat["alphaMode"] = "BLEND"
    doc = {
        "asset": {"version": "2.0", "generator": "rso ui.colour_mesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 1, "COLOR_0": 2}, "indices": 0, "material": 0}]}
        ],
        "materials": [mat],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offs[0], "byteLength": len(parts[0]), "target": 34963},
            {"buffer": 0, "byteOffset": offs[1], "byteLength": len(parts[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offs[2], "byteLength": len(parts[2]), "target": 34962},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": int(F.size), "type": "SCALAR"},
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(V),
                "type": "VEC3",
                "min": [float(x) for x in lo],
                "max": [float(x) for x in hi],
            },
            {"bufferView": 2, "componentType": 5121, "normalized": True, "count": len(V), "type": "VEC4"},
        ],
    }
    js = _pad(json.dumps(doc, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


def lifted_patch(mesh, faces: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices lifted ``offset`` along the whole mesh's vertex normals, faces re-indexed,
    used vertex indices of ``mesh``) of the sub-mesh ``faces``; vertices are shared."""
    used, inv = np.unique(mesh.faces[faces], return_inverse=True)
    V = mesh.vertices[used] + offset * mesh.vertex_normals[used]
    return V, inv.reshape(-1, 3), used


def vertex_colours(face_rgb: np.ndarray, faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Per-vertex colour = mean of the colours of its faces (``faces`` index the vertices)."""
    acc = np.zeros((n_vertices, 3))
    cnt = np.zeros(n_vertices)
    for k in range(3):
        np.add.at(acc, faces[:, k], face_rgb)
        np.add.at(cnt, faces[:, k], 1)
    return np.round(acc / np.maximum(cnt, 1)[:, None]).astype(np.uint8)


def show_patch(scene, cache: NodeCache, prefix: str, mesh, faces, face_rgb, offset: float, opacity=1.0):
    """Show the faces ``faces`` of ``mesh``, lifted ``offset`` along the normals and coloured
    per face by ``face_rgb`` (blended to per-vertex colours), as ONE GLB node under ``prefix``.
    Keyed by a hash of the exact geometry and colours: the same picture again only toggles
    visibility in ``cache``. Returns the handle."""
    V, F, used = lifted_patch(mesh, faces, overlay_lift(mesh, offset))  # clear of a simplified copy
    C = vertex_colours(np.asarray(face_rgb, float), F, len(used))
    V32, F32 = V.astype(np.float32), F.astype(np.uint32)
    h = hashlib.blake2b(digest_size=16)
    for a in (V32, F32, C):
        h.update(np.ascontiguousarray(a).data)
    h.update(repr((len(V32), len(F32), float(opacity))).encode())
    key = (prefix, h.hexdigest())
    return cache.show(key, lambda: scene.add_glb(f"{prefix}/m{key[1][:12]}", glb(V32, F32, C, opacity)))


def alive(h) -> bool:
    return h is not None and not h._impl.removed


class NodeCache:
    """Scene nodes keyed by what they show; at most one shown at a time, the last ``size``
    kept (hidden) so showing one of them again only sends a visibility update."""

    def __init__(self, size: int = 4):
        self.size = size
        self._nodes: OrderedDict[Hashable, object] = OrderedDict()
        self.shown: Hashable | None = None

    def show(self, key: Hashable, build: Callable[[], object]):
        """Show the node for ``key`` (built by ``build()`` when not cached or gone); hide the
        rest. Returns its handle."""
        h = self._nodes.pop(key, None)
        if not alive(h):
            h = build()
        elif not h.visible:
            h.visible = True
        self._nodes[key] = h
        for k, o in self._nodes.items():
            if k != key and alive(o) and o.visible:
                o.visible = False
        while len(self._nodes) > self.size:
            _, old = self._nodes.popitem(last=False)
            _remove(old)
        self.shown = key
        return h

    def hide(self) -> None:
        for o in self._nodes.values():
            if alive(o) and o.visible:
                o.visible = False
        self.shown = None

    def clear(self) -> None:
        for o in self._nodes.values():
            _remove(o)
        self._nodes.clear()
        self.shown = None

    def __len__(self) -> int:
        return len(self._nodes)


def _remove(h) -> None:
    if alive(h):
        h.remove()
