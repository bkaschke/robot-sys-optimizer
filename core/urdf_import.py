"""Import a user-supplied robot URDF (folder or .zip) into a self-contained folder.

Resolves ``package://pkg/...`` and ``file://`` mesh URIs to paths relative to the URDF,
so yourdfpy, Pinocchio and Viser all load it the same way. Zip extraction is guarded
against path traversal and decompression bombs.
"""

from __future__ import annotations

import io
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_UNZIPPED_BYTES = 500 * 1024 * 1024
MAX_FILES = 5000


class URDFImportError(ValueError):
    pass


@dataclass
class URDFInfo:
    path: Path
    name: str
    root_link: str
    leaf_link: str  # end of the longest chain: the default flange
    actuated_joints: list[str]


def safe_extract_zip(data: bytes, dest: Path) -> None:
    """Extract a zip into ``dest``, rejecting absolute/parent paths and oversized archives."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise URDFImportError("not a valid zip file") from e
    infos = zf.infolist()
    if len(infos) > MAX_FILES:
        raise URDFImportError(f"zip has more than {MAX_FILES} entries")
    if sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
        raise URDFImportError("zip expands to more than 500 MB")
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for info in infos:
        name = PurePosixPath(info.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts or (name.parts and ":" in name.parts[0]):
            raise URDFImportError(f"unsafe path in zip: {info.filename}")
        target = (dest / Path(*name.parts)).resolve()
        if not target.is_relative_to(dest):
            raise URDFImportError(f"unsafe path in zip: {info.filename}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _find_urdf(root: Path) -> Path:
    urdfs = sorted(root.rglob("*.urdf"), key=lambda p: (len(p.parts), str(p)))
    if not urdfs:
        if any(root.rglob("*.xacro")):
            raise URDFImportError("only .xacro found; convert it to a plain .urdf first")
        raise URDFImportError("no .urdf file found")
    return urdfs[0]


def _resolve_uri(uri: str, urdf_dir: Path, search_root: Path) -> str:
    """Map a mesh URI to a path relative to the URDF's folder."""
    if uri.startswith("package://"):
        pkg, _, rel = uri[len("package://") :].partition("/")
        candidates = [d for d in [search_root, *search_root.rglob(pkg)] if d.is_dir()]
        for base in candidates:
            for cand in (base / rel, base / pkg / rel):
                if cand.is_file():
                    return Path(_relpath(cand, urdf_dir)).as_posix()
        # fall back: match by file name anywhere in the import
        hits = list(search_root.rglob(PurePosixPath(rel).name))
        if len(hits) == 1:
            return Path(_relpath(hits[0], urdf_dir)).as_posix()
        raise URDFImportError(f"cannot resolve {uri}")
    if uri.startswith("file://"):
        name = PurePosixPath(uri[len("file://") :]).name
        hits = list(search_root.rglob(name))
        if len(hits) == 1:
            return Path(_relpath(hits[0], urdf_dir)).as_posix()
        raise URDFImportError(f"cannot resolve {uri} inside the imported files")
    if not (urdf_dir / uri).is_file():
        raise URDFImportError(f"mesh not found: {uri}")
    return uri


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target.resolve(), start.resolve())


def _chain_info(xml_text: str) -> tuple[str, str, str, list[str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise URDFImportError(f"URDF is not valid XML: {e}") from e
    if root.tag != "robot":
        raise URDFImportError("root element is not <robot>")
    links = [link.get("name") for link in root.findall("link")]
    children: dict[str, list[str]] = {}
    child_set = set()
    actuated = []
    for j in root.findall("joint"):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        children.setdefault(parent, []).append(child)
        child_set.add(child)
        if j.get("type") in ("revolute", "continuous", "prismatic"):
            actuated.append(j.get("name"))
    roots = [link for link in links if link not in child_set]
    if len(roots) != 1:
        raise URDFImportError(f"URDF must have exactly one root link, found {roots}")

    def deepest(link: str) -> tuple[int, str]:
        kids = children.get(link, [])
        if not kids:
            return 0, link
        d, leaf = max(deepest(k) for k in kids)
        return d + 1, leaf

    return root.get("name") or "robot", roots[0], deepest(roots[0])[1], actuated


def import_urdf(source: Path, dest: Path) -> URDFInfo:
    """Copy/resolve a URDF folder (or the folder a zip was extracted to) into ``dest``."""
    source, dest = source.resolve(), dest.resolve()
    if source != dest:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    urdf = _find_urdf(dest)
    text = urdf.read_text(encoding="utf-8")
    text = re.sub(
        r'filename="([^"]+)"',
        lambda m: f'filename="{_resolve_uri(m.group(1), urdf.parent, dest)}"',
        text,
    )
    name, root_link, leaf, actuated = _chain_info(text)
    if not actuated:
        raise URDFImportError("URDF has no movable joints")
    urdf.write_text(text, encoding="utf-8")
    return URDFInfo(urdf, name, root_link, leaf, actuated)


def inspect_urdf(path: Path) -> URDFInfo:
    """Chain info of an already-resolved URDF file."""
    name, root_link, leaf, actuated = _chain_info(Path(path).read_text(encoding="utf-8"))
    return URDFInfo(Path(path), name, root_link, leaf, actuated)


def import_urdf_zip(data: bytes, dest: Path) -> URDFInfo:
    if dest.exists():
        shutil.rmtree(dest)
    safe_extract_zip(data, dest)
    return import_urdf(dest, dest)
