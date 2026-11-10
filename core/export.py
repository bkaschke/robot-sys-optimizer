"""Exports (M10): TCP poses per station (CSV), the optimized end-effector URDF, and a
dimensioned side view of the tool envelope (SVG). No viewer imports.

Frames: world (cell), base (robot base_link), flange (tool0). Lengths in the CSV are mm,
angles degrees, orientations as quaternions (w, x, y, z) and URDF rpy.
"""

from __future__ import annotations

import csv
import io
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from core.robot import Robot
from core.schema import Base, EndEffector, ExtensionValues, Solution, Station, Targets
from core.tool import ToolModel
from core.transforms import make_T, matrix_to_rpy, matrix_to_wxyz, pose_to_matrix, rpy_to_matrix

# --- station poses -----------------------------------------------------------------------------


def station_pose(base: Base, station: Station, T_world_base0: np.ndarray) -> np.ndarray:
    """Robot base pose (world) of a planned station.

    Carriers recompute it from the stored DOF values (exact for inverted gantry mounts);
    free-floor stations keep the base orientation they were planned with, or yaw only.
    """
    if base.mode == "movable" and station.carrier:
        from core.carrier import Carrier

        c = Carrier(base, T_world_base0)
        v = {d: station.carrier[d] for d in c.dofs if d in station.carrier}
        if len(v) == len(c.dofs):
            return c.T_world_base(v)
    R0 = np.asarray(T_world_base0, float)[:3, :3]
    yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
    R = rpy_to_matrix((0.0, 0.0, station.yaw - yaw0)) @ R0
    return make_T(R, station.xyz)


# --- CSV -------------------------------------------------------------------------------------------

CSV_COLUMNS = [
    "point_id",
    "group",
    "status",
    "reason",
    "station",
    "zone",
    "seq",
    "config",
    "carrier",
    "tcp",
    "base_x_mm",
    "base_y_mm",
    "base_z_mm",
    "base_yaw_deg",
    "tcp_world_x_mm",
    "tcp_world_y_mm",
    "tcp_world_z_mm",
    "tcp_world_qw",
    "tcp_world_qx",
    "tcp_world_qy",
    "tcp_world_qz",
    "tcp_base_x_mm",
    "tcp_base_y_mm",
    "tcp_base_z_mm",
    "tcp_base_roll_deg",
    "tcp_base_pitch_deg",
    "tcp_base_yaw_deg",
    "q1_deg",
    "q2_deg",
    "q3_deg",
    "q4_deg",
    "q5_deg",
    "q6_deg",
    "spin_deg",
    "tilt_deg",
    "standoff_mm",
    "clearance_mm",
    "manipulability",
    "limit_margin_deg",
]


def tcp_rows(
    solution: Solution,
    point_results: dict,
    targets: Targets,
    robot: Robot,
    tool: ToolModel,
    base: Base,
    T_world_base0: np.ndarray,
) -> list[dict]:
    """One row per target point, in station order (unreached points last). With zoning,
    rows follow station -> zone -> path order, and ``zone`` / ``seq`` / ``config`` are set.

    Poses are recomputed from the stored joint solution: T_world_tcp = T_world_base @ FK(q)
    @ T_flange_tcp, with the solution's tool geometry when it has one.
    """
    values = solution.tool_values
    groups = {p.id: p.group for p in targets.points}
    rows: list[dict] = []

    def key(kv):
        r = point_results.get(kv[0])
        z = r.zone if r is not None and r.zone is not None else -1
        s = r.seq if r is not None and r.seq is not None else kv[0]
        return (kv[1], z, s, kv[0])

    order = sorted(solution.assignment.items(), key=key)
    for pid, k in order:
        st = solution.stations[k]
        T_wb = station_pose(base, st, T_world_base0)
        r = point_results.get(pid)
        row = {c: "" for c in CSV_COLUMNS}
        carrier = ";".join(f"{d}={v:.4f}" for d, v in st.carrier.items())
        row.update(
            point_id=pid,
            group=groups.get(pid, ""),
            station=k + 1,
            carrier=carrier,
            base_x_mm=f"{T_wb[0, 3] * 1000:.2f}",
            base_y_mm=f"{T_wb[1, 3] * 1000:.2f}",
            base_z_mm=f"{T_wb[2, 3] * 1000:.2f}",
            base_yaw_deg=f"{np.degrees(np.arctan2(T_wb[1, 0], T_wb[0, 0])):.3f}",
        )
        if r is None or r.q is None:
            row.update(status="assigned", reason="joint solution not stored (re-run the plan)")
            rows.append(row)
            continue
        T_bt = robot.fk(r.q) @ tool.T_flange_tcp(r.tcp, values)
        T_wt = T_wb @ T_bt
        qw = matrix_to_wxyz(T_wt[:3, :3])
        rpy = np.degrees(matrix_to_rpy(T_bt[:3, :3]))
        if r.zone is not None:
            row.update(zone=r.zone + 1, seq=(r.seq or 0) + 1, config=r.config)
        row.update(
            status=r.status,
            tcp=r.tcp,
            tcp_world_x_mm=f"{T_wt[0, 3] * 1000:.3f}",
            tcp_world_y_mm=f"{T_wt[1, 3] * 1000:.3f}",
            tcp_world_z_mm=f"{T_wt[2, 3] * 1000:.3f}",
            tcp_world_qw=f"{qw[0]:.8f}",
            tcp_world_qx=f"{qw[1]:.8f}",
            tcp_world_qy=f"{qw[2]:.8f}",
            tcp_world_qz=f"{qw[3]:.8f}",
            tcp_base_x_mm=f"{T_bt[0, 3] * 1000:.3f}",
            tcp_base_y_mm=f"{T_bt[1, 3] * 1000:.3f}",
            tcp_base_z_mm=f"{T_bt[2, 3] * 1000:.3f}",
            tcp_base_roll_deg=f"{rpy[0]:.4f}",
            tcp_base_pitch_deg=f"{rpy[1]:.4f}",
            tcp_base_yaw_deg=f"{rpy[2]:.4f}",
            spin_deg=f"{r.spin_deg:.1f}",
            tilt_deg=f"{r.tilt_deg:.2f}",
            standoff_mm=f"{r.standoff * 1000:.1f}",
            clearance_mm=f"{min(r.clearance, 99.999) * 1000:.1f}",
            manipulability=f"{r.manipulability:.4f}",
            limit_margin_deg=f"{r.limit_margin_deg:.1f}",
        )
        for j, v in enumerate(np.degrees(r.q)):
            row[f"q{j + 1}_deg"] = f"{v:.4f}"
        rows.append(row)
    for pid in solution.unreached:
        r = point_results.get(pid)
        row = {c: "" for c in CSV_COLUMNS}
        row.update(
            point_id=pid,
            group=groups.get(pid, ""),
            status="unreachable",
            reason=getattr(r, "reason", "") if r is not None else "",
        )
        rows.append(row)
    return rows


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# --- tool URDF ---------------------------------------------------------------------------------------


def _origin(T: np.ndarray) -> str:
    xyz = " ".join(f"{v:.9g}" for v in T[:3, 3])
    rpy = " ".join(f"{v:.9g}" for v in matrix_to_rpy(T[:3, :3]))
    return f'<origin xyz="{xyz}" rpy="{rpy}"/>'


def tool_urdf(ee: EndEffector, values: ExtensionValues | None, head_urdf: Path | None) -> str:
    """URDF of the tool with the given geometry, rooted at ``flange`` (= robot tool0).

    Chain: flange -> mount -> extension segment links (cylinder + sphere caps)
    -> head root (the original head URDF's links/joints, copied) -> tcp_<name> frames.
    Head mesh paths stay relative to the head URDF's folder (copy that folder alongside).
    """
    tm = ToolModel(ee, None)
    v = tm.values(values)
    x = ee.extension
    out = [
        f'<robot name="{ee.name}_optimized">',
        '  <link name="flange"/>',
        '  <link name="mount"/>',
        f'  <joint name="flange_mount" type="fixed"><parent link="flange"/><child link="mount"/>'
        f"{_origin(tm.T_flange_mount)}</joint>",
    ]
    T_inv_mount = np.linalg.inv(tm.T_flange_mount)
    for k, (a, b) in enumerate(tm.extension_segments(v)):
        length = float(np.linalg.norm(b - a))
        z = (b - a) / length
        ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        xx = ref - (ref @ z) * z
        xx /= np.linalg.norm(xx)
        T_seg = make_T(np.column_stack([xx, np.cross(z, xx), z]), a)  # flange frame, z along segment
        r = x.radius
        geo = f'<geometry><cylinder radius="{r}" length="{length:.9g}"/></geometry>'
        cap = f'<geometry><sphere radius="{r}"/></geometry>'
        body = (
            f'<visual><origin xyz="0 0 {length / 2:.9g}"/>{geo}</visual>'
            f'<collision><origin xyz="0 0 {length / 2:.9g}"/>{geo}</collision>'
            f'<collision><origin xyz="0 0 0"/>{cap}</collision>'
            f'<collision><origin xyz="0 0 {length:.9g}"/>{cap}</collision>'
        )
        out.append(f'  <link name="extension_{k}">{body}</link>')
        out.append(
            f'  <joint name="mount_extension_{k}" type="fixed"><parent link="mount"/>'
            f'<child link="extension_{k}"/>{_origin(T_inv_mount @ T_seg)}</joint>'
        )
    T_mount_head = T_inv_mount @ tm.T_flange_head(v)
    head_root = "head"
    if head_urdf is not None:
        root = ET.fromstring(Path(head_urdf).read_text(encoding="utf-8"))
        children = {j.find("child").get("link") for j in root.findall("joint")}
        head_root = next(lk.get("name") for lk in root.findall("link") if lk.get("name") not in children)
        for el in root:
            if el.tag in ("link", "joint", "material"):
                out.append("  " + ET.tostring(el, encoding="unicode").strip())
    else:
        out.append(f'  <link name="{head_root}"/>')
    out.append(
        f'  <joint name="mount_head" type="fixed"><parent link="mount"/><child link="{head_root}"/>'
        f"{_origin(T_mount_head)}</joint>"
    )
    for t in ee.tcps:
        out.append(f'  <link name="tcp_{t.name}"/>')
        out.append(
            f'  <joint name="{head_root}_tcp_{t.name}" type="fixed"><parent link="{head_root}"/>'
            f'<child link="tcp_{t.name}"/>{_origin(pose_to_matrix(t.T_head_tcp))}</joint>'
        )
    out.append("</robot>")
    return "\n".join(out) + "\n"


def export_tool(
    ee: EndEffector, values: ExtensionValues | None, head_urdf: Path | None, out_dir: Path
) -> Path:
    """Write <out_dir>/<name>_optimized.urdf and copy the head's mesh folder next to it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ee.name}_optimized.urdf"
    path.write_text(tool_urdf(ee, values, head_urdf), encoding="utf-8")
    if head_urdf is not None:
        src = Path(head_urdf).parent
        for item in src.iterdir():
            if item.is_dir():
                shutil.copytree(item, out_dir / item.name, dirs_exist_ok=True)
    return path


# --- envelope sketch (SVG) -----------------------------------------------------------------------------


def envelope_svg(tool: ToolModel, values: ExtensionValues | None = None, width: int = 520) -> str:
    """Dimensioned side view of the tool in the bend plane (mount y-z after roll), mm.

    Shows the flange face, mount, extension capsules (radius), head silhouettes (convex
    hull of each link mesh, projected), TCPs, and the L / theta / d / TCP-offset values.
    """
    from scipy.spatial import ConvexHull

    v = tool.values(values)
    pose = tool.ee.extension.type == "pose"
    if pose:  # the plane that holds the head's sideways offset (else: its tilt direction)
        T_mh = tool.T_mount_head(v)
        side = T_mh[:2, 3] if np.linalg.norm(T_mh[:2, 3]) > 1e-6 else T_mh[:2, 2]
        phi = float(np.arctan2(-side[0], side[1])) if np.linalg.norm(side) > 1e-9 else 0.0
    else:
        phi = float(np.radians(v.roll_deg))
    T_plane = tool.T_flange_mount @ make_T(rpy_to_matrix((0, 0, phi)))
    T_pf = np.linalg.inv(T_plane)

    def proj(P: np.ndarray) -> np.ndarray:  # flange-frame points -> (horizontal = y, down = z) in mm
        Q = P @ T_pf[:3, :3].T + T_pf[:3, 3]
        return np.column_stack([Q[:, 1], Q[:, 2]]) * 1000

    shapes, pts_all = [], []
    r_mm = tool.ee.extension.radius * 1000
    for a, b in tool.extension_segments(v):
        p = proj(np.array([a, b]))
        shapes.append(("seg", p))
        pts_all += [p - r_mm, p + r_mm]
    T_fh = tool.T_flange_head(v)
    for hl in tool.links:
        V = hl.mesh.vertices @ (T_fh @ hl.T_head_link)[:3, :3].T + (T_fh @ hl.T_head_link)[:3, 3]
        p = proj(V)
        if len(p) >= 3:
            try:
                p = p[ConvexHull(p).vertices]
            except Exception:  # degenerate (flat) projection
                pass
        shapes.append(("hull", p))
        pts_all.append(p)
    tcps = [(t.name, proj(tool.T_flange_tcp(t.name, v)[None, :3, 3])[0]) for t in tool.ee.tcps]
    pts_all += [np.array([[-60.0, 0.0], [60.0, 0.0]])] + [np.array([p]) for _, p in tcps]
    allp = np.vstack(pts_all)
    lo, hi = allp.min(axis=0) - 40, allp.max(axis=0) + 40
    scale = (width - 220) / max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    height = int((hi[1] - lo[1]) * scale) + 20

    def xy(p):
        return (p[0] - lo[0]) * scale + 10, (p[1] - lo[1]) * scale + 10

    el = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    fx0, fy = xy((-60, 0))
    fx1, _ = xy((60, 0))
    el.append(
        f'<line x1="{fx0:.1f}" y1="{fy:.1f}" x2="{fx1:.1f}" y2="{fy:.1f}" stroke="#333" stroke-width="3"/>'
    )
    el.append(f'<text x="{fx1 + 4:.1f}" y="{fy + 4:.1f}" fill="#333">flange</text>')
    for kind, p in shapes:
        if kind == "seg":
            (x0, y0), (x1, y1) = xy(p[0]), xy(p[1])
            el.append(
                f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#5a8cdc" '
                f'stroke-width="{2 * r_mm * scale:.1f}" stroke-linecap="round" opacity="0.8"/>'
            )
        else:
            path = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(q) for q in p))
            el.append(f'<polygon points="{path}" fill="#c8ccd4" stroke="#555" stroke-width="1"/>')
    for name, p in tcps:
        cx, cy = xy(p)
        el.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#e03c3c"/>')
        el.append(f'<text x="{cx + 6:.1f}" y="{cy + 4:.1f}" fill="#e03c3c">TCP {name}</text>')
    if pose:
        lines = [
            f"head x y z = {v.x * 1000:.0f}, {v.y * 1000:.0f}, {v.z * 1000:.0f} mm",
            f"head rx ry rz = {v.rx_deg:.1f}, {v.ry_deg:.1f}, {v.rz_deg:.1f}°",
            f"connector {tool.length(v) * 1000:.0f} mm (spline)",
            f"capsule r = {r_mm:.0f} mm",
        ]
    else:
        lines = [
            f"L = {v.L * 1000:.1f} mm",
            f"θ = {v.theta_deg:.1f}°",
            f"d = {v.d * 1000:.1f} mm",
            f"roll = {v.roll_deg:.1f}°",
            f"capsule r = {r_mm:.0f} mm",
        ]
    for name, _ in tcps:
        off = tool.T_flange_tcp(name, v)[:3, 3] * 1000
        lines.append(f"TCP {name}: ({off[0]:.0f}, {off[1]:.0f}, {off[2]:.0f}) mm")
    for i, t in enumerate(lines):
        el.append(f'<text x="{width - 205}" y="{24 + 16 * i}" fill="#222">{t}</text>')
    el.append(
        f'<text x="10" y="{height - 6}" fill="#777">side view in the bend plane, flange at top, '
        f"tool axis pointing down</text>"
    )
    el.append("</svg>")
    return "\n".join(el)
