"""Self-contained HTML summary report (M10). Inline CSS, inline SVG, base64 PNG images; no
external requests, so the file can be mailed or printed to PDF from any browser."""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

UR_NOTICE = (
    "UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' "
    "Terms and Conditions for Use of Graphical Documentation."
)

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px auto;max-width:1100px;color:#1d1f23;
background:#fff;padding:0 16px;line-height:1.4}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
.meta{color:#666;font-size:13px}.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.kpi{border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:120px}
.kpi b{display:block;font-size:22px}.kpi span{color:#666;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:6px 0}
th,td{border-bottom:1px solid #eee;padding:4px 6px;text-align:right}th{background:#f6f7f9}
td:first-child,th:first-child{text-align:left}.bar{background:#5a8cdc;height:10px;display:inline-block}
.ok{color:#1f8a43}.warn{color:#b07800}.bad{color:#c62828}.note{color:#666;font-size:12px}
img{max-width:100%;border:1px solid #ddd;border-radius:6px}.scroll{max-height:420px;overflow:auto}
@media print{.scroll{max-height:none}}
@media (prefers-color-scheme: dark){body{background:#16181c;color:#e6e8eb}th{background:#23262c}
th,td{border-color:#2c3038}.kpi,h2,img{border-color:#2c3038}.meta,.note,.kpi span{color:#9aa0a8}
.ok{color:#4cc27a}.warn{color:#e0a83a}.bad{color:#ef6b6b}svg{border-radius:6px}}
"""


def png_data_uri(image: np.ndarray) -> str:
    """RGB(A) uint8 array -> data URI (uses trimesh's bundled PIL)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class ReportInput:
    title: str
    robot: str
    tool: str
    process: str
    base_mode: str
    solution: object  # schema.Solution
    rows: list[dict]  # core.export.tcp_rows
    station_poses: list[np.ndarray]
    envelope_svg: str = ""
    images: list[tuple[str, str]] = field(default_factory=list)  # (caption, data URI)
    length_samples: list = field(default_factory=list)  # opt.tool_geometry.LSample
    notes: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def _e(x) -> str:
    return html.escape(str(x))


def build_report(r: ReportInput) -> str:
    sol = r.solution
    n_total = len(sol.assignment) + len(sol.unreached)
    status = [row["status"] for row in r.rows]
    n_marg = status.count("marginal")
    tv = sol.tool_values
    parts = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(r.title)}</title><style>{CSS}</style></head><body>",
        f"<h1>{_e(r.title)}</h1><div class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
        f"robot {_e(r.robot)} · tool {_e(r.tool)} ({_e(r.process)}) · base {_e(r.base_mode)}</div>",
        "<div class='kpis'>",
        f"<div class='kpi'><b>{len(sol.stations)}</b><span>stations</span></div>",
        f"<div class='kpi'><b class='{'ok' if not sol.unreached else 'bad'}'>"
        f"{len(sol.assignment)}/{n_total}</b><span>points reached</span></div>",
        f"<div class='kpi'><b class='{'warn' if n_marg else 'ok'}'>{n_marg}</b><span>marginal</span></div>",
        f"<div class='kpi'><b>{sol.cost:.2f}</b><span>cost J</span></div>",
    ]
    if tv is not None:
        parts.append(f"<div class='kpi'><b>{tv.L * 1000:.0f} mm</b><span>tool length L</span></div>")
    parts.append("</div>")

    for caption, uri in r.images:
        parts.append(
            f"<figure><img src='{uri}' alt='{_e(caption)}'><figcaption class='note'>{_e(caption)}"
            "</figcaption></figure>"
        )

    parts.append("<h2>Cost breakdown</h2><table><tr><th>term</th><th>weighted value</th><th></th></tr>")
    total = max(sol.cost, 1e-9)
    for k, v in sorted(sol.breakdown.items(), key=lambda kv: -kv[1]):
        parts.append(
            f"<tr><td>{_e(k)}</td><td>{v:.3f}</td><td style='text-align:left'>"
            f"<span class='bar' style='width:{max(1, 300 * v / total):.0f}px'></span></td></tr>"
        )
    parts.append("</table>")

    parts.append(
        "<h2>Stations</h2><table><tr><th>station</th><th>x (mm)</th><th>y (mm)</th><th>z (mm)</th>"
        "<th>yaw (°)</th><th>carrier</th><th>points</th></tr>"
    )
    for k, (st, T) in enumerate(zip(sol.stations, r.station_poses, strict=True)):
        n = sum(1 for s in sol.assignment.values() if s == k)
        car = (
            ", ".join(
                f"{d} {v * (57.29578 if d == 'yaw' else 1000):.0f}{'°' if d == 'yaw' else ' mm'}"
                for d, v in st.carrier.items()
            )
            or "-"
        )
        parts.append(
            f"<tr><td>S{k + 1}</td><td>{T[0, 3] * 1000:.0f}</td><td>{T[1, 3] * 1000:.0f}</td>"
            f"<td>{T[2, 3] * 1000:.0f}</td><td>{np.degrees(np.arctan2(T[1, 0], T[0, 0])):.1f}</td>"
            f"<td>{_e(car)}</td><td>{n}</td></tr>"
        )
    parts.append("</table>")

    if sol.zones:
        switches = len(sol.zones) - len({z.station for z in sol.zones})
        jumps = sum(z.wrap_jumps for z in sol.zones)
        parts.append(
            f"<h2>Joint-configuration zones</h2><p class='note'>{len(sol.zones)} zones, "
            f"{switches + jumps} configuration changes ({switches} zone switches, {jumps} wrist jumps &gt; "
            "180°). Configuration = shoulder (S), elbow (E), wrist (W) branch.</p>"
            "<table><tr><th>zone</th><th>station</th><th>configuration</th><th>points</th>"
            "<th>wrist jumps</th><th>path (point ids)</th></tr>"
        )
        for i, z in enumerate(sol.zones):
            ids = ", ".join(str(p) for p in z.points[:40]) + (" …" if len(z.points) > 40 else "")
            parts.append(
                f"<tr><td>Z{i + 1}</td><td>S{z.station + 1}</td><td>{_e(z.config)}</td>"
                f"<td>{len(z.points)}</td><td>{z.wrap_jumps}</td>"
                f"<td style='text-align:left'>{_e(ids)}</td></tr>"
            )
        parts.append("</table>")

    if r.envelope_svg:
        parts.append("<h2>Tool geometry (envelope sketch)</h2>" + r.envelope_svg)
    if r.length_samples:
        parts.append(
            "<h2>Coverage vs tool length</h2><table><tr><th>L (mm)</th><th>full coverage</th>"
            "<th>reached</th><th>stations</th></tr>"
        )
        for s in r.length_samples:
            parts.append(
                f"<tr><td>{s.L * 1000:.1f}</td><td class='{'ok' if s.feasible else 'bad'}'>"
                f"{'yes' if s.feasible else 'no'}</td><td>{s.reached}/{s.total}</td>"
                f"<td>{s.stations}</td></tr>"
            )
        parts.append("</table>")

    cols = [
        "point_id",
        "station",
        *(["zone", "seq", "config"] if sol.zones else []),
        "status",
        "tcp",
        "tcp_world_x_mm",
        "tcp_world_y_mm",
        "tcp_world_z_mm",
        "tilt_deg",
        "clearance_mm",
        "torque_util",
        "limit_margin_deg",
        "reason",
    ]
    parts.append(
        "<h2>Points</h2><div class='scroll'><table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    )
    for row in r.rows:
        cls = {"reachable": "ok", "marginal": "warn", "unreachable": "bad"}.get(row["status"], "")
        parts.append(
            "<tr>"
            + "".join(f"<td class='{cls if c == 'status' else ''}'>{_e(row.get(c, ''))}</td>" for c in cols)
            + "</tr>"
        )
    parts.append("</table></div>")

    if r.settings:
        parts.append("<h2>Settings</h2><table>")
        for k, v in r.settings.items():
            parts.append(f"<tr><td>{_e(k)}</td><td style='text-align:left'>{_e(v)}</td></tr>")
        parts.append("</table>")
    notes = [*r.notes, UR_NOTICE]
    parts.append("<h2>Notes</h2><ul class='note'>" + "".join(f"<li>{_e(n)}</li>" for n in notes) + "</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)
