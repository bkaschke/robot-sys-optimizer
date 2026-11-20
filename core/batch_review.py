"""Batch review page: ONE self-contained ``review.html`` (no viser, no ui, no network).

One card per job with the two offline pictures (``ui.batch_materialize.job_pictures``, embedded
as data URIs), status, selection rule and confidence, warnings and numbers. Inline CSS (light and
dark), a few lines of inline JS for the filter row (without scripts every card stays visible),
no external fonts, links or images. Order: needs_review, failed, partial / cancelled, done by
ascending confidence (manual jobs last), then pending / running.
"""

from __future__ import annotations

import base64
import html
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from core import batch as bt

FILTERS = (
    ("all", "All", None),
    ("needs_review", "Needs review", ("needs_review",)),
    ("failed", "Failed", ("failed", "partial", "cancelled")),
    ("done", "Done", ("done",)),
)
_RANK = {"needs_review": 0, "failed": 1, "partial": 2, "cancelled": 2, "done": 3, "running": 4, "pending": 5}
_LABEL = {"needs_review": "Needs review"}


def job_order(jobs: list[bt.BatchJob]) -> list[bt.BatchJob]:
    """Review order (see the module docstring); stable within a group."""

    def key(j: bt.BatchJob):
        auto = j.selection_rule == "tool_surface" and j.confidence is not None
        conf = j.confidence if auto else 2.0  # manual (or unknown) jobs last among the done ones
        return (_RANK.get(j.status, 5), conf if j.status == "done" else 0.0)

    return sorted(jobs, key=key)


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _href(rel: str) -> str:
    return quote(rel.replace("\\", "/"), safe="/:._-~")


def _picture(job: bt.BatchJob, view: str, batch_dir: Path) -> str:
    s = job.summary or {}
    rel = (s.get("review") or {}).get(view)
    reason = s.get("review_note") or "no picture yet (open the review page from the batch panel to render it)"
    if rel:
        p = batch_dir / rel
        try:
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            return f'<img alt="{view} view of {_e(job.name or job.id)}" src="data:image/png;base64,{data}">'
        except OSError:
            reason = f"picture file {rel} is missing"
    return f'<div class="nopic">{_e(reason)}</div>'


def _confidence(job: bt.BatchJob) -> str:
    if job.selection_rule != "tool_surface":
        return f'<span class="rule">manual selection ({len(job.selection)} faces)</span>'
    if job.confidence is None:
        return '<span class="rule">auto selection, not detected yet</span>'
    pct = round(100 * job.confidence)
    low = job.confidence < job.min_confidence
    return (
        f'<span class="rule">auto, confidence <b>{pct}%</b></span>'
        f'<span class="bar{" low" if low else ""}" title="minimum {round(100 * job.min_confidence)}%">'
        f'<i style="width:{max(0, min(100, pct))}%"></i></span>'
    )


def _facts(job: bt.BatchJob) -> str:
    s = job.summary or {}
    rows = []
    if s.get("stations"):
        rows.append(("Reached", f"{s.get('reached', '')} / {s.get('total', '')}"))
        rows.append(("Stations", s["stations"]))
        if s.get("cost") is not None:
            rows.append(("Cost", f"{s['cost']:.2f}"))
    if job.run_s:
        rows.append(("Run time", f"{job.run_s:.0f} s"))
    return "".join(f"<div><dt>{k}</dt><dd>{_e(v)}</dd></div>" for k, v in rows)


def _card(job: bt.BatchJob, batch_dir: Path) -> str:
    s = job.summary or {}
    warns = [*job.warnings]
    if job.error:
        warns.append(job.error)
    notes = "".join(f"<li>{_e(w)}</li>" for w in warns)
    links = []
    if s.get("report"):
        links.append(f'<a href="{_href(s["report"])}">Report</a>')
    folder = job.out_dir or job.id
    if (batch_dir / folder).is_dir():
        links.append(f'<a href="{_href(folder)}/">Project folder</a>')
    detect = f'<p class="detect">{_e(job.detect_summary)}</p>' if job.detect_summary else ""
    name = job.name or job.id
    ident = f" <small>{_e(job.id)}</small>" if name != job.id else ""
    return (
        f'<article class="card" data-status="{_e(job.status)}" id="{_e(job.id)}">'
        f'<div class="pics">{_picture(job, "iso", batch_dir)}{_picture(job, "top", batch_dir)}</div>'
        f'<div class="body"><header><h2>{_e(name)}{ident}</h2>'
        f'<span class="pill {_e(job.status)}">{_e(_LABEL.get(job.status, job.status))}</span></header>'
        f'<p class="part">{_e(job.part)}</p><div class="sel">{_confidence(job)}</div>{detect}'
        f"{'<ul class=warn>' + notes + '</ul>' if notes else ''}"
        f"<dl>{_facts(job)}</dl>"
        f"{'<nav>' + ' '.join(links) + '</nav>' if links else ''}</div></article>"
    )


CSS = """
:root{--bg:#f4f5f7;--card:#fff;--fg:#1d2330;--mute:#5f6877;--line:#d9dde4;--ok:#1b7f3b;--okbg:#dff3e5;
--bad:#b3261e;--badbg:#fbe3e1;--warn:#8a5300;--warnbg:#fdecc8;--info:#2457a6;--infobg:#e1ebfa;--idle:#555e6d;
--idlebg:#e8eaee;--ph:#eef0f4}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1d222b;--fg:#e6e9ef;--mute:#9aa3b2;
--line:#323a47;--ok:#6fd08c;--okbg:#17321f;--bad:#ff8a80;--badbg:#3b1b19;--warn:#f3c26b;--warnbg:#3b2d10;--info:#8db6ff;
--infobg:#18294a;--idle:#b4bcc9;--idlebg:#2a303b;--ph:#252b35}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,"Segoe UI",Arial,sans-serif}
main{max-width:1280px;margin:0 auto;padding:20px 16px 40px}
h1{margin:0 0 4px;font-size:1.5rem}.meta{color:var(--mute);margin:0 0 12px}
.counts{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px;padding:0;list-style:none}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 16px}
.filters button{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:4px 12px;cursor:pointer}
.filters button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;
break-inside:avoid;page-break-inside:avoid}
.card[hidden]{display:none}
.pics{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
.pics img,.nopic{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#fff}
.nopic{background:var(--ph);color:var(--mute);font-size:.85rem;padding:12px;display:flex;
align-items:center;justify-content:center;text-align:center}
.body{padding:10px 14px 14px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
h2{margin:0;font-size:1.05rem}h2 small{color:var(--mute);font-weight:400}
.part{margin:0;color:var(--mute);font-size:.85rem;overflow-wrap:anywhere}
.pill{border-radius:999px;padding:1px 10px;font-size:.8rem;font-weight:600;white-space:nowrap;
background:var(--idlebg);color:var(--idle)}
.pill.done{background:var(--okbg);color:var(--ok)}.pill.failed{background:var(--badbg);color:var(--bad)}
.pill.needs_review,.pill.partial,.pill.cancelled{background:var(--warnbg);color:var(--warn)}
.pill.running{background:var(--infobg);color:var(--info)}
.sel{display:flex;align-items:center;gap:10px;margin:6px 0}
.bar{display:inline-block;width:90px;height:8px;background:var(--idlebg);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}.bar.low i{background:var(--warn)}
.detect{margin:4px 0;color:var(--mute);font-size:.85rem}
.warn{margin:6px 0;padding:6px 10px 6px 26px;background:var(--warnbg);color:var(--warn);
border-radius:6px;font-size:.85rem}
dl{display:flex;flex-wrap:wrap;gap:4px 18px;margin:8px 0 0;font-variant-numeric:tabular-nums}
dl div{display:flex;gap:6px}dt{color:var(--mute)}dd{margin:0;font-weight:600}
nav{margin-top:8px;display:flex;gap:14px}a{color:var(--info)}
@media (max-width:560px){.pics{grid-template-columns:1fr}}
@media print{body{background:#fff}.filters{display:none}.grid{display:block}.card{margin-bottom:10px}}
"""

JS = """
(function(){var b=document.querySelectorAll('.filters button'),c=document.querySelectorAll('.card');
b.forEach(function(x){x.addEventListener('click',function(){var s=x.dataset.f.split(' ');
b.forEach(function(y){y.setAttribute('aria-pressed',y===x)});
c.forEach(function(k){k.hidden=s[0]!=='all'&&s.indexOf(k.dataset.status)<0})})})})();
"""


def review_html(manifest: bt.BatchManifest, batch_dir, name: str = "") -> str:
    """The whole page as one string (pictures embedded; no external URLs)."""
    batch_dir = Path(batch_dir)
    title = name or batch_dir.resolve().name or "Batch"
    jobs = job_order(manifest.jobs)
    count = Counter(j.status for j in jobs)
    pills = "".join(
        f'<li><span class="pill {k}">{_e(_LABEL.get(k, k))} {n}</span></li>' for k, n in count.items()
    )
    buttons = "".join(
        f'<button type="button" data-f="{" ".join(st) if st else "all"}" '
        f'aria-pressed="{str(k == "all").lower()}">'
        f"{label} ({len(jobs) if st is None else sum(count[x] for x in st)})</button>"
        for k, label, st in FILTERS
    )
    cards = "\n".join(_card(j, batch_dir) for j in jobs) or "<p>The batch has no jobs.</p>"
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Batch review: {_e(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>Batch review: {_e(title)}</h1>"
        f'<p class="meta">{len(jobs)} job(s); cell study {_e(manifest.cell_project or "-")}; '
        f"page written {when}. Pictures are offline renders: highlighted = the selected tool surface "
        "(secondary candidates in orange).</p>"
        f'<ul class="counts">{pills}</ul><div class="filters">{buttons}</div>'
        f'<section class="grid">{cards}</section></main><script>{JS}</script></body></html>'
    )


def write_review(manifest: bt.BatchManifest, batch_dir) -> Path:
    """Write ``<batch>/review.html`` atomically and return its path."""
    batch_dir = Path(batch_dir)
    p = batch_dir / "review.html"
    tmp = p.with_name("review.html.tmp")
    tmp.write_text(review_html(manifest, batch_dir), encoding="utf-8")
    os.replace(tmp, p)
    return p
