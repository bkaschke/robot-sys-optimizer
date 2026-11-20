"""Run tool-surface detectors over the synthetic mandrel set and/or a folder of samples.

    python scripts/tool_surface_eval.py --synthetic
    python scripts/tool_surface_eval.py path/to/samples --scale 0.001
    python scripts/tool_surface_eval.py --synthetic --detector mypkg.mod:detect

A detector is ``detect(mesh) -> bool mask per face``. Without --detector the two baselines
(normal_up, largest_smooth) are run. A samples folder holds one sub-folder per sample with
mandrel.(stl|step) and tool_surface.(stl|step).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mandrel_cases import iter_cases  # noqa: E402
from core.tool_surface_eval import BASELINES, load_sample, score  # noqa: E402


def resolve_detector(spec: str):
    if spec in BASELINES:
        return BASELINES[spec]
    if ":" not in spec:
        raise SystemExit(f"--detector must be 'module:function' or one of {sorted(BASELINES)}, got {spec!r}")
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def iter_samples(folder: Path, scale: float):
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        try:
            mesh, truth, info = load_sample(sub, scale=scale)
        except (FileNotFoundError, NotImplementedError, ValueError) as e:
            print(f"SKIP {sub.name}: {e}", file=sys.stderr)
            continue
        yield sub.name, mesh, truth, info


def run(cases, detectors: dict) -> list[dict]:
    rows = []
    for name, mesh, truth, info in cases:
        row = {
            "case": name,
            "faces": len(mesh.faces),
            "wt": info.get("watertight"),
            "hard": "hard" in info.get("tags", ()),
        }
        for dn, fn in detectors.items():
            t0 = time.perf_counter()
            try:
                pred = fn(mesh)
                s = score(pred, truth, mesh)
            except Exception as e:  # report, keep going
                print(f"FAILED {dn} on {name}: {type(e).__name__}: {e}", file=sys.stderr)
                s = {"iou": float("nan"), "precision": float("nan"), "recall": float("nan")}
            s["time"] = time.perf_counter() - t0
            row[dn] = s
        rows.append(row)
    return rows


def print_table(rows: list[dict], detectors: dict) -> None:
    for dn in detectors:
        print(f"\n== {dn} ==")
        print(f"{'case':26s}{'faces':>8s}{'wt':>4s}{'IoU':>7s}{'prec':>7s}{'rec':>7s}{'time_s':>8s}")
        acc = {"iou": [], "precision": [], "recall": [], "time": []}
        groups = {"EASY": [], "HARD": []}
        for r in rows:
            s = r[dn]
            tag = " *" if r["hard"] else ""
            print(
                f"{r['case']:26s}{r['faces']:8d}{'Y' if r['wt'] else 'N':>4s}"
                f"{s['iou']:7.3f}{s['precision']:7.3f}{s['recall']:7.3f}{s['time']:8.2f}{tag}"
            )
            groups["HARD" if r["hard"] else "EASY"].append(s)
        print("(* = hard case)")
        for label, ss in [*groups.items(), ("ALL", groups["EASY"] + groups["HARD"])]:
            if not ss:
                continue
            m = {k: sum(x[k] for x in ss) / len(ss) for k in acc}
            print(
                f"{'MEAN ' + label + f' (n={len(ss)})':26s}{'':8s}{'':>4s}{m['iou']:7.3f}"
                f"{m['precision']:7.3f}{m['recall']:7.3f}{m['time']:8.2f}"
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="folder of samples (sub-folder per sample)")
    ap.add_argument("--synthetic", action="store_true", help="run the synthetic mandrel set")
    ap.add_argument("--detector", action="append", help="module:function (repeatable); default: baselines")
    ap.add_argument("--scale", type=float, default=1.0, help="sample file units -> metres (mm files: 0.001)")
    a = ap.parse_args(argv)
    if not a.folder and not a.synthetic:
        ap.error("give a samples folder and/or --synthetic")
    dets = {s: resolve_detector(s) for s in (a.detector or list(BASELINES))}
    if a.synthetic:
        rows = run(((c.name, m, t, i) for c, m, t, i in iter_cases()), dets)
        print_table(rows, dets)
    if a.folder:
        rows = run(iter_samples(Path(a.folder), a.scale), dets)
        print_table(rows, dets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
