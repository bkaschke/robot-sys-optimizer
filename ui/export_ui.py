"""Results tab → Export (M10): TCP-pose CSV, optimized tool URDF (zip with meshes), and the
HTML report for the selected solution. Files go to <project>/export/ and are also sent to
the browser that clicked as downloads."""

from __future__ import annotations

import io
import logging
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import viser

from core import export as ex
from core.report import PdfError
from ui import report_make as rm
from ui import report_views as rv
from ui.part_heatmap import tolerance_for  # noqa: F401  (re-exported; tests import it from here)
from ui.plan_ids import key as plan_key
from ui.session import Session
from ui.widgets import notify

log = logging.getLogger(__name__)

# v10 T21: Program format label -> core.robot_program format (None = from the robot)
PROGRAM_FORMATS = {
    "Auto": None,
    "URScript (UR)": "urscript",
    "TP / .ls (FANUC)": "fanuc",
    "RAPID (ABB)": "rapid",
}


class ExportPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        selected: Callable[[], object],
        length_samples: Callable[[object], list] | None = None,
    ):
        self.s = session
        self.server = server
        self.stale: list[str] = []  # components changed, when exported after 'Export anyway'
        self.selected = selected
        self.length_samples = length_samples or (lambda sol: [])
        gui = server.gui
        self.folder = gui.add_folder("Export")
        with self.folder:
            self.screenshot = gui.add_checkbox(
                "Include 3D views in report",
                True,
                hint="framed views with the grid and gizmos hidden; the camera is restored afterwards",
            )
            self.view_boxes = {  # R3: which views the report gets
                v: gui.add_checkbox(f"Report view: {v}", v in rv.DEFAULT_VIEWS) for v in rv.ORDER
            }
            self.csv = gui.add_button("TCP poses (CSV)")
            self.uframes = gui.add_button(
                "Userframes (CSV)", hint="robot base -> part frame for every stop, in this robot's format"
            )
            self.urdf = gui.add_button("Optimized tool URDF (zip)")
            self.prog_format = gui.add_dropdown(  # v10 T21
                "Program format", tuple(PROGRAM_FORMATS), hint="Auto: the robot's own language"
            )
            self.program = gui.add_button("Robot program", hint="from the built paths (Paths & simulation)")
            self.report_format = gui.add_dropdown(  # v9 #10: one report button
                "Report format", ("PDF", "HTML"), hint="PDF: the HTML report printed by Edge headless"
            )
            self.report = gui.add_button("Export report")
            from core.step_export import FLANGES

            self.step = gui.add_button(
                "Tool (STEP)", hint="extension tube + flange plate (CadQuery)"
            )  # v8 T17
            self.flange = gui.add_dropdown(
                "Flange", tuple(FLANGES), hint="ISO 9409-1 pattern of the STEP plate"
            )
            self.package = gui.add_button(
                "Everything (zip)",
                hint="handover package: CSVs, report, tool URDF, built paths and project.json with a README",
            )
            self.gate_md = gui.add_markdown("", visible=False)  # why the exports are disabled (T16)
            self.info = gui.add_markdown("_Files are saved to the project folder's export/ directory._")
        self.csv.on_click(lambda e: self._run(e, self.export_csv))
        self.uframes.on_click(lambda e: self._run(e, self.export_userframes))
        self.urdf.on_click(lambda e: self._run(e, self.export_urdf))
        self.program.on_click(lambda e: self._run(e, self.export_program))
        self.report.on_click(
            lambda e: self._run(
                e, self.export_pdf if self.report_format.value == "PDF" else self.export_report
            )
        )
        self.step.on_click(lambda e: self._run(e, self.export_step))
        self.package.on_click(lambda e: self._run(e, self.export_package))
        self.tube_od = lambda: None  # app.py: Tool geometry -> Tube OD

    # --- helpers ---------------------------------------------------------------------------------

    @property
    def out_dir(self) -> Path:
        return self.s.project_dir / "export"

    def _run(self, event, fn, confirmed: bool = False) -> None:
        sol = self.selected()
        if sol is None:
            return notify(event, "Nothing to export", "Plan stations first and select a solution")
        from ui.staleness import solution_changes

        changes = solution_changes(self.s, sol)
        client = getattr(event, "client", None)
        if changes and not confirmed and client is not None:
            return self._ask(event, fn, changes)
        self.stale = list(changes) if changes else []
        try:
            path, data = fn(sol, getattr(event, "client", None))
        except (OSError, ValueError, KeyError) as err:
            return notify(event, "Export failed", str(err))
        except Exception as err:  # R14: never fail silently; the cause goes to the log
            log.exception("export failed")
            return notify(
                event, "Export failed", f"{type(err).__name__}: {str(err)[:200]} (nothing was exported)"
            )
        self.info.content = f"Saved `{path}`"
        client = getattr(event, "client", None)
        if client is not None:
            client.send_file_download(path.name, data)
        notify(event, "Exported", str(path), ok=True)

    def _ask(self, event, fn, changes: list[str]) -> None:
        """Out-of-date result: ask before exporting (modal for the clicking browser)."""
        with event.client.gui.add_modal("Result is out of date") as modal:
            event.client.gui.add_markdown(
                f"The setup changed since this result was computed (**{', '.join(changes)}**). "
                "The export would describe the old setup. Re-run the optimizer, or export anyway."
            )
            go = event.client.gui.add_button("Export anyway", color="red")
            stop = event.client.gui.add_button("Cancel")

        @go.on_click
        def _(e) -> None:
            modal.close()
            self._run(e, fn, confirmed=True)

        stop.on_click(lambda e: modal.close())

    def _tool(self, sol):
        return rm.tool_of(self.s)

    def rows(self, sol) -> list[dict]:
        return rm.rows(self.s, sol)

    # --- exports -----------------------------------------------------------------------------------

    def export_csv(self, sol, client=None) -> tuple[Path, bytes]:
        return rm.write_text(self.out_dir, "tcp_poses.csv", rm.make_csv(self.s, sol, self.stale))

    def export_userframes(self, sol, client=None) -> tuple[Path, bytes]:
        """Robot base -> part frame per stop, in the loaded robot's frame convention."""
        return rm.write_text(self.out_dir, "userframes.csv", rm.make_userframes(self.s, sol))

    def export_program(self, sol, client=None) -> tuple[Path, bytes]:
        """v10 T21: URScript / FANUC TP / ABB RAPID from the selected plan's built paths."""
        from core.robot_program import FORMATS, default_format, program

        s, sim = self.s, getattr(self, "sim", None)
        if s.robot is None or s.part is None or s.tool_model is None:
            raise ValueError("needs a robot, a tool and a part")
        if sim is None or sim.plan is None or sim.key != plan_key(sol) or not sim.plan.samples:
            raise ValueError("build the paths of this plan first (Results -> Paths & simulation)")
        fmt = PROGRAM_FORMATS[self.prog_format.value] or default_format(
            s.robot_name or "", s.robot.spec.frame_convention
        )
        if fmt is None:  # v12 #3: no guess for a robot of unknown language
            raise ValueError(
                f"the program language of {s.robot_name or 'this robot'} is not known: "
                "choose a Program format (URScript, TP or RAPID)"
            )
        tcp = next((p.tcp for p in sim.plan.samples if p.tcp), s.tool.tcps[0].name)
        carrier = s.carrier() if s.base.mode == "movable" else None
        text = program(fmt, sim.plan, s.tool_model.T_flange_tcp(tcp), s.part.T_world_part, carrier, s.cycle)
        data = text.encode("utf-8")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"rso_path{FORMATS[fmt]}"
        path.write_bytes(data)
        return path, data

    def export_step(self, sol, client=None) -> tuple[Path, bytes]:
        """v8 T17: the selected result's tool (its optimized values, else the nominal) as STEP."""
        from core.step_export import tool_step

        tool = self.s.tool
        if tool is None:
            raise ValueError("no tool loaded")
        try:
            import cadquery  # noqa: F401
        except ImportError as err:
            msg = (
                "STEP export is not included in this build; URDF export works"
                if getattr(sys, "frozen", False)
                else "STEP export needs CadQuery (pip install cadquery); URDF export works without"
            )
            raise ValueError(msg) from err
        path = tool_step(
            tool,
            sol.tool_values,
            self.out_dir / f"{tool.name}_tool.step",
            self.tube_od(),
            flange=self.flange.value,
        )
        return path, path.read_bytes()

    def export_urdf(self, sol, client=None) -> tuple[Path, bytes]:
        s = self.s
        tool = s.tool
        if tool is None:
            raise ValueError("no tool loaded")
        head = s.resolve(tool.urdf) if tool.urdf else None
        folder = self.out_dir / f"{tool.name}_optimized"
        ex.export_tool(tool, sol.tool_values, head, folder)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in folder.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(folder.parent).as_posix())
        zpath = self.out_dir / f"{tool.name}_optimized.zip"
        zpath.write_bytes(buf.getvalue())
        return zpath, buf.getvalue()

    def sections(self, sol) -> list[tuple[str, str]]:
        """v8 T19: the report's extra sections, as HTML (from the GUI's own texts)."""
        return rm.sections(self.s, sol, getattr(self, "sim", None), getattr(self, "frame", None))

    def export_pdf(self, sol, client=None) -> tuple[Path, bytes]:
        """v8 T19: the HTML report printed to PDF by Edge headless."""
        from core.report import html_to_pdf

        html_path, _ = self.export_report(sol, client)
        pdf = self.out_dir / "report.pdf"
        try:
            html_to_pdf(html_path, pdf)
        except PdfError as err:
            raise ValueError(str(err)) from err
        return pdf, pdf.read_bytes()

    def export_package(self, sol, client=None) -> tuple[Path, bytes]:
        """Every export of the selected plan in one zip with a README (ui.handover)."""
        from ui import handover as ho

        steps = [
            ("TCP poses (CSV)", lambda: self.export_csv(sol, client)),
            ("Userframes (CSV)", lambda: self.export_userframes(sol, client)),
            ("HTML report", lambda: self.export_report(sol, client)),
            ("Tool URDF", lambda: self.export_urdf(sol, client)),
        ]
        sim = getattr(self, "sim", None)
        if sim is not None and sim.plan is not None and sim.key == plan_key(sol) and self.s.part is not None:
            from core.paths import path_csv

            text = path_csv(sim.plan, self.s.part.T_world_part)
            steps.append(("Path CSV", lambda: rm.write_text(self.out_dir, "paths.csv", text)))
            steps.append(("Robot program", lambda: self.export_program(sol, client)))  # v10 T21
        steps.append(("project.json", lambda: ho.project_step(self.s, self.out_dir)))
        path, data, failed = ho.build(
            self.out_dir, self.s.project_dir.name, steps, ho.plan_lines(self.s, sol, self.stale)
        )
        if failed:
            log.warning("handover package without: %s", "; ".join(f"{w} ({why})" for w, why in failed))
        return path, data

    def margins_clearance(self) -> float:
        return rm.margins_clearance(self.s)

    def geometry(self) -> tuple[list[dict], str | None]:
        return rm.geometry(self.s)

    def export_report(self, sol, client=None) -> tuple[Path, bytes]:
        images, notes = [], []
        if self.screenshot.value and client is not None:  # no browser (tests, scripts): no views, no note
            views = tuple(v for v, b in self.view_boxes.items() if b.value)
            shots = rv.capture(self.server, client, self.s, sol, views=views)
            images, notes = list(shots), list(shots.notes)  # a failure leaves a note; the report still works
        text = rm.make_report(
            self.s,
            sol,
            images=images,
            stale=self.stale,
            sim=getattr(self, "sim", None),
            frame=getattr(self, "frame", None),
            length_samples=self.length_samples(sol),
            notes=notes,
        )
        return rm.write_text(self.out_dir, "report.html", text)
