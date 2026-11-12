"""Results tab → Export (M10): TCP-pose CSV, optimized tool URDF (zip with meshes), and the
HTML report for the selected solution. Files go to <project>/export/ and are also sent to
the browser that clicked as downloads."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import viser

from core import export as ex
from core.report import ReportInput, build_report, png_data_uri
from core.robot import Robot
from ui.session import Session
from ui.widgets import notify


class ExportPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        selected: Callable[[], object],
        length_samples: Callable[[object], list] | None = None,
    ):
        self.s = session
        self.selected = selected
        self.length_samples = length_samples or (lambda sol: [])
        gui = server.gui
        self.folder = gui.add_folder("Export")
        with self.folder:
            self.screenshot = gui.add_checkbox("Include 3D view snapshot in report", True)
            self.csv = gui.add_button("TCP poses (CSV)")
            self.uframes = gui.add_button(
                "Userframes (CSV)", hint="robot base -> part frame for every stop, in this robot's format"
            )
            self.urdf = gui.add_button("Optimized tool URDF (zip)")
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
            self.gate_md = gui.add_markdown("", visible=False)  # why the exports are disabled (T16)
            self.info = gui.add_markdown("_Files are saved to the project folder's export/ directory._")
        self.csv.on_click(lambda e: self._run(e, self.export_csv))
        self.uframes.on_click(lambda e: self._run(e, self.export_userframes))
        self.urdf.on_click(lambda e: self._run(e, self.export_urdf))
        self.report.on_click(
            lambda e: self._run(
                e, self.export_pdf if self.report_format.value == "PDF" else self.export_report
            )
        )
        self.step.on_click(lambda e: self._run(e, self.export_step))
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
        try:
            path, data = fn(sol, getattr(event, "client", None))
        except (OSError, ValueError, KeyError) as err:
            return notify(event, "Export failed", str(err))
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
        tm = self.s.tool_model
        if tm is None:
            raise ValueError("no tool loaded")
        return tm

    def rows(self, sol) -> list[dict]:
        s = self.s
        if s.robot is None:
            raise ValueError("no robot in this study (tab 1 Robot & cell)")
        return ex.tcp_rows(
            sol,
            s.result_points.get(sol.label, {}),
            s.targets,
            Robot(s.robot.spec),
            self._tool(sol),
            s.base,
            s.T_world_base,
        )

    # --- exports -----------------------------------------------------------------------------------

    def export_csv(self, sol, client=None) -> tuple[Path, bytes]:
        data = ex.rows_to_csv(self.rows(sol)).encode("utf-8")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "tcp_poses.csv"
        path.write_bytes(data)
        return path, data

    def export_userframes(self, sol, client=None) -> tuple[Path, bytes]:
        """Robot base -> part frame per stop, in the loaded robot's frame convention."""
        from core.userframe import userframe_csv, userframe_rows

        s = self.s
        if s.robot is None or s.part is None:
            raise ValueError("needs a robot and a part")
        conv = s.robot.spec.frame_convention
        rows = userframe_rows(sol, s.base, s.T_world_base, s.part.T_world_part, conv)
        data = userframe_csv(rows, conv).encode("utf-8")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "userframes.csv"
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
            raise ValueError(
                "STEP export needs CadQuery (pip install cadquery); URDF export works without"
            ) from err
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
        from core.report import md_html
        from ui.log_panel import unreached_report
        from ui.summary_card import card

        s, out = self.s, []
        out.append(("Summary", md_html(card(s, sol))))
        cyc = [f"Estimated cycle time: **{sol.cycle_s:.0f} s**" if sol.cycle_s is not None else ""]
        sim = getattr(self, "sim", None)
        if (
            sim is not None
            and sim.plan is not None
            and sim.label == sol.label
            and getattr(sim, "cycle", None)
        ):
            c = sim.cycle
            parts = ", ".join(f"{k} {v:.0f} s" for k, v in c.items() if k != "total" and v > 0.05)
            cyc.append(f"Cycle time of the built paths: **{c['total']:.0f} s** ({parts}).")
        if any(cyc):
            out.append(("Cycle time", md_html("  \n".join(x for x in cyc if x))))
        tol = getattr(s, "tolerance", None)
        if tol is not None and tol.label == sol.label:
            from ui.tolerance_ui import summary

            out.append(("Tolerance check", md_html(summary(tol))))
        msg = unreached_report(s.result_points.get(sol.label, {}), sol.unreached, sol.label)
        out.append(("Unreached points", md_html(msg) if msg else "<p>Every point is reached.</p>"))
        return out

    def export_pdf(self, sol, client=None) -> tuple[Path, bytes]:
        """v8 T19: the HTML report printed to PDF by Edge headless."""
        from core.report import html_to_pdf

        html_path, _ = self.export_report(sol, client)
        pdf = self.out_dir / "report.pdf"
        if not html_to_pdf(html_path, pdf):
            raise ValueError("no Edge / Chrome found to print the PDF; the HTML report was written")
        return pdf, pdf.read_bytes()

    def export_report(self, sol, client=None) -> tuple[Path, bytes]:
        s = self.s
        tm = self._tool(sol)
        rows = self.rows(sol)
        images = []
        notes = []
        if self.screenshot.value and client is not None:
            try:
                img = client.camera.get_render(720, 1280, transport_format="png", timeout=10.0)
                images.append(("3D view at export time", png_data_uri(img)))
            except Exception as err:  # renderer unavailable / timed out: the report still works
                notes.append(f"3D snapshot not included ({type(err).__name__}).")
        # per-point notes (if any) go into the report
        flagged = sorted(
            {n for r in s.result_points.get(sol.label, {}).values() for n in getattr(r, "notes", [])}
        )
        notes += flagged
        if not s.result_points.get(sol.label):
            notes.append(
                "Per-point joint solutions are not stored for this solution (loaded from a project): "
                "re-run the plan for the full point table."
            )
        tool = s.tool
        report = ReportInput(
            title=f"Cell plan: {sol.label}",
            robot=s.robot.spec.name,
            tool=tool.name if tool else "-",
            process=tool.process if tool else "-",
            base_mode=f"{s.base.mode}" + (f" ({s.base.carrier})" if s.base.mode == "movable" else ""),
            solution=sol,
            rows=rows,
            station_poses=[ex.station_pose(s.base, st, s.T_world_base) for st in sol.stations],
            envelope_svg=ex.envelope_svg(tm, sol.tool_values),
            images=images,
            length_samples=self.length_samples(sol),
            notes=notes,
            sections=self.sections(sol),
            settings={
                "part": s.part.mesh_path if s.part else "-",
                "targets": f"{len(s.targets.points)} points in {len(s.targets.groups)} group(s)",
                "standoff min / preferred (m)": f"{s.base.standoff_m.carrier_part_min} / "
                f"{s.base.standoff_m.carrier_part_preferred}",
                "tool values": sol.tool_values.model_dump()
                if sol.tool_values
                else tool.extension.nominal.model_dump(),
            },
        )
        data = build_report(report).encode("utf-8")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "report.html"
        path.write_bytes(data)
        return path, data
