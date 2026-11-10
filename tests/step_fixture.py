"""A STEP tool for the tests: a 60 x 60 x 20 mm plate with a 20 mm probe 120 mm long on top
(CadQuery), plus two named coordinate systems as CAD datum features export them."""

from pathlib import Path

DATUMS = """#9001 = CARTESIAN_POINT('',(0.,0.,130.));
#9002 = DIRECTION('',(0.,0.,1.));
#9003 = DIRECTION('',(1.,0.,0.));
#9004 = AXIS2_PLACEMENT_3D('CSYS_TCP',#9001,#9002,#9003);
#9005 = CARTESIAN_POINT('',(0.,0.,-10.));
#9006 = DIRECTION('',(0.,0.,-1.));
#9007 = DIRECTION('',(1.,0.,0.));
#9008 = AXIS2_PLACEMENT_3D('CSYS_FLANGE',#9005,#9006,#9007);
"""


def probe_step(folder: Path) -> Path:
    import cadquery as cq

    body = cq.Workplane("XY").box(60, 60, 20).faces(">Z").workplane().circle(10).extrude(120)
    path = folder / "probe.step"
    cq.exporters.export(body, str(path))
    text = path.read_text()
    i = text.rindex("ENDSEC;")  # the end of the DATA section
    path.write_text(text[:i] + DATUMS + text[i:])
    return path
