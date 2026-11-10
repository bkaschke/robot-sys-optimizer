"""Generate the sample end effectors (URDF + STL meshes) and their tool-library entries.

    python scripts/make_sample_tools.py

Writes assets/tools/<name>/ and tools/<name>.yaml. Geometry is simple on purpose: these
are stand-ins until real tool CAD is available.
"""

import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.process import apply_preset  # noqa: E402
from core.schema import TCP, EndEffector, Extension, ExtensionValues, Pose  # noqa: E402
from core.tool import save_tool  # noqa: E402

LINK = """  <link name="{name}">
    <visual><geometry><mesh filename="meshes/{name}.stl"/></geometry></visual>
    <collision><geometry><mesh filename="meshes/{name}.stl"/></geometry></collision>
    <inertial>
      <origin xyz="{com}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>
"""
JOINT = """  <joint name="{parent}-{child}" type="fixed">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{xyz}" rpy="0 0 0"/>
  </joint>
"""


def _cyl(r: float, z0: float, z1: float) -> trimesh.Trimesh:
    m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=48)
    m.apply_translation((0, 0, (z0 + z1) / 2))
    return m


def _box(x: float, y: float, z0: float, z1: float) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(x, y, z1 - z0))
    m.apply_translation((0, 0, (z0 + z1) / 2))
    return m


def write_tool(name: str, links: list[tuple], joints: list[tuple]) -> Path:
    folder = ROOT / "assets" / "tools" / name
    (folder / "meshes").mkdir(parents=True, exist_ok=True)
    body = [f'<robot name="{name}">']
    for lname, mesh, mass, com in links:
        mesh.export(folder / "meshes" / f"{lname}.stl")
        body.append(LINK.format(name=lname, mass=mass, com=" ".join(map(str, com))))
    for parent, child, xyz in joints:
        body.append(JOINT.format(parent=parent, child=child, xyz=" ".join(map(str, xyz))))
    body.append("</robot>\n")
    path = folder / f"{name}.urdf"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def main() -> None:
    # Orbital sander: motor body 0..150 mm, 150 mm pad disc 20 mm thick below it.
    write_tool(
        "orbital_sander",
        [
            ("sander_body", _cyl(0.05, 0.0, 0.15), 2.5, (0, 0, 0.075)),
            ("pad_link", _cyl(0.075, 0.0, 0.02), 0.4, (0, 0, 0.01)),
        ],
        [("sander_body", "pad_link", (0, 0, 0.15))],
    )
    sander = EndEffector(
        name="orbital_sander",
        urdf="assets/tools/orbital_sander/orbital_sander.urdf",
        T_flange_mount=Pose(xyz=(0, 0, 0.02)),
        extension=Extension(
            type="bent",
            L=(0.05, 0.6),
            theta_deg=(0, 45),
            radius=0.04,
            nominal=ExtensionValues(L=0.15),
        ),
        tcps=[TCP(name="pad", T_head_tcp=Pose(xyz=(0, 0, 0.17)), contact_links=["pad_link"])],
    )
    save_tool(apply_preset(sander, "sanding"))

    # Line scanner: 80 x 200 x 120 mm housing; TCP on the sensor window.
    write_tool("scanner", [("scanner_body", _box(0.08, 0.2, 0.0, 0.12), 1.6, (0, 0, 0.06))], [])
    scanner = EndEffector(
        name="scanner",
        urdf="assets/tools/scanner/scanner.urdf",
        extension=Extension(L=(0.0, 0.4), radius=0.03, nominal=ExtensionValues(L=0.1)),
        tcps=[TCP(name="sensor", T_head_tcp=Pose(xyz=(0, 0, 0.12)))],
    )
    save_tool(apply_preset(scanner, "scanning"))
    print("wrote orbital_sander, scanner")


if __name__ == "__main__":
    main()
