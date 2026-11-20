# PyInstaller spec for the Robot Cell Optimizer (Windows, folder mode). Run it through
# scripts/build_exe.py, which sets PYTHONNOUSERSITE=1 (the user site-packages must not leak in),
# puts <env>/Library/bin on PATH and writes the stage folder that holds _version.py.
#   RSO_NO_CAD=1    build without CadQuery / OCP / VTK (about 500 MB smaller; no STEP import/export)
#   RSO_ONEFILE=1   one-file exe (rejected: unpacks ~1 GB on every start; for experiments only)
#   RSO_STAGE=dir   folder with the generated _version.py
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

HERE = Path(SPECPATH)
SRC = Path(os.environ.get("RSO_SRC", HERE.parent))  # the repo root
STAGE = os.environ.get("RSO_STAGE")
ONEFILE = os.environ.get("RSO_ONEFILE") == "1"
NO_CAD = os.environ.get("RSO_NO_CAD") == "1"

# --- read-only resources, at the bundle root (the code finds them via Path(__file__).parents[1])
datas = [
    (str(SRC / "assets"), "assets"),
    (str(SRC / "tools"), "tools"),  # the tool library (tools/*.yaml)
    (str(SRC / "examples"), "examples"),  # Open -> examples
    (str(SRC / "docs" / "USER_GUIDE.html"), "docs"),
]
seed = SRC / "cache" / "reachability"  # prebuilt maps (CI), copied to the user cache on first use
if seed.is_dir() and any(seed.glob("*.npz")):
    datas.append((str(seed), "cache/reachability"))
if STAGE and (Path(STAGE) / "_version.py").is_file():  # core.provenance reads it (no git in the exe)
    datas.append((str(Path(STAGE) / "_version.py"), "."))
datas += collect_data_files("viser")  # client/build/index.html (self-contained web client)
datas += collect_data_files("trimesh")  # resources/ (templates, schema, units)
datas += collect_data_files("yourdfpy")
datas += collect_data_files("collada")  # resources/schema-1.4.1.xml (robot meshes are .dae)
datas += collect_data_files("pinocchio")
datas += collect_data_files("ortools")
if not NO_CAD:
    datas += collect_data_files("cadquery")
    datas += collect_data_files("ezdxf")  # fonts / resources used at import

hiddenimports = []
for pkg in ("core", "opt", "ui"):  # ui.jobs resolves "module:function" by name in the worker
    hiddenimports += collect_submodules(pkg)
hiddenimports += collect_submodules("viser")
hiddenimports += collect_submodules("ortools.sat")
hiddenimports += ["collada", "dateutil.parser", "lxml.etree", "rtree", "embreex", "manifold3d"]

binaries = []
binaries += collect_dynamic_libs("coal")
binaries += collect_dynamic_libs("pinocchio")
binaries += collect_dynamic_libs("rtree")

excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "PyQt5", "PyQt6",
    "PySide2", "PySide6", "sphinx", "PyInstaller", "pip", "setuptools.command", "trame",
    "trame_vuetify", "trame_vtk", "wslink", "sqlalchemy.testing", "torch",
]
if NO_CAD:
    excludes += ["cadquery", "OCP", "vtk", "vtkmodules", "casadi", "ezdxf", "nlopt", "multimethod"]

a = Analysis(
    [str(HERE / "launch.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    # numba cache (@njit(cache=True)): load this module from its .py file, so numba uses NUMBA_CACHE_DIR
    # (from the PYZ its code has a relative file name and numba falls back to %LOCALAPPDATA%\numba)
    module_collection_mode={"core.ik.offset_wrist_nb": "py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
name = "RobotCellOptimizer"
if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=name, console=True, upx=False)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=name, console=True, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name=name, upx=False)
