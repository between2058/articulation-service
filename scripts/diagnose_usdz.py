"""
Diagnose material / texture bindings inside a produced USDZ.

Usage:
    python scripts/diagnose_usdz.py outputs/robot_xxxxxxxx.usdz

Prints:
    - Stage up-axis, default prim
    - Every Mesh prim:
        * has primvars:st ?  how many values ? interpolation ?
        * bound material path ?
    - Every Material prim:
        * surface output connected to which shader ?
        * UsdPreviewSurface inputs (diffuseColor: value + connection ?)
        * UsdUVTexture.file (asset path) + st connection
        * UsdPrimvarReader_float2.varname
    - Lists assets inside the .usdz archive
"""

import sys
import zipfile
from pathlib import Path

from pxr import Usd, UsdGeom, UsdShade, Sdf


def dump_archive(path: Path) -> None:
    print("=" * 70)
    print(f"USDZ archive: {path}")
    print("=" * 70)
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:50s}  {info.file_size:>10d} bytes")


def dump_meshes(stage: Usd.Stage) -> None:
    print("\n--- Meshes ---")
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = prim.GetPath()
        print(f"\nMesh: {path}")

        # primvars:st
        st = prim.GetAttribute("primvars:st")
        if st and st.IsValid():
            val = st.Get()
            interp = st.GetMetadata("interpolation")
            n = len(val) if val else 0
            print(f"  primvars:st           : {n} values, interpolation={interp}, typeName={st.GetTypeName()}")
        else:
            print("  primvars:st           : MISSING  <-- texture cannot sample UV")

        # Material binding
        mat, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            print(f"  bound material        : {mat.GetPath()}")
        else:
            print("  bound material        : NONE     <-- no material bound")


def dump_materials(stage: Usd.Stage) -> None:
    print("\n--- Materials ---")
    found = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        found = True
        mat = UsdShade.Material(prim)
        print(f"\nMaterial: {prim.GetPath()}")

        surf_out = mat.GetSurfaceOutput()
        if surf_out:
            srcs = surf_out.GetConnectedSources()
            if srcs and srcs[0]:
                src = srcs[0][0]
                print(f"  surface -> {src.source.GetPath()}.{src.sourceName}")
            else:
                print("  surface output       : NOT CONNECTED  <-- material has no shader")

        for child in prim.GetChildren():
            if not child.IsA(UsdShade.Shader):
                continue
            sh = UsdShade.Shader(child)
            sid = sh.GetIdAttr().Get()
            print(f"  Shader {child.GetName()} id={sid}")
            for inp in sh.GetInputs():
                val = inp.Get()
                conn_sources = inp.GetConnectedSources()
                conn_str = ""
                if conn_sources and conn_sources[0]:
                    src = conn_sources[0][0]
                    conn_str = f"  <-- connected to {src.source.GetName()}.{src.sourceName}"
                print(f"    input {inp.GetBaseName():15s} = {val!r:40s}{conn_str}")

    if not found:
        print("  <no Material prims found — texture cannot show>")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: diagnose_usdz.py <path.usdz|path.usda>")
        return 1
    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        print(f"not found: {path}")
        return 1

    if path.suffix == ".usdz":
        dump_archive(path)

    stage = Usd.Stage.Open(str(path))
    if not stage:
        print("failed to open stage")
        return 1

    print(f"\nupAxis       : {UsdGeom.GetStageUpAxis(stage)}")
    print(f"default prim : {stage.GetDefaultPrim().GetPath()}")

    dump_meshes(stage)
    dump_materials(stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
