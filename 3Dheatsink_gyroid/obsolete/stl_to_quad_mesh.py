#!/usr/bin/env python3
"""
stl_to_quad_mesh.py — Python driver for stl_to_quad_cgal.cpp.

Compiles the CGAL C++ program on-the-fly (using g++ against the system CGAL
headers) and converts gyroid_surface_lattice.stl → gyroid_lattice_quad.obj.

The C++ pipeline (stl_to_quad_cgal.cpp):
  1. Read STL → CGAL::Surface_mesh  (soup API, handles non-manifold input)
  2. Repair polygon soup, orient, build Surface_mesh, remove degeneracies
  3. Discrete Gaussian curvature (angle-defect) per vertex
  4a. Stage-1 simplification (edge-length cost, fast): N → COARSE_FACES
  4b. Stage-2 Garland-Heckbert QEM (curvature-aware): COARSE_FACES → TARGET_FACES
      — High-curvature vertices are pinned (never collapsed)
      — QEM quadric error inherently preserves saddle/ridge geometry
  5. Isotropic remeshing (CGAL PMP, edge length = P% of bbox diagonal)
  6. Catmull-Clark subdivision (1 iter) → pure quad mesh  (each tri → 3 quads)
  7. Write OBJ

CGAL 5.6 (libcgal-dev) must be installed; g++, libgmp-dev, libmpfr-dev are
required for compilation.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CPP_SRC    = SCRIPT_DIR / "stl_to_quad_cgal.cpp"
CPP_BIN    = SCRIPT_DIR / "stl_to_quad_cgal"      # compiled binary
INPUT_STL  = SCRIPT_DIR / "gyroid_surface_lattice.stl"
OUTPUT_OBJ = SCRIPT_DIR / "gyroid_lattice_quad.obj"

# ── Mesh parameters ────────────────────────────────────────────────────────────
# Stage-2 target: faces after Garland-Heckbert (curvature-aware) simplification.
# 50 000 → ~150 000 quads after Catmull-Clark (1 CC iter triples face count).
TARGET_FACES = 50_000

# Stage-1 target: fast edge-length-cost pre-simplification.
# 0 = auto (10 × TARGET_FACES, capped at 500 000).
COARSE_FACES = 500_000

# Target isotropic edge length expressed as a percentage of the bbox diagonal.
# For the gyroid domain (≈4×2.5×10 mm, diag≈11 mm): 1.5% → ~0.17 mm edge.
REMESH_EDGE_PCT = 1.5

# Percentile threshold for curvature-based vertex pinning.
# Vertices above this percentile of |Gaussian curvature| will never be
# collapsed by QEM, preserving sharp gyroid ridges and channel corners.
CURV_PIN_PERCENTILE = 90
# ──────────────────────────────────────────────────────────────────────────────


def compile_cgal(src: Path, out: Path) -> None:
    """Compile the CGAL C++ program if it is out of date."""
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"[build] Binary is up to date: {out.name}")
        return

    cmd = [
        "g++", "-std=c++17", "-O2",
        "-I/usr/include/eigen3",   # Eigen3 required by GarlandHeckbert policies
        str(src), "-o", str(out),
        "-lgmp", "-lmpfr",
    ]
    print(f"[build] Compiling {src.name} …")
    print(f"        {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMPILE ERROR:\n", result.stderr)
        sys.exit(1)
    print(f"[build] OK ({time.time()-t0:.1f}s)")


def run_quad_mesh(
    input_stl:  Path,
    output_obj: Path,
    target_faces:       int   = TARGET_FACES,
    coarse_faces:       int   = COARSE_FACES,
    remesh_edge_pct:    float = REMESH_EDGE_PCT,
    curv_pin_percentile:float = CURV_PIN_PERCENTILE,
) -> None:
    """Run the compiled CGAL binary."""
    if not input_stl.exists():
        sys.exit(f"ERROR: input not found: {input_stl}")

    cmd = [
        str(CPP_BIN),
        str(input_stl),
        str(output_obj),
        "--target-faces",           str(target_faces),
        "--coarse-faces",           str(coarse_faces),
        "--remesh-edge-pct",        str(remesh_edge_pct),
        "--curv-pin-percentile",    str(curv_pin_percentile),
    ]

    print(f"\n[run]  {input_stl.name}  →  {output_obj.name}")
    print(f"       coarse_faces={coarse_faces}  target_faces={target_faces}  "
          f"edge_pct={remesh_edge_pct}%  curv_pin={curv_pin_percentile}th pct")
    print(f"       {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        sys.exit(f"ERROR: binary exited with code {result.returncode}")

    if output_obj.exists():
        size_mb = output_obj.stat().st_size / 1e6
        print(f"\n[done] {output_obj.name}  ({size_mb:.1f} MB)  "
              f"in {elapsed:.1f}s")
    else:
        print(f"\n[warn] Output file not found: {output_obj}")


def main() -> None:
    compile_cgal(CPP_SRC, CPP_BIN)
    run_quad_mesh(INPUT_STL, OUTPUT_OBJ,
                  target_faces=TARGET_FACES,
                  coarse_faces=COARSE_FACES,
                  remesh_edge_pct=REMESH_EDGE_PCT,
                  curv_pin_percentile=CURV_PIN_PERCENTILE)


if __name__ == "__main__":
    main()
