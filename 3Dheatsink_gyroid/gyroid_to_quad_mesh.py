#!/usr/bin/env python3
"""
gyroid_to_quad_mesh.py — Generate a quad mesh of the optimised gyroid surface
                          directly from the implicit function via CGAL.

Pipeline
--------
1. Read the gyroid_case_config.yaml (unit, wall, domain, flow axis, RBF ctrl pts).
2. Build the baked RBF field (thin-plate-spline, trilinear lookup) — same maths
   as gyroid_to_stl.py so the mesh matches the actual manufactured geometry.
3. Write a compact binary params file for the CGAL C++ mesher.
4. Compile gyroid_implicit_mesh.cpp (if needed) and run it to produce a
   combined triangle-mesh STL of BOTH wall sheets:
       G(p) = +half_t  →  positive gyroid sheet
       G(p) = -half_t  →  negative gyroid sheet
   CGAL's make_surface_mesh is intrinsically curvature-aware: the distance_bound
   criterion forces more refinement where the surface curves rapidly.
5. Optionally convert the combined STL to a quad-dominant mesh by calling
   stl_to_quad_cgal (Garland-Heckbert QEM + Catmull-Clark → pure quads).

Usage
-----
    python gyroid_to_quad_mesh.py
        [--config  gyroid_case_config.yaml]
        [--ctrl    app/gyroid_ctrl_pts_checkpoint.txt]
        [--out     gyroid_implicit_quad.obj]
        [--angular 30]     (CGAL angle bound, deg)
        [--radius  0.15]   (CGAL circumradius bound, mm)
        [--distance 0.07]  (CGAL surface-deviation bound, mm)
        [--no-quad]        (skip quad conversion, keep triangle STL)
        [--target-faces N] (faces after QEM simplification, default 50000)

Dependencies
------------
    numpy, scipy, pyyaml   (Python side)
    libcgal-dev, g++, libgmp-dev, libmpfr-dev   (CGAL C++ side)
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Reuse the gyroid maths from the existing gyroid_to_stl module ─────────────
# We import selectively so this script can run standalone if needed.
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
from gyroid_to_stl import _BakedRBF, _gyroid_rotation_matrix, read_yaml_params

# ── Paths ──────────────────────────────────────────────────────────────────────
CPP_SRC_MESH = _DIR / "gyroid_implicit_mesh.cpp"
CPP_BIN_MESH = _DIR / "gyroid_implicit_mesh"

CPP_SRC_QUAD = _DIR / "stl_to_quad_cgal.cpp"
CPP_BIN_QUAD = _DIR / "stl_to_quad_cgal"

PARAMS_FILE  = _DIR / "gyroid_mesh_params.bin"


# ── Binary params file writer ─────────────────────────────────────────────────

def write_params_binary(path: Path,
                        xmin, xmax, ymin, ymax, zmin, zmax,
                        k_base, half_t,
                        rot_matrix,      # None or (3,3) ndarray
                        rbf_field) -> None:   # None or _BakedRBF
    """
    Write the binary parameter file consumed by gyroid_implicit_mesh.cpp.

    Format (all little-endian):
        magic    8 B   "GYROID01"
        domain   8×f64 xmin xmax ymin ymax zmin zmax k_base half_t
        has_rot  u8
        R[9]     9×f64  (always written; identity if has_rot=0)
        has_rbf  u8
        If has_rbf:
          dims     3×i32   nx ny nz
          origin   3×f64   gx_min gy_min gz_min
          spacing  3×f64   gdx    gdy    gdz
          data     nx×ny×nz×3 × f64  (C-order, inner dim = channel)
    """
    with open(path, 'wb') as f:
        f.write(b'GYROID01')

        # Domain + physics
        f.write(struct.pack('<8d', xmin, xmax, ymin, ymax, zmin, zmax,
                            k_base, half_t))

        # Rotation matrix
        if rot_matrix is not None:
            f.write(struct.pack('<B', 1))
            f.write(np.asarray(rot_matrix, dtype='<f8').flatten().tobytes())
        else:
            f.write(struct.pack('<B', 0))
            f.write(np.eye(3, dtype='<f8').flatten().tobytes())

        # RBF grid
        if rbf_field is not None:
            f.write(struct.pack('<B', 1))
            nx, ny, nz = rbf_field._shape
            f.write(struct.pack('<3i', nx, ny, nz))
            gx_min = float(rbf_field._axes[0][0])
            gy_min = float(rbf_field._axes[1][0])
            gz_min = float(rbf_field._axes[2][0])
            gdx = float(rbf_field._step[0])
            gdy = float(rbf_field._step[1])
            gdz = float(rbf_field._step[2])
            f.write(struct.pack('<6d', gx_min, gy_min, gz_min, gdx, gdy, gdz))
            # grid shape: (nx, ny, nz, 3) — write as C-order f64
            rbf_field._grid.astype('<f8').tofile(f)
        else:
            f.write(struct.pack('<B', 0))

    size_kb = path.stat().st_size / 1024
    print(f"  Params file: {path.name}  ({size_kb:.0f} kB)")


# ── Compile helpers ───────────────────────────────────────────────────────────

def _compile(src: Path, out: Path, extra_flags: list[str] = ()) -> None:
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"[build] {out.name} is up to date")
        return
    cmd = ["g++", "-std=c++17", "-O3",
           "-I/usr/include/eigen3",
           str(src), "-o", str(out),
           "-lgmp", "-lmpfr"] + list(extra_flags)
    print(f"[build] {src.name}  →  {out.name}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE ERROR:\n", r.stderr)
        sys.exit(1)
    print(f"[build] OK ({time.time()-t0:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = _DIR

    # ── Argument defaults read from YAML ──────────────────────────────────────
    defaults = dict(unit=1.5, wall=0.30, kbound=2.0,
                    xmin=0.0, xmax=5.0,
                    ymin=0.0, ymax=2.5,
                    zmin=0.0, zmax=10.0,
                    gyroid_rot_vec=None)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = (Path(pre_args.config)
                   if pre_args.config
                   else script_dir / 'gyroid_case_config.yaml')
    if config_path.exists():
        defaults.update(read_yaml_params(config_path))

    parser = argparse.ArgumentParser(
        description='Generate a quad mesh of the gyroid surface via CGAL '
                    'implicit meshing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',   default=str(config_path))
    parser.add_argument('--ctrl',     default=str(script_dir / 'app'
                                      / 'gyroid_ctrl_pts_checkpoint.txt'))
    parser.add_argument('--out',      default=str(script_dir
                                      / 'gyroid_implicit_quad.obj'))
    parser.add_argument('--unit',     type=float, default=defaults['unit'])
    parser.add_argument('--wall',     type=float, default=defaults['wall'])
    parser.add_argument('--kbound',   type=float, default=defaults['kbound'])
    parser.add_argument('--bake',     type=float, default=0.3,
                        help='RBF bake-grid spacing in mm')
    parser.add_argument('--xmin',     type=float, default=defaults['xmin'])
    parser.add_argument('--xmax',     type=float, default=defaults['xmax'])
    parser.add_argument('--ymin',     type=float, default=defaults['ymin'])
    parser.add_argument('--ymax',     type=float, default=defaults['ymax'])
    parser.add_argument('--zmin',     type=float, default=defaults['zmin'])
    parser.add_argument('--zmax',     type=float, default=defaults['zmax'])
    # CGAL meshing criteria
    parser.add_argument('--angular',  type=float, default=30.0,
                        help='Angular bound (deg): triangle quality lower bound')
    parser.add_argument('--radius',   type=float, default=0.15,
                        help='Circumradius bound (mm): controls mesh density')
    parser.add_argument('--distance', type=float, default=0.07,
                        help='Surface-distance bound (mm): curvature-adaptive '
                             'refinement threshold')
    parser.add_argument('--margin',   type=float, default=-1.0,
                        help='Boundary taper width (mm): surface fades to +1 '
                             'within this distance of the domain wall. '
                             'Default: 3 × --radius (auto).')
    # Quad conversion
    parser.add_argument('--no-quad',  action='store_true',
                        help='Skip Catmull-Clark quad conversion; output STL only')
    parser.add_argument('--target-faces', type=int, default=50_000,
                        dest='target_faces',
                        help='Faces after GH simplification (before Catmull-Clark)')
    parser.add_argument('--coarse-faces', type=int, default=500_000,
                        dest='coarse_faces')
    args = parser.parse_args()

    ctrl_path  = Path(args.ctrl)
    out_path   = Path(args.out)
    stl_path   = out_path.with_suffix('.stl')   # intermediate triangle mesh

    k_base = 2.0 * math.pi / args.unit
    sqrt3  = math.sqrt(3)
    half_t = 0.5 * args.wall * k_base * sqrt3

    print(f"\n{'═'*60}")
    print(f"  Gyroid → quad mesh via CGAL implicit surface mesher")
    print(f"{'═'*60}")
    print(f"  Unit     : {args.unit} mm  →  k_base = {k_base:.4f} rad/mm")
    print(f"  Wall     : {args.wall} mm  →  half_t = {half_t:.4f}")
    print(f"  Domain   : x[{args.xmin},{args.xmax}]  "
          f"y[{args.ymin},{args.ymax}]  z[{args.zmin},{args.zmax}] mm")
    auto_margin = 3.0 * args.radius if args.margin < 0 else args.margin
    print(f"  CGAL     : angular={args.angular}°  "
          f"radius={args.radius} mm  dist={args.distance} mm  "
          f"margin={auto_margin:.3g} mm")

    # ── Load control points and build baked RBF ───────────────────────────────
    rbf_field  = None
    rot_matrix = None

    if ctrl_path.exists():
        data = np.loadtxt(ctrl_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ctrl_pts = data[:, :3]
        dk_ctrl  = data[:, 3:6]

        if np.any(np.abs(dk_ctrl) > 1e-12):
            print(f"\n  Control pts: {len(ctrl_pts)}  "
                  f"(dk ∈ [{dk_ctrl.min():.3g}, {dk_ctrl.max():.3g}] rad/mm)")
            print(f"  Baking RBF field (spacing={args.bake} mm) …")
            t0 = time.time()
            bbox_min = ctrl_pts.min(axis=0) - 0.5
            bbox_max = ctrl_pts.max(axis=0) + 0.5
            rbf_field = _BakedRBF(ctrl_pts, dk_ctrl, bbox_min, bbox_max,
                                  bake_spacing=args.bake)
            nx, ny, nz = rbf_field._shape
            print(f"  RBF grid   : {nx}×{ny}×{nz}  ({time.time()-t0:.1f}s)")
        else:
            print(f"  Control pts: {len(ctrl_pts)}  (all dk = 0 → uniform gyroid)")
    else:
        print(f"  WARNING: ctrl file not found → using uniform gyroid")

    # ── Gyroid rotation from flow-direction ───────────────────────────────────
    rot_vec = defaults.get('gyroid_rot_vec')
    if rot_vec is not None:
        rot_matrix = _gyroid_rotation_matrix(np.array(rot_vec, dtype=float))
        print(f"  Gyroid rotation: ({rot_vec[0]:.4f}, {rot_vec[1]:.4f}, {rot_vec[2]:.4f})")

    # ── Write binary params file ──────────────────────────────────────────────
    print("\n  Writing params file …")
    write_params_binary(
        PARAMS_FILE,
        args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax,
        k_base, half_t,
        rot_matrix, rbf_field,
    )

    # ── Compile CGAL mesher ───────────────────────────────────────────────────
    print()
    _compile(CPP_SRC_MESH, CPP_BIN_MESH)

    # ── Run CGAL mesher ───────────────────────────────────────────────────────
    print(f"\n  Running CGAL implicit surface mesher …")
    cmd_mesh = [
        str(CPP_BIN_MESH),
        str(PARAMS_FILE),
        str(stl_path),
        "--angular",  str(args.angular),
        "--radius",   str(args.radius),
        "--distance", str(args.distance),
        "--margin",   str(args.margin),
    ]
    print(f"  {' '.join(cmd_mesh)}\n")
    t0 = time.time()
    r = subprocess.run(cmd_mesh, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR: CGAL mesher exited with code {r.returncode}")
    elapsed = time.time() - t0
    stl_mb = stl_path.stat().st_size / 1e6
    print(f"\n  Triangle STL: {stl_path.name}  ({stl_mb:.1f} MB)  [{elapsed:.1f}s]")

    # ── Quad conversion ───────────────────────────────────────────────────────
    if args.no_quad:
        print(f"\n  Skipping quad conversion (--no-quad).  Output: {stl_path}")
        return

    print(f"\n  Converting to quad mesh via stl_to_quad_cgal …")
    _compile(CPP_SRC_QUAD, CPP_BIN_QUAD)

    cmd_quad = [
        str(CPP_BIN_QUAD),
        str(stl_path),
        str(out_path),
        "--target-faces", str(args.target_faces),
        "--coarse-faces", str(args.coarse_faces),
        "--remesh-edge-pct", "1.5",
        "--curv-pin-percentile", "90",
    ]
    print(f"  {' '.join(cmd_quad)}\n")
    t0 = time.time()
    r = subprocess.run(cmd_quad, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR: quad converter exited with code {r.returncode}")
    elapsed = time.time() - t0

    if out_path.exists():
        obj_mb = out_path.stat().st_size / 1e6
        print(f"\n  Quad OBJ: {out_path.name}  ({obj_mb:.1f} MB)  [{elapsed:.1f}s]")
    print(f"\n  Done.  Output: {out_path}")


if __name__ == '__main__':
    main()
