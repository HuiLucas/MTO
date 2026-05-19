#!/usr/bin/env python3
"""
gyroid_to_stl.py  –  Extract the gyroid wall surface as a high-resolution STL.

The isosurface extracted is:

    |G(x,y,z)| - half_thickness = 0       (i.e. the solid/fluid interface)

where
    G  = sin(kx·x)·cos(ky·y) + sin(ky·y)·cos(kz·z) + sin(kz·z)·cos(kx·x)
    kx, ky, kz  =  k_base  +  RBF(dk_ctrl)(x, y, z)

Usage
-----
    python gyroid_to_stl.py \\
        [--ctrl  app/gyroid_ctrl_pts_checkpoint.txt] \\
        [--out   gyroid_surface.stl] \\
        [--unit  1.5]    # gyroid cell size in mm  → sets k_base
        [--wall  0.60]   # wall thickness in mm
        [--res   0.025]  # voxel size in mm (lower = higher definition)
        [--xmin 0] [--xmax 4] [--ymin 0] [--ymax 2.5] [--zmin 0] [--zmax 10]
        [--mirror-y]     # mirror the half-domain across y=ymax to get the full part
        [--config gyroid_case_config.yaml]   # read unit/wall from YAML instead

Dependencies
------------
    numpy, scipy, scikit-image  (pip install scikit-image)
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from skimage.measure import marching_cubes


# ── RBF field (same thin-plate-spline as the optimizer) ───────────────────────

class _BakedRBF:
    """
    Fast spatial-frequency field via thin-plate-spline RBF baked onto a
    regular grid with trilinear lookup.  Mirrors the optimizer's implementation
    so the same checkpoint produces the same geometry.
    """

    def __init__(self, ctrl_pts: np.ndarray, dk_ctrl: np.ndarray,
                 bbox_min: np.ndarray, bbox_max: np.ndarray,
                 bake_spacing: float = 0.3):
        self._bbox_min = bbox_min
        self._bbox_max = bbox_max

        rbfs = [RBFInterpolator(ctrl_pts, dk_ctrl[:, ax],
                                kernel='thin_plate_spline', degree=1)
                for ax in range(3)]

        axes = [np.arange(lo, hi + bake_spacing, bake_spacing)
                for lo, hi in zip(bbox_min, bbox_max)]
        BX, BY, BZ = np.meshgrid(*axes, indexing='ij')
        pts = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
        baked = np.column_stack([rbf(pts) for rbf in rbfs])
        nx, ny, nz = BX.shape
        self._grid = baked.reshape(nx, ny, nz, 3)
        self._axes = axes
        self._step = np.array([a[1] - a[0] if len(a) > 1 else 1.0 for a in axes])
        self._shape = (nx, ny, nz)

    def __call__(self, pts_mm: np.ndarray) -> np.ndarray:
        """Trilinear lookup; returns (N, 3) dk perturbations."""
        pts = np.clip(pts_mm, self._bbox_min, self._bbox_max)
        nx, ny, nz = self._shape
        gx = (pts[:, 0] - self._axes[0][0]) / self._step[0]
        gy = (pts[:, 1] - self._axes[1][0]) / self._step[1]
        gz = (pts[:, 2] - self._axes[2][0]) / self._step[2]
        ix = np.clip(gx.astype(int), 0, nx - 2)
        iy = np.clip(gy.astype(int), 0, ny - 2)
        iz = np.clip(gz.astype(int), 0, nz - 2)
        tx = (gx - ix)[:, None]
        ty = (gy - iy)[:, None]
        tz = (gz - iz)[:, None]
        g = self._grid
        return (g[ix,   iy,   iz  ] * (1-tx)*(1-ty)*(1-tz)
              + g[ix+1, iy,   iz  ] *    tx *(1-ty)*(1-tz)
              + g[ix,   iy+1, iz  ] * (1-tx)*   ty *(1-tz)
              + g[ix+1, iy+1, iz  ] *    tx *   ty *(1-tz)
              + g[ix,   iy,   iz+1] * (1-tx)*(1-ty)*   tz
              + g[ix+1, iy,   iz+1] *    tx *(1-ty)*   tz
              + g[ix,   iy+1, iz+1] * (1-tx)*   ty *   tz
              + g[ix+1, iy+1, iz+1] *    tx *   ty *   tz)


# ── Gyroid scalar field ────────────────────────────────────────────────────────

def gyroid_G(pts_mm: np.ndarray, k_base: float,
             rbf_field: _BakedRBF | None) -> np.ndarray:
    """
    Evaluate G at every point in pts_mm (N,3).
    If rbf_field is None, uses uniform k_base everywhere.
    """
    dk = rbf_field(pts_mm) if rbf_field is not None else np.zeros((len(pts_mm), 3))
    x  = pts_mm[:, 0]; y = pts_mm[:, 1]; z = pts_mm[:, 2]
    kx = k_base + dk[:, 0]
    ky = k_base + dk[:, 1]
    kz = k_base + dk[:, 2]
    return (np.sin(kx * x) * np.cos(ky * y)
          + np.sin(ky * y) * np.cos(kz * z)
          + np.sin(kz * z) * np.cos(kx * x))


def build_sdf_volume(xv: np.ndarray, yv: np.ndarray, zv: np.ndarray,
                     k_base: float, half_thickness: float,
                     rbf_field: _BakedRBF | None,
                     batch: int = 500_000) -> np.ndarray:
    """
    Build the scalar volume  f = |G| - half_thickness  over the grid defined
    by 1-D arrays xv, yv, zv (all in mm).  Returns shape (Nx, Ny, Nz) float32.
    The isosurface f = 0 is the gyroid wall boundary.
    """
    nx, ny, nz = len(xv), len(yv), len(zv)
    XG, YG, ZG = np.meshgrid(xv, yv, zv, indexing='ij')
    pts = np.column_stack([XG.ravel(), YG.ravel(), ZG.ravel()])  # (N, 3)

    N   = len(pts)
    G   = np.empty(N, dtype=np.float32)
    for i in range(0, N, batch):
        sl = pts[i:i + batch]
        G[i:i + batch] = gyroid_G(sl, k_base, rbf_field).astype(np.float32)
        if (i // batch) % 20 == 0 and i > 0:
            print(f"  Evaluating G … {i:,}/{N:,}  ({100*i/N:.0f}%)", end='\r')

    print(f"  Evaluating G … {N:,}/{N:,} (100%)        ")
    sdf = np.abs(G).reshape(nx, ny, nz) - np.float32(half_thickness)
    return sdf


# ── STL writer (vectorised binary) ────────────────────────────────────────────

def write_binary_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """
    Write a binary STL from vertex (V,3) and face-index (F,3) arrays.
    Per-face normals are computed analytically.
    """
    n_tri = len(faces)
    v0 = verts[faces[:, 0]].astype('<f4')
    v1 = verts[faces[:, 1]].astype('<f4')
    v2 = verts[faces[:, 2]].astype('<f4')

    normals = np.cross(v1 - v0, v2 - v0).astype('<f4')
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1.0)

    # Binary STL: 50 bytes/triangle = 12 float32 (normal + 3 verts) + uint16 attr
    # Build as (n_tri, 50) uint8 by concatenating float and attr byte views.
    float_data = np.concatenate([normals, v0, v1, v2], axis=1)   # (n_tri, 12) f32
    fb  = float_data.view(np.uint8).reshape(n_tri, 48)
    ab  = np.zeros((n_tri, 2), dtype=np.uint8)
    buf = np.concatenate([fb, ab], axis=1).tobytes()              # (n_tri, 50) → bytes

    with open(path, 'wb') as fh:
        fh.write(b'\x00' * 80)                                    # header
        fh.write(struct.pack('<I', n_tri))
        fh.write(buf)

    size_mb = path.stat().st_size / 1e6
    print(f"  Wrote {n_tri:,} triangles  ({size_mb:.1f} MB)  →  {path}")


# ── Mirror helper ─────────────────────────────────────────────────────────────

def mirror_mesh_y(verts: np.ndarray, faces: np.ndarray,
                  y_mirror: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect the mesh across the plane y = y_mirror and append to original.
    Flips face winding on the mirrored copy to keep outward normals consistent.
    """
    v_mir       = verts.copy()
    v_mir[:, 1] = 2.0 * y_mirror - v_mir[:, 1]
    f_mir       = faces[:, [0, 2, 1]]            # reverse winding
    verts_out   = np.concatenate([verts, v_mir], axis=0)
    faces_out   = np.concatenate([faces, f_mir + len(verts)], axis=0)
    return verts_out, faces_out


def simplify_mesh_in_memory(verts: np.ndarray, faces: np.ndarray,
                            target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce triangle count with Open3D if the mesh exceeds target_faces."""
    if len(faces) <= target_faces:
        return verts, faces

    print(f"\n  Simplifying mesh from {len(faces):,} down to {target_faces:,} faces...")
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )

        verts_simplified = np.asarray(mesh.vertices)
        faces_simplified = np.asarray(mesh.triangles)
        print(f"  Simplified to: {len(verts_simplified):,} vertices, {len(faces_simplified):,} triangles")
        return verts_simplified, faces_simplified
    except ImportError:
        print("  WARNING: 'open3d' not installed. Skipping decimation.")
        print("  To enable automatic size reduction, run: pip install open3d")
        return verts, faces


# ── YAML config reader (optional) ────────────────────────────────────────────

def read_yaml_params(yaml_path: Path) -> dict:
    """Minimal YAML parser for the gyroid_case_config.yaml – no PyYAML required."""
    params = {}
    try:
        import yaml
        with open(yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        opt = cfg.get('optimization', {})
        params['unit']   = float(opt.get('unit', 1.5))
        params['wall']   = float(opt.get('wall', 0.30))
        params['kbound'] = float(opt.get('kbound', 2.0))
        geo = cfg.get('geometry', {})
        size = geo.get('size_mm', [4.0, 2.5, 10.0])
        params['xmax'] = float(size[0])
        params['ymax'] = float(size[1])
        params['zmax'] = float(size[2])
    except ImportError:
        # Fallback: naive line-by-line parse for the three keys we need
        text = yaml_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            for key in ('unit', 'wall'):
                if line.startswith(key + ':'):
                    try:
                        params[key] = float(line.split(':')[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass
    return params


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent

    # ── defaults – can be overridden by --config then by explicit flags ────────
    defaults = dict(unit=1.5, wall=0.30, kbound=2.0,
                    xmin=0.0, xmax=4.0,
                    ymin=0.0, ymax=2.5,
                    zmin=0.0, zmax=10.0)

    # Pre-scan for --config so we can load it before argparse finalises defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        cfg = read_yaml_params(Path(pre_args.config))
        defaults.update(cfg)
    elif (script_dir / 'gyroid_case_config.yaml').exists():
        cfg = read_yaml_params(script_dir / 'gyroid_case_config.yaml')
        defaults.update(cfg)

    parser = argparse.ArgumentParser(
        description='Export gyroid wall surface to a high-resolution STL.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ctrl',    default=str(script_dir / 'app' / 'gyroid_ctrl_pts_checkpoint.txt'),
                        help='Path to gyroid_ctrl_pts_checkpoint.txt')
    parser.add_argument('--out',     default=str(script_dir / 'gyroid_surface.stl'),
                        help='Output STL file path')
    parser.add_argument('--config',  default=None,
                        help='Path to gyroid_case_config.yaml (auto-detected if omitted)')
    parser.add_argument('--unit',    type=float, default=defaults['unit'],
                        help='Gyroid cell size in mm (sets k_base)')
    parser.add_argument('--wall',    type=float, default=defaults['wall'],
                        help='Minimum physical wall thickness in mm (must match optimizer --wall)')
    parser.add_argument('--kbound',  type=float, default=defaults['kbound'],
                        help='±bound on dk in rad/mm (must match optimizer --kbound); used to compute G-threshold')
    parser.add_argument('--res',     type=float, default=0.01,
                        help='Voxel size in mm – smaller = higher definition')
    parser.add_argument('--target-faces', type=int, default=5_000_000,
                        help='Maximum face count after in-memory mesh decimation')
    parser.add_argument('--xmin',    type=float, default=defaults['xmin'])
    parser.add_argument('--xmax',    type=float, default=defaults['xmax'])
    parser.add_argument('--ymin',    type=float, default=defaults['ymin'])
    parser.add_argument('--ymax',    type=float, default=defaults['ymax'])
    parser.add_argument('--zmin',    type=float, default=defaults['zmin'])
    parser.add_argument('--zmax',    type=float, default=defaults['zmax'])
    parser.add_argument('--mirror-y', action='store_true',
                        help='Mirror across y = ymax to reconstruct the full part from a half-symmetry domain')
    parser.add_argument('--bake',    type=float, default=0.3,
                        help='RBF bake-grid spacing in mm (controls RBF evaluation accuracy)')
    args = parser.parse_args()

    ctrl_path = Path(args.ctrl)
    out_path  = Path(args.out)
    res       = args.res
    k_base    = 2.0 * math.pi / args.unit
    # G-threshold uses the same formula as the optimizer:
    #   half_t = 0.5 × wall_mm × k_base × √3
    # Guarantees min wall ≥ wall_mm at k=k_base, worst-case triple-point geometry (C=√3).
    # Using k_max = k_base+kbound instead would demand half_t > G_MAX=1.5 → all solid.
    _sqrt3 = math.sqrt(3)
    half_t = 0.5 * args.wall * k_base * _sqrt3

    print(f"\n{'═'*60}")
    print(f"  Gyroid → STL export")
    print(f"{'═'*60}")
    print(f"  Checkpoint : {ctrl_path}")
    print(f"  Unit size  : {args.unit} mm   →  k_base = {k_base:.4f} rad/mm")
    print(f"  min wall   : {args.wall} mm   →  G_half_threshold = {half_t:.4f}  "
          f"(k_base×√3 = {k_base * _sqrt3:.3f} rad/mm, G_MAX = 1.5)")
    print(f"  Voxel size : {res} mm")
    print(f"  Domain     : x[{args.xmin},{args.xmax}]  "
          f"y[{args.ymin},{args.ymax}]  z[{args.zmin},{args.zmax}]  mm")

    # ── Build voxel grid axes ──────────────────────────────────────────────────
    xv = np.arange(args.xmin, args.xmax + res, res, dtype=np.float32)
    yv = np.arange(args.ymin, args.ymax + res, res, dtype=np.float32)
    zv = np.arange(args.zmin, args.zmax + res, res, dtype=np.float32)
    # Trim last point if it overshoots the domain
    xv = xv[xv <= args.xmax + 1e-9]
    yv = yv[yv <= args.ymax + 1e-9]
    zv = zv[zv <= args.zmax + 1e-9]
    print(f"  Grid       : {len(xv)} × {len(yv)} × {len(zv)} "
          f"= {len(xv)*len(yv)*len(zv):,} voxels")

    # ── Load control points ────────────────────────────────────────────────────
    rbf_field = None
    if ctrl_path.exists():
        data = np.loadtxt(ctrl_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ctrl_pts = data[:, :3]
        dk_ctrl  = data[:, 3:6]

        has_perturbation = np.any(np.abs(dk_ctrl) > 1e-12)
        if has_perturbation:
            print(f"  Control pts: {len(ctrl_pts)}  "
                  f"(dk range [{dk_ctrl.min():.4g}, {dk_ctrl.max():.4g}] rad/mm)")
            print(f"  Building baked RBF field …")
            bbox_min = ctrl_pts.min(axis=0) - 0.5
            bbox_max = ctrl_pts.max(axis=0) + 0.5
            rbf_field = _BakedRBF(ctrl_pts, dk_ctrl, bbox_min, bbox_max,
                                  bake_spacing=args.bake)
            print(f"  RBF field ready.")
        else:
            print(f"  Control pts: {len(ctrl_pts)}  (all dk = 0 → uniform gyroid)")
    else:
        print(f"  WARNING: {ctrl_path} not found – using uniform gyroid (dk = 0)")

    # ── Evaluate |G| - half_thickness over the voxel grid ─────────────────────
    print(f"\n  Computing |G| - half_thickness …")
    sdf = build_sdf_volume(xv, yv, zv, k_base, half_t, rbf_field)
    print(f"  SDF range: [{sdf.min():.4g}, {sdf.max():.4g}]"
          f"   solid_frac ≈ {(sdf < 0).mean():.3f}")

    # ── Marching cubes at level 0 ──────────────────────────────────────────────
    print(f"\n  Running marching cubes …")
    verts_idx, faces, normals_mc, _ = marching_cubes(
        sdf,
        level=0.0,
        spacing=(res, res, res),   # converts voxel indices → mm
        gradient_direction='descent',
        allow_degenerate=False,
    )

    # marching_cubes returns coords relative to the grid origin; shift to domain
    verts = verts_idx + np.array([args.xmin, args.ymin, args.zmin], dtype=np.float32)

    print(f"  Extracted  : {len(verts):,} vertices, {len(faces):,} triangles")

    if len(faces) == 0:
        sys.exit("  ERROR: no isosurface found – check that half_thickness is within "
                 "the range of |G| (G_max ≈ 1.5 for standard gyroid).")

    # ── Optional y-mirror ─────────────────────────────────────────────────────
    if args.mirror_y:
        print(f"  Mirroring across y = {args.ymax} …")
        verts, faces = mirror_mesh_y(verts, faces, y_mirror=args.ymax)
        print(f"  After mirror: {len(verts):,} vertices, {len(faces):,} triangles")

    # ── Optional in-memory decimation ────────────────────────────────────────
    if args.target_faces > 0:
        verts, faces = simplify_mesh_in_memory(verts, faces, args.target_faces)

    # ── Write STL ─────────────────────────────────────────────────────────────
    print(f"\n  Writing STL …")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_stl(out_path, verts, faces)

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
