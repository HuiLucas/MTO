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


# ── Encapsulation wall mesh (analytic) ───────────────────────────────────────

def _box_mesh(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts, faces) for a closed axis-aligned box with outward normals."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 0-3 z=z0
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 4-7 z=z1
    ], dtype=np.float32)
    f = np.array([
        [0, 3, 2], [0, 2, 1],  # bottom  −z
        [4, 5, 6], [4, 6, 7],  # top     +z
        [0, 1, 5], [0, 5, 4],  # front   −y
        [2, 3, 7], [2, 7, 6],  # back    +y
        [3, 0, 4], [3, 4, 7],  # left    −x
        [1, 2, 6], [1, 6, 5],  # right   +x
    ], dtype=np.int32)
    return v, f


def build_encap_mesh(xmin: float, xmax: float,
                     ymin: float, ymax: float,
                     zmin: float, zmax: float,
                     thickness: float,
                     open_faces: set) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the encapsulation wall mesh analytically as closed rectangular prisms.

    Each closed face gets one box placed immediately outside the domain boundary.
    Adjacent closed faces share corner volumes (the corner region is covered by
    both neighbouring boxes), which is fine for slicers and 3-D printing.
    The perpendicular extent of each box mirrors the SDF logic: on closed sides
    the box reaches t beyond the domain (filling the corner); on open sides it
    stops exactly at the domain boundary (generating a visible end-cap face).

    Returns (verts (N,3), faces (M,3)) ready to concatenate with the lattice mesh.
    Valid face names: xmin, xmax, ymin, ymax, zmin, zmax.
    """
    t = thickness
    x_lo = xmin - t if 'xmin' not in open_faces else xmin
    x_hi = xmax + t if 'xmax' not in open_faces else xmax
    y_lo = ymin - t if 'ymin' not in open_faces else ymin
    y_hi = ymax + t if 'ymax' not in open_faces else ymax
    z_lo = zmin - t if 'zmin' not in open_faces else zmin
    z_hi = zmax + t if 'zmax' not in open_faces else zmax

    boxes: list[tuple] = []
    if 'xmin' not in open_faces:
        boxes.append(([xmin - t, y_lo, z_lo], [xmin,     y_hi, z_hi]))
    if 'xmax' not in open_faces:
        boxes.append(([xmax,     y_lo, z_lo], [xmax + t, y_hi, z_hi]))
    if 'ymin' not in open_faces:
        boxes.append(([x_lo, ymin - t, z_lo], [x_hi, ymin,     z_hi]))
    if 'ymax' not in open_faces:
        boxes.append(([x_lo, ymax,     z_lo], [x_hi, ymax + t, z_hi]))
    if 'zmin' not in open_faces:
        boxes.append(([x_lo, y_lo, zmin - t], [x_hi, y_hi, zmin    ]))
    if 'zmax' not in open_faces:
        boxes.append(([x_lo, y_lo, zmax    ], [x_hi, y_hi, zmax + t]))

    if not boxes:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    all_v, all_f, offset = [], [], 0
    for lo, hi in boxes:
        v, f = _box_mesh(np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32))
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    return np.concatenate(all_v), np.concatenate(all_f)


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
        params['encap_wall_mm']    = float(geo.get('encap_wall_mm', 0.0))
        params['encap_open_faces'] = list(geo.get('encap_open_faces', ['zmin', 'zmax']))
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
                    zmin=0.0, zmax=10.0,
                    encap_wall_mm=0.0,
                    encap_open_faces=['zmin', 'zmax'])

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
    parser.add_argument('--res',     type=float, default=0.008,
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
    parser.add_argument('--encap-wall', type=float, default=defaults['encap_wall_mm'],
                        dest='encap_wall',
                        help='Encapsulation shell thickness in mm (0 = disabled). '
                             'Adds solid outer walls that seal the heat-exchanger body.')
    parser.add_argument('--encap-open-faces', nargs='+',
                        default=defaults['encap_open_faces'],
                        dest='encap_open_faces',
                        metavar='FACE',
                        help='Domain faces left open in the encapsulation shell '
                             '(xmin xmax ymin ymax zmin zmax). '
                             'Typically the inlet/outlet faces. '
                             'When --mirror-y is used, add ymax to keep the symmetry plane open.')
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

    # ── Resolve encapsulation open_faces (needed before grid extension) ────────
    open_faces: set = set()
    if args.encap_wall > 0:
        open_faces = set(args.encap_open_faces)
        if args.mirror_y and 'ymax' not in open_faces:
            # ymax is the symmetry plane – a wall there creates a double slab at
            # the centre of the mirrored domain.  The mirrored ymin wall is the
            # correct far-side outer wall after mirror_mesh_y.
            open_faces.add('ymax')
        print(f"  Encap wall : {args.encap_wall} mm  (open faces: {sorted(open_faces)})")

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

    # ── Close the lattice: pad all 6 faces with one solid voxel ─────────────────
    # Each boundary face gets a layer of –1 (solid) just outside the domain.
    # Marching cubes then generates flat cap triangles that seal every open edge
    # where the gyroid sheet is cut off at the domain wall, producing a
    # water-tight closed solid.
    sdf_closed = np.pad(sdf, 1, constant_values=np.float32(1.0))
    del sdf
    # Origin shifts back by one voxel to match the padded grid.
    mc_origin = np.array([args.xmin - res, args.ymin - res, args.zmin - res],
                         dtype=np.float32)

    # ── Marching cubes on gyroid lattice ──────────────────────────────────────
    print(f"\n  Running marching cubes (lattice) …")
    verts_idx, faces, _, _ = marching_cubes(
        sdf_closed,
        level=0.0,
        spacing=(res, res, res),
        gradient_direction='descent',
        allow_degenerate=False,
    )
    del sdf_closed
    verts = verts_idx + mc_origin
    print(f"  Extracted  : {len(verts):,} vertices, {len(faces):,} triangles")

    if len(faces) == 0:
        sys.exit("  ERROR: no isosurface found – check that half_thickness is within "
                 "the range of |G| (G_max ≈ 1.5 for standard gyroid).")

    # ── Optional y-mirror (lattice) ───────────────────────────────────────────
    if args.mirror_y:
        print(f"  Mirroring lattice across y = {args.ymax} …")
        verts, faces = mirror_mesh_y(verts, faces, y_mirror=args.ymax)
        print(f"  After mirror: {len(verts):,} vertices, {len(faces):,} triangles")

    # ── Decimate lattice ──────────────────────────────────────────────────────
    if args.target_faces > 0:
        verts, faces = simplify_mesh_in_memory(verts, faces, args.target_faces)

    # ── Optional encapsulation: analytic box mesh (separate solid) ────────────
    encap_verts = encap_faces = None
    if args.encap_wall > 0:
        print(f"\n  Building encapsulation mesh (t = {args.encap_wall} mm) …")
        encap_verts, encap_faces = build_encap_mesh(
            args.xmin, args.xmax,
            args.ymin, args.ymax,
            args.zmin, args.zmax,
            args.encap_wall, open_faces,
        )
        print(f"  Encap mesh : {len(encap_verts):,} vertices, {len(encap_faces):,} triangles")

        if args.mirror_y:
            encap_verts, encap_faces = mirror_mesh_y(encap_verts, encap_faces,
                                                     y_mirror=args.ymax)

    # ── Write STL(s) ──────────────────────────────────────────────────────────
    print(f"\n  Writing STL …")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if encap_verts is not None:
        # Two separate closed solids → two files derived from --out.
        lattice_path = out_path.with_name(out_path.stem + '_lattice' + out_path.suffix)
        encap_path   = out_path.with_name(out_path.stem + '_encap'   + out_path.suffix)
        write_binary_stl(lattice_path, verts, faces)
        write_binary_stl(encap_path,   encap_verts, encap_faces)
    else:
        write_binary_stl(out_path, verts, faces)

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
