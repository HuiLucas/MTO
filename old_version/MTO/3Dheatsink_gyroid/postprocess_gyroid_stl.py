"""
postprocess_gyroid_stl.py
=========================
Generate a watertight solid STL of the optimised Gyroid heatsink lattice.

No LEAP71 / PicoGK dependency — uses numpy / scipy / scikit-image / trimesh.

Three source modes
------------------
  gamma      – (RECOMMENDED) reads the gamma field that OpenFOAM actually
               solved from app/0/gamma + the cached cell centres.
               Guaranteed to match the CFD, no RBF artefacts.
  checkpoint – reconstructs geometry from gyroid_ctrl_pts_checkpoint.txt via
               the baked RBF field (same 0.4 mm trilinear grid the optimizer
               uses, so results are consistent with gamma mode).
  final      – same as checkpoint but reads gyroid_ctrl_pts_optimised.txt.

How the solid is produced (all modes)
--------------------------------------
1. A smooth SDF is built on a regular 3-D voxel grid:
     gamma mode : sdf = 0.5 - gamma(x,y,z)   (nearest-neighbour from mesh)
     ctrl modes : sdf = 0.5*wall - |G(kx,ky,kz)|  evaluated via baked RBF
2. The SDF is padded with −1 at the border (caps walls at domain boundary).
3. marching_cubes at level 0 extracts the watertight solid surface.
4. Small floating fragments are removed by volume-fraction threshold.
5. Exported via trimesh (fallback: numpy-stl).

Usage
-----
    python postprocess_gyroid_stl.py --mode gamma       [options]
    python postprocess_gyroid_stl.py --mode checkpoint  [options]
    python postprocess_gyroid_stl.py --mode final       [options]

Options
-------
  --mode          gamma | checkpoint | final  (default: gamma)
  --ctrl          explicit ctrl-pts file (overrides checkpoint/final default)
  --gamma-time    time directory to read gamma from  (default: 0)
  --out           output STL path  (default: app/gyroid_<mode>.stl)
  --voxel         voxel size in mm  (default: 0.05)
  --unit          TPMS cell size in mm — sets k_base  (default: 1.5)
  --wall          wall thickness in mm  (default: 0.20)
  --case          OpenFOAM case directory relative to script  (default: app)
  --min-vol-frac  drop components below this fraction of largest (default: 0.01)
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import KDTree

# ── scikit-image ──────────────────────────────────────────────────────────────
try:
    from skimage.measure import marching_cubes
except ImportError:
    sys.exit(
        "ERROR: scikit-image not found.\n"
        "Install with:  pip install scikit-image\n"
        "  or:          conda install -c conda-forge scikit-image"
    )

# ── trimesh (primary export) / numpy-stl (fallback) ──────────────────────────
try:
    import trimesh
    _HAVE_TRIMESH = True
except ImportError:
    _HAVE_TRIMESH = False
    try:
        from stl import mesh as _stl_mesh
        _HAVE_NUMPY_STL = True
    except ImportError:
        _HAVE_NUMPY_STL = False

if not _HAVE_TRIMESH and not _HAVE_NUMPY_STL:
    sys.exit(
        "ERROR: neither trimesh nor numpy-stl found.\n"
        "Install with:  pip install trimesh\n"
        "  or:          pip install numpy-stl"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Domain / TPMS defaults  (must match gyroid_rbf_optimizer.py)
# ═══════════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent

OPT_XMIN, OPT_XMAX = 0.0,  4.0    # mm
OPT_YMIN, OPT_YMAX = 0.0,  2.5    # mm
OPT_ZMIN, OPT_ZMAX = 0.0, 10.0    # mm

F_UNIT_SIZE      = 1.5    # TPMS cell size (mm)
F_WALL_THICKNESS = 0.20   # gyroid solid shell thickness (mm)
VOXEL_SIZE       = 0.05   # mm
RBF_BAKE_SPACING = 0.4    # mm  — must match optimizer bake_spacing


# ── OpenFOAM scalar field reader ──────────────────────────────────────────────

def read_scalar_field(path: Path) -> np.ndarray:
    """Parse an OpenFOAM ASCII volScalarField; returns 1-D internalField array."""
    text = path.read_text()
    m = re.search(r'internalField\s+uniform\s+([0-9Ee.+\-]+)', text)
    if m:
        n_m = re.search(r'nonuniform List<scalar>\s+(\d+)', text)
        if n_m:
            return np.full(int(n_m.group(1)), float(m.group(1)))
        raise ValueError(f"Cannot determine cell count from {path}")
    m = re.search(
        r'internalField\s+nonuniform List<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)',
        text, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse internalField in {path}")
    n = int(m.group(1))
    vals = np.fromstring(m.group(2), sep='\n', count=n)
    if len(vals) != n:
        vals = np.array(m.group(2).split(), dtype=float)
    return vals


# ── Baked RBF frequency field ─────────────────────────────────────────────────

class RBFFrequencyField:
    """
    RBF thin-plate-spline baked to a dense grid → fast trilinear lookup.
    Mirrors gyroid_rbf_optimizer.RBFFrequencyField exactly.
    """

    def __init__(
        self,
        ctrl_pts:     np.ndarray,   # (N, 3) mm
        dk_ctrl:      np.ndarray,   # (N, 3) rad/mm
        bbox_min:     np.ndarray,
        bbox_max:     np.ndarray,
        bake_spacing: float = RBF_BAKE_SPACING,
    ):
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)

        print(f"  Fitting RBF ({len(ctrl_pts)} ctrl pts) …")
        rbfs = [
            RBFInterpolator(ctrl_pts, dk_ctrl[:, ax],
                            kernel='thin_plate_spline', degree=1)
            for ax in range(3)
        ]

        bake_axes = [np.arange(lo, hi + bake_spacing, bake_spacing)
                     for lo, hi in zip(self.bbox_min, self.bbox_max)]
        BX, BY, BZ = np.meshgrid(*bake_axes, indexing='ij')
        pts = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
        nx, ny, nz = BX.shape
        print(f"  Baking {nx}×{ny}×{nz} grid at {bake_spacing} mm …")
        baked = np.column_stack([rbf(pts) for rbf in rbfs])
        self._grid      = baked.reshape(nx, ny, nz, 3)
        self._bake_axes = bake_axes
        self._nx, self._ny, self._nz = nx, ny, nz
        self._step = np.array([a[1] - a[0] if len(a) > 1 else 1.0
                                for a in bake_axes])

    def get_dk_batch(self, pts_mm: np.ndarray) -> np.ndarray:
        """Vectorised trilinear lookup; returns (N, 3) dk in rad/mm."""
        pts = np.clip(pts_mm, self.bbox_min, self.bbox_max)
        gx = (pts[:, 0] - self._bake_axes[0][0]) / self._step[0]
        gy = (pts[:, 1] - self._bake_axes[1][0]) / self._step[1]
        gz = (pts[:, 2] - self._bake_axes[2][0]) / self._step[2]
        ix = np.clip(gx.astype(int), 0, self._nx - 2)
        iy = np.clip(gy.astype(int), 0, self._ny - 2)
        iz = np.clip(gz.astype(int), 0, self._nz - 2)
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


# ── grid helpers ──────────────────────────────────────────────────────────────

def make_grid(voxel_size: float):
    """Return (xs, ys, zs, X, Y, Z, pts) for the optimisation domain."""
    xs = np.arange(OPT_XMIN, OPT_XMAX + voxel_size * 0.5, voxel_size)
    ys = np.arange(OPT_YMIN, OPT_YMAX + voxel_size * 0.5, voxel_size)
    zs = np.arange(OPT_ZMIN, OPT_ZMAX + voxel_size * 0.5, voxel_size)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return xs, ys, zs, X, Y, Z, pts


def pad_and_origin(sdf: np.ndarray, voxel_size: float):
    """Pad SDF with -1 border; return (padded, origin_mm)."""
    padded = np.pad(sdf, pad_width=1, mode='constant', constant_values=-1.0)
    origin = np.array([OPT_XMIN - voxel_size,
                       OPT_YMIN - voxel_size,
                       OPT_ZMIN - voxel_size])
    return padded, origin


# ── mode: ctrl pts (checkpoint / final) ──────────────────────────────────────

def load_ctrl_pts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments='#')
    return data[:, :3], data[:, 3:]


def build_sdf_from_ctrl(
    ctrl_pts:       np.ndarray,
    dk_ctrl:        np.ndarray,
    k_base:         float,
    half_thickness: float,
    voxel_size:     float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a voxel SDF from RBF control-point frequency perturbations.
    Uses the baked trilinear RBF field (same as the optimizer) to avoid the
    wild oscillations that raw TPS evaluation produces between control points.
    """
    xs, ys, zs, X, Y, Z, pts = make_grid(voxel_size)
    nx, ny, nz = len(xs), len(ys), len(zs)
    print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels at {voxel_size} mm")

    # baked field — same 0.4 mm trilinear grid the optimizer uses
    field_min = ctrl_pts.min(axis=0)
    field_max = ctrl_pts.max(axis=0)
    field = RBFFrequencyField(ctrl_pts, dk_ctrl, field_min, field_max,
                              RBF_BAKE_SPACING)

    print("  Evaluating baked RBF on voxel grid …")
    dk = field.get_dk_batch(pts)                    # (N, 3)
    kx = (k_base + dk[:, 0]).reshape(nx, ny, nz)
    ky = (k_base + dk[:, 1]).reshape(nx, ny, nz)
    kz = (k_base + dk[:, 2]).reshape(nx, ny, nz)

    print("  Evaluating Gyroid implicit field …")
    G = (np.sin(kx * X) * np.cos(ky * Y)
       + np.sin(ky * Y) * np.cos(kz * Z)
       + np.sin(kz * Z) * np.cos(kx * X))

    sdf  = (half_thickness - np.abs(G)).astype(np.float32)
    frac = float((sdf > 0).mean())
    print(f"  Solid voxel fraction: {frac:.4f}  "
          f"({int((sdf > 0).sum()):,} of {nx*ny*nz:,} voxels)")

    return pad_and_origin(sdf, voxel_size)


# ── mode: gamma ───────────────────────────────────────────────────────────────

def build_sdf_from_gamma(
    gamma_path:  Path,
    case_dir:    Path,
    voxel_size:  float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the OpenFOAM gamma field onto a regular voxel grid.

    Strategy: recover the structured hex grid that blockMesh produced by
    finding the unique x/y/z cell-centre coordinates, rebuild the 3-D gamma
    array on that native grid, then use RegularGridInterpolator (trilinear)
    to resample to the desired voxel spacing.  This is exact — it only
    interpolates between grid-adjacent cells (one cell-width apart) and
    faithfully reproduces thin walls.

    Falls back to KD-tree nearest-neighbour if the mesh is unstructured.

    sdf = 0.5 - gamma  →  positive inside solid (gamma < 0.5).
    """
    # ── cell centres ──────────────────────────────────────────────────────
    cc_cache = case_dir / 'cell_centers_mm.npy'
    if not cc_cache.exists():
        sys.exit(
            f"ERROR: cell centres cache not found at {cc_cache}.\n"
            "Run the optimiser for at least one iteration first "
            "(it writes cell_centers_mm.npy on startup)."
        )
    print(f"  Loading cell centres from {cc_cache} …")
    cc_mm = np.load(cc_cache)                       # (N_cells, 3) mm
    n_cells = len(cc_mm)
    print(f"  {n_cells:,} cells")

    # ── gamma field ────────────────────────────────────────────────────────
    print(f"  Reading gamma from {gamma_path} …")
    gamma = read_scalar_field(gamma_path)           # (N_cells,)
    print(f"  gamma range [{gamma.min():.4f}, {gamma.max():.4f}]  "
          f"solid_frac (gamma<0.5) = {(gamma < 0.5).mean():.4f}")

    # ── recover structured grid from cell centres ──────────────────────────
    dec = 4   # rounding decimals for grouping (avoids float noise)
    xu = np.unique(np.round(cc_mm[:, 0], dec))
    yu = np.unique(np.round(cc_mm[:, 1], dec))
    zu = np.unique(np.round(cc_mm[:, 2], dec))
    n_expected = len(xu) * len(yu) * len(zu)
    structured = abs(n_expected - n_cells) / n_cells < 0.01

    if structured:
        print(f"  Structured grid: {len(xu)}×{len(yu)}×{len(zu)} "
              f"= {n_expected:,}  (cell size ≈ "
              f"{xu[1]-xu[0]:.3f}×{yu[1]-yu[0]:.3f}×{zu[1]-zu[0]:.3f} mm)")

        # Map each cell to its (i,j,k) index and fill gamma_3d
        gamma_3d = np.ones((len(xu), len(yu), len(zu)), dtype=np.float32)
        xi = np.searchsorted(xu, np.round(cc_mm[:, 0], dec))
        yi = np.searchsorted(yu, np.round(cc_mm[:, 1], dec))
        zi = np.searchsorted(zu, np.round(cc_mm[:, 2], dec))
        xi = np.clip(xi, 0, len(xu) - 1)
        yi = np.clip(yi, 0, len(yu) - 1)
        zi = np.clip(zi, 0, len(zu) - 1)
        gamma_3d[xi, yi, zi] = gamma.astype(np.float32)

        print("  Trilinear interpolation to voxel grid …")
        interp = RegularGridInterpolator(
            (xu, yu, zu), gamma_3d,
            method='linear', bounds_error=False, fill_value=1.0,
        )
        xs, ys, zs, X, Y, Z, pts = make_grid(voxel_size)
        nx, ny, nz = len(xs), len(ys), len(zs)
        print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels at {voxel_size} mm")
        gamma_grid = interp(pts).reshape(nx, ny, nz).astype(np.float32)

    else:
        print(f"  Unstructured mesh detected — falling back to nearest-neighbour")
        xs, ys, zs, X, Y, Z, pts = make_grid(voxel_size)
        nx, ny, nz = len(xs), len(ys), len(zs)
        print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels at {voxel_size} mm")
        tree = KDTree(cc_mm)
        _, idxs = tree.query(pts, workers=-1)
        gamma_grid = gamma[idxs].reshape(nx, ny, nz).astype(np.float32)

    # sdf > 0 inside solid, < 0 in fluid
    sdf  = (0.5 - gamma_grid).astype(np.float32)
    frac = float((sdf > 0).mean())
    print(f"  Solid voxel fraction: {frac:.4f}  "
          f"({int((sdf > 0).sum()):,} of {nx*ny*nz:,} voxels)")

    return pad_and_origin(sdf, voxel_size)


# ── marching cubes + component filter + export ────────────────────────────────

def mesh_and_export(
    padded:       np.ndarray,
    origin:       np.ndarray,
    voxel_size:   float,
    stl_path:     Path,
    min_vol_frac: float,
) -> None:
    print("  Running marching cubes …")
    verts, faces, _n, _v = marching_cubes(
        padded, level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
    )
    verts += origin
    print(f"  Raw mesh: {len(verts):,} vertices, {len(faces):,} triangles")

    if _HAVE_TRIMESH:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        components = mesh.split(only_watertight=False)
        n_total = len(components)
        if n_total > 1:
            vols = [abs(float(c.volume)) for c in components]
            max_vol   = max(vols)
            threshold = max_vol * min_vol_frac
            kept      = [(c, v) for c, v in zip(components, vols)
                         if v >= threshold]
            n_removed = n_total - len(kept)
            disc_vol  = sum(v for v in vols if v < threshold)
            print(f"  Components: {n_total} → keeping {len(kept)}  "
                  f"(removed {n_removed} with vol < {threshold:.4f} mm³; "
                  f"discarded {disc_vol:.4f} mm³)")
            if kept:
                mesh = (trimesh.util.concatenate([c for c, _ in kept])
                        if len(kept) > 1 else kept[0][0])
            else:
                print("  WARNING: threshold too high — keeping all.")
        else:
            print("  Components: 1 (nothing to remove)")

        print(f"  Final mesh: {len(mesh.vertices):,} vertices, "
              f"{len(mesh.faces):,} triangles  watertight={mesh.is_watertight}")
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(stl_path))
    else:
        print("  (trimesh unavailable — skipping fragment removal)")
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        solid = _stl_mesh.Mesh(np.zeros(len(faces), dtype=_stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                solid.vectors[i][j] = verts[f[j]]
        solid.save(str(stl_path))

    print(f"  Exported → {stl_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a solid STL of the optimised Gyroid heatsink.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--mode', choices=['gamma', 'checkpoint', 'final'], default='gamma',
        help=('gamma      → read app/<gamma-time>/gamma directly (recommended)\n'
              'checkpoint → reconstruct from gyroid_ctrl_pts_checkpoint.txt\n'
              'final      → reconstruct from gyroid_ctrl_pts_optimised.txt'),
    )
    parser.add_argument('--ctrl', type=Path, default=None,
                        help='Explicit ctrl-pts file (overrides checkpoint/final default).')
    parser.add_argument('--gamma-time', default='0',
                        help='Time directory to read gamma from (gamma mode).')
    parser.add_argument('--out',  type=Path, default=None,
                        help='Output STL path (default: app/gyroid_<mode>.stl).')
    parser.add_argument('--case', default='app',
                        help='OpenFOAM case directory (relative to script).')
    parser.add_argument('--voxel', type=float, default=VOXEL_SIZE,
                        help='Voxel size in mm.')
    parser.add_argument('--unit',  type=float, default=F_UNIT_SIZE,
                        help='TPMS cell size in mm (sets k_base; ctrl modes only).')
    parser.add_argument('--wall',  type=float, default=F_WALL_THICKNESS,
                        help='Wall thickness in mm (ctrl modes only).')
    parser.add_argument('--min-vol-frac', type=float, default=0.01,
                        help='Remove components with volume < this fraction of largest.')
    args = parser.parse_args()

    case_dir = (_SCRIPT_DIR / args.case).resolve()
    stl_path = (args.out.resolve() if args.out
                else case_dir / f'gyroid_{args.mode}.stl')

    t0 = time.time()

    if args.mode == 'gamma':
        gamma_path = case_dir / args.gamma_time / 'gamma'
        if not gamma_path.exists():
            sys.exit(
                f"ERROR: gamma field not found at {gamma_path}.\n"
                "Run the optimiser for at least one iteration first,\n"
                "or specify a different --gamma-time."
            )
        print(f"\nMode: gamma  ({gamma_path})")
        padded, origin = build_sdf_from_gamma(
            gamma_path, case_dir, args.voxel,
        )

    else:
        if args.ctrl is not None:
            ctrl_path = args.ctrl.resolve()
        elif args.mode == 'final':
            ctrl_path = case_dir / 'gyroid_ctrl_pts_optimised.txt'
        else:
            ctrl_path = case_dir / 'gyroid_ctrl_pts_checkpoint.txt'

        if not ctrl_path.exists():
            sys.exit(
                f"ERROR: ctrl-pts file not found: {ctrl_path}\n"
                f"  For --mode final:      run the optimiser to completion.\n"
                f"  For --mode checkpoint: at least one iteration must have run."
            )

        print(f"\nMode: {args.mode}  ({ctrl_path})")
        ctrl_pts, dk_ctrl = load_ctrl_pts(ctrl_path)
        print(f"  {len(ctrl_pts)} control points   "
              f"max |dk| = {np.abs(dk_ctrl).max():.4f} rad/mm")

        k_base = 2.0 * math.pi / args.unit
        padded, origin = build_sdf_from_ctrl(
            ctrl_pts, dk_ctrl, k_base, 0.5 * args.wall, args.voxel
        )

    mesh_and_export(padded, origin, args.voxel, stl_path, args.min_vol_frac)
    print(f"\nTotal time : {time.time() - t0:.1f} s")
    print(f"STL written: {stl_path}")


if __name__ == '__main__':
    main()
