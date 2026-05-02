"""
postprocess_gyroid_stl.py
=========================
Generate an STL of the optimised Gyroid heatsink lattice using LEAP71 /
PicoGK.  Reads the optimised RBF control-point frequency perturbations from
  gyroid_ctrl_pts_optimised.txt
and produces
  gyroid_optimised.stl

The bounding geometry is the exact optimisation domain:
  x ∈ [0, 4] mm   (channel width)
  y ∈ [0, 2.5] mm  (channel height)
  z ∈ [0, 10] mm   (channel length)

which matches the OpenFOAM blockMesh (convertToMeters 0.01, main block
vertices at x=[0,0.4], y=[0,0.25], z=[0,1.0] cm).

Run from anywhere:
    conda run -n LEAP71 python postprocess_gyroid_stl.py [--ctrl path/to/file.txt]

The LEAP71 viewer window will open briefly while voxels are built, then
close automatically when the STL has been written.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

# ── locate leap71_bindings.py ─────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_LEAP71_ROOT = Path('/workspace/LEAP71_version_KBE')
if not (_LEAP71_ROOT / 'leap71_bindings.py').exists():
    # fallback: search upward from script dir
    for _p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents]:
        if (_p / 'LEAP71_version_KBE' / 'leap71_bindings.py').exists():
            _LEAP71_ROOT = _p / 'LEAP71_version_KBE'
            break
    else:
        raise FileNotFoundError(
            'Cannot find leap71_bindings.py.  '
            'Set _LEAP71_ROOT to the correct path.'
        )
sys.path.insert(0, str(_LEAP71_ROOT))

# ── X11 / Mesa (headless-safe) ─────────────────────────────────────────────
os.environ.setdefault('DISPLAY', ':2')
_xdg = f'/tmp/xdg-runtime-{os.getuid()}'
os.makedirs(_xdg, mode=0o700, exist_ok=True)
os.environ['XDG_RUNTIME_DIR'] = _xdg
os.environ.setdefault('GDK_BACKEND', 'x11')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swr,llvmpipe'
os.environ['MESA_SHADER_CACHE'] = 'false'
os.environ['MESA_SHADER_CACHE_DIR'] = ''
os.environ['ZSTD_NBTHREADS'] = '1'

import leap71_bindings as leap71
from leap71_bindings import (
    Single, Vector3,
    LocalFrame, BaseBox,
    Sh, Cp,
    DelegateImplicit, Func,
    export_voxels_to_stl,
    run_in_library,
)

# ═══════════════════════════════════════════════════════════════════════════
# Optimisation / simulation domain  (must match gyroid_rbf_optimizer.py)
# ═══════════════════════════════════════════════════════════════════════════
# blockMeshDict: convertToMeters 0.01, main block x=[0,0.4] y=[0,0.25] z=[0,1] cm
OPT_XMIN, OPT_XMAX = 0.0,  4.0    # mm
OPT_YMIN, OPT_YMAX = 0.0,  2.5    # mm
OPT_ZMIN, OPT_ZMAX = 0.0, 10.0    # mm

# Bounding box centre and extents for BaseBox
BOX_SIZE   = (OPT_XMAX - OPT_XMIN,
              OPT_YMAX - OPT_YMIN,
              OPT_ZMAX - OPT_ZMIN)   # (width_x, depth_y, height_z) mm
BOX_CENTER = ((OPT_XMIN + OPT_XMAX) / 2,
              (OPT_YMIN + OPT_YMAX) / 2,
              (OPT_ZMIN + OPT_ZMAX) / 2)  # mm

# ═══════════════════════════════════════════════════════════════════════════
# Gyroid TPMS parameters  (must match gyroid_rbf_optimizer.py)
# ═══════════════════════════════════════════════════════════════════════════
F_UNIT_SIZE      = 1.5    # TPMS cell size (mm)
F_WALL_THICKNESS = 0.20   # wall thickness of the Gyroid solid shell (mm)

# ═══════════════════════════════════════════════════════════════════════════
# Voxelisation settings
# ═══════════════════════════════════════════════════════════════════════════
VOXEL_SIZE   = 0.05   # mm  (smaller = finer STL; 0.05 captures 0.20 mm walls well)
BAKE_SPACING = 0.4    # mm  (must match optimizer bake_spacing for identical results)


# ── RBF frequency field (mirrors optimizer's RBFFrequencyField) ───────────

class RBFFrequencyField:
    """
    3D spatial-frequency perturbation field:
        RBF thin-plate spline → dense baked grid → trilinear lookup.
    Stores (dk_x, dk_y, dk_z).  Actual wavenumber = k_base + dk(x,y,z).
    Mirrors the implementation in gyroid_rbf_optimizer.py.
    """

    def __init__(
        self,
        ctrl_pts:     np.ndarray,   # (N, 3) control-point positions (mm)
        dk_ctrl:      np.ndarray,   # (N, 3) frequency perturbations (rad/mm)
        bbox_min:     np.ndarray,   # (3,)   field extent min (mm)
        bbox_max:     np.ndarray,   # (3,)   field extent max (mm)
        bake_spacing: float,        # dense grid step (mm)
    ):
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)

        print(f'  Fitting RBF ({len(ctrl_pts)} control points) …')
        self._rbf = [
            RBFInterpolator(ctrl_pts, dk_ctrl[:, ax],
                            kernel='thin_plate_spline', degree=1)
            for ax in range(3)
        ]

        bake_axes = [np.arange(lo, hi + bake_spacing, bake_spacing)
                     for lo, hi in zip(self.bbox_min, self.bbox_max)]
        BX, BY, BZ = np.meshgrid(*bake_axes, indexing='ij')
        pts = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
        nx, ny, nz = BX.shape
        print(f'  Baking dense grid {nx}×{ny}×{nz} at {bake_spacing} mm …')
        baked = np.column_stack([rbf(pts) for rbf in self._rbf])
        self._grid      = baked.reshape(nx, ny, nz, 3)
        self._bake_axes = bake_axes
        self._nx, self._ny, self._nz = nx, ny, nz
        self._step = np.array([a[1] - a[0] if len(a) > 1 else 1.0 for a in bake_axes])
        print('  RBFFrequencyField ready.')

    def get_dk(self, x: float, y: float, z: float) -> tuple:
        """Single-point trilinear lookup; returns (dk_x, dk_y, dk_z) in rad/mm."""
        d = self._trilinear(x, y, z)
        return float(d[0]), float(d[1]), float(d[2])

    def _trilinear(self, x, y, z) -> np.ndarray:
        x = float(np.clip(x, self.bbox_min[0], self.bbox_max[0]))
        y = float(np.clip(y, self.bbox_min[1], self.bbox_max[1]))
        z = float(np.clip(z, self.bbox_min[2], self.bbox_max[2]))
        gx = (x - self._bake_axes[0][0]) / self._step[0]
        gy = (y - self._bake_axes[1][0]) / self._step[1]
        gz = (z - self._bake_axes[2][0]) / self._step[2]
        ix = min(int(gx), self._nx - 2)
        iy = min(int(gy), self._ny - 2)
        iz = min(int(gz), self._nz - 2)
        tx = gx - ix;  ty = gy - iy;  tz = gz - iz
        g = self._grid
        return (g[ix,   iy,   iz  ] * (1-tx)*(1-ty)*(1-tz)
              + g[ix+1, iy,   iz  ] *    tx *(1-ty)*(1-tz)
              + g[ix,   iy+1, iz  ] * (1-tx)*   ty *(1-tz)
              + g[ix+1, iy+1, iz  ] *    tx *   ty *(1-tz)
              + g[ix,   iy,   iz+1] * (1-tx)*(1-ty)*   tz
              + g[ix+1, iy,   iz+1] *    tx *(1-ty)*   tz
              + g[ix,   iy+1, iz+1] * (1-tx)*   ty *   tz
              + g[ix+1, iy+1, iz+1] *    tx *   ty *   tz)


def load_ctrl_pts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse schwarz_ctrl_pts_*.txt  (columns: x y z dx dy dz, all mm).
    Returns (ctrl_pts (N,3), disp_ctrl (N,3)).
    """
    data = np.loadtxt(path, comments='#')
    return data[:, :3], data[:, 3:]


def build_rbf_field(ctrl_pts: np.ndarray, disp_ctrl: np.ndarray) -> RBFDeformationField:
    """
    Build the RBF field with the same field bbox as the optimizer:
      field bbox = ctrl_pts bounding box  (the optimizer generates ctrl pts on a grid
      that already includes the amp_bound + 0.2 mm grow, so the ctrl-pt extents
      ARE the field extents).
    """
    field_min = ctrl_pts.min(axis=0)
    field_max = ctrl_pts.max(axis=0)
    print(f'  Field extent: {field_min} → {field_max} mm')
    return RBFDeformationField(ctrl_pts, disp_ctrl, field_min, field_max, BAKE_SPACING)


def build_and_export(ctrl_pts_path: Path, stl_path: Path) -> None:
    """Full pipeline: load ctrl pts → build field → voxelise → export STL."""

    # ── Load optimised control-point displacements ─────────────────────────
    print(f'Loading control points from {ctrl_pts_path} …')
    ctrl_pts, disp_ctrl = load_ctrl_pts(ctrl_pts_path)
    print(f'  {len(ctrl_pts)} control points,  '
          f'max |disp| = {np.abs(disp_ctrl).max():.4f} mm')

    # ── Build RBF deformation field ────────────────────────────────────────
    field = build_rbf_field(ctrl_pts, disp_ctrl)

    # ── PicoGK / LEAP71 geometry construction ──────────────────────────────
    k = 2.0 * math.pi / F_UNIT_SIZE   # Schwarz-P spatial frequency (rad/mm)

    def schwarz_rbf_sdf(vec) -> Single:
        """
        SDF for the deformed Schwarz-P surface.
        Mirrors ImplicitRandomizedSchwarzPrimitive.fSignedDistance:
          vecNoise  = field.vecGetData(vecPt)
          vecNewPt  = vecPt + vecNoise
          dist      = |cos(k*x) + cos(k*y) + cos(k*z)| - 0.5*wallThickness
        """
        x  = float(vec.X);  y  = float(vec.Y);  z  = float(vec.Z)
        dx, dy, dz = field.get_displacement(x, y, z)
        dist = (math.cos(k * (x + dx)) +
                math.cos(k * (y + dy)) +
                math.cos(k * (z + dz)))
        return Single(float(abs(dist) - 0.5 * F_WALL_THICKNESS))

    def task():
        # Bounding box — exactly the simulation/optimisation domain
        cx, cy, cz   = BOX_CENTER
        wx, dy, hz   = BOX_SIZE
        frame        = LocalFrame(leap71.vec3(cx, cy, cz))
        print(f'  Building bounding box: '
              f'x=[{OPT_XMIN}, {OPT_XMAX}]  '
              f'y=[{OPT_YMIN}, {OPT_YMAX}]  '
              f'z=[{OPT_ZMIN}, {OPT_ZMAX}] mm')
        vox_bounding = BaseBox(frame, hz, wx, dy).oConstructVoxels()

        # Implicit Schwarz-P surface with RBF deformation
        print('  Creating DelegateImplicit …')
        impl = DelegateImplicit(Func[Vector3, Single](schwarz_rbf_sdf))

        # Intersect bounding box with the implicit
        print('  Intersecting (each voxel calls back into Python) …')
        vox_solid = Sh.voxIntersectImplicit(vox_bounding, impl)

        # Export STL
        print(f'  Exporting STL → {stl_path}')
        export_voxels_to_stl(vox_solid, stl_path)
        print('  Done.')

    run_in_library(task, voxel_size=VOXEL_SIZE, output_dir=stl_path.parent,
                   headless=False)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate STL of optimised Schwarz-P heatsink via LEAP71.')
    parser.add_argument(
        '--ctrl', type=Path,
        default=_SCRIPT_DIR / 'app' / 'schwarz_ctrl_pts_optimised.txt',
        help='Path to schwarz_ctrl_pts_optimised.txt  (default: app/…)')
    parser.add_argument(
        '--out', type=Path,
        default=_SCRIPT_DIR / 'app' / 'schwarz_optimised.stl',
        help='Output STL path  (default: app/schwarz_optimised.stl)')
    parser.add_argument(
        '--voxel', type=float, default=VOXEL_SIZE,
        help=f'Voxel size in mm  (default: {VOXEL_SIZE})')
    args = parser.parse_args()

    VOXEL_SIZE = args.voxel   # allow CLI override

    t0 = time.time()
    build_and_export(args.ctrl, args.out)
    print(f'\nTotal time: {time.time() - t0:.1f} s')
    print(f'STL written to: {args.out}')
