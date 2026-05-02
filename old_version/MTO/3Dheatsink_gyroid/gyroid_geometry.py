"""
gyroid_geometry.py — standalone helper for generating and inspecting
the gyroid gamma field on the 3D heat-sink OpenFOAM mesh.

Useful for:
  • Visualising what the Gyroid looks like before optimising.
  • Generating the initial gamma field to seed the first OpenFOAM iteration.
  • Loading a VDB from the LEAP71 notebook and sampling onto the OpenFOAM mesh
    as an alternative to the analytic SDF approach.

Usage examples
--------------
# Generate initial (uniform-frequency) Gyroid gamma and write to app/0/gamma
python gyroid_geometry.py --mode init --case app

# Load an existing VDB from the LEAP71 notebook and map it onto the mesh
python gyroid_geometry.py --mode vdb --vdb ../../LEAP71_version_KBE/Examples/ImplicitGyroid_Python.vdb --case app

# Visualise the gamma field as a 2D slice plot (requires matplotlib)
python gyroid_geometry.py --mode plot --case app --time 10
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

# reuse I/O helpers from the optimizer module
sys.path.insert(0, str(Path(__file__).parent))
from gyroid_rbf_optimizer import (
    get_cell_centers_mm,
    write_gamma_field,
    read_scalar_field,
    gyroid_sdf_batch,
    gamma_from_sdf,
    OPT_XMIN, OPT_XMAX,
    OPT_YMIN, OPT_YMAX,
    OPT_ZMIN, OPT_ZMAX,
    F_UNIT_SIZE, F_WALL_THICKNESS, SDF_EPSILON,
)


# ── VDB sampling ──────────────────────────────────────────────────────────────

def gamma_from_vdb(
    vdb_path:        Path,
    cell_centers_mm: np.ndarray,
    epsilon:         float = SDF_EPSILON,
) -> np.ndarray:
    """
    Map a VDB signed-distance field (from the LEAP71 PicoGK notebook) onto the
    OpenFOAM mesh cell centres using trilinear interpolation.

    The VDB stores the SDF of the Gyroid geometry:
      SDF < 0  →  solid wall  →  gamma = 0
      SDF > 0  →  fluid       →  gamma = 1

    Requires the vdb_numpy extension built in LEAP71_version_KBE/vdb_reader.
    """
    vdb_reader_dir = Path(__file__).parents[4] / 'LEAP71_version_KBE' / 'vdb_reader'
    if str(vdb_reader_dir) not in sys.path:
        sys.path.insert(0, str(vdb_reader_dir))
    try:
        import vdb_numpy
    except ImportError as exc:
        raise ImportError(
            f"vdb_numpy not found in {vdb_reader_dir}. "
            "Build with:  cd <vdb_reader_dir> && python setup.py build_ext --inplace"
        ) from exc

    vdb_path = vdb_path.resolve()
    print(f"  Reading VDB: {vdb_path} …")
    arr, origin, voxel_size_mm = vdb_numpy.read_vdb_as_numpy(str(vdb_path))
    # arr shape: (nz, ny, nx)  — stored as (slow, …, fast)
    nz, ny, nx = arr.shape
    ox, oy, oz = origin   # mm

    print(f"    VDB shape  : {arr.shape},  voxel_size = {voxel_size_mm} mm")
    print(f"    VDB origin : ({ox:.2f}, {oy:.2f}, {oz:.2f}) mm")

    # Trilinear interpolation of VDB onto cell centres
    # Cell centres are in mm; VDB axes: x along dim-2, y dim-1, z dim-0
    # (PicoGK VDB convention: array[iz, iy, ix])
    pts = cell_centers_mm                          # (N, 3)
    gx  = (pts[:, 0] - ox) / voxel_size_mm        # fractional index along x
    gy  = (pts[:, 1] - oy) / voxel_size_mm
    gz  = (pts[:, 2] - oz) / voxel_size_mm

    # Clamp to valid range
    gx  = np.clip(gx, 0, nx - 1.001)
    gy  = np.clip(gy, 0, ny - 1.001)
    gz  = np.clip(gz, 0, nz - 1.001)

    ix  = gx.astype(int);  tx = gx - ix
    iy  = gy.astype(int);  ty = gy - iy
    iz  = gz.astype(int);  tz = gz - iz

    ix1 = np.minimum(ix + 1, nx - 1)
    iy1 = np.minimum(iy + 1, ny - 1)
    iz1 = np.minimum(iz + 1, nz - 1)

    # Trilinear interpolation (arr indexed [iz, iy, ix])
    sdf = (arr[iz,  iy,  ix ] * (1-tx)*(1-ty)*(1-tz)
         + arr[iz,  iy,  ix1] *    tx *(1-ty)*(1-tz)
         + arr[iz,  iy1, ix ] * (1-tx)*   ty *(1-tz)
         + arr[iz,  iy1, ix1] *    tx *   ty *(1-tz)
         + arr[iz1, iy,  ix ] * (1-tx)*(1-ty)*   tz
         + arr[iz1, iy,  ix1] *    tx *(1-ty)*   tz
         + arr[iz1, iy1, ix ] * (1-tx)*   ty *   tz
         + arr[iz1, iy1, ix1] *    tx *   ty *   tz)

    gamma = gamma_from_sdf(sdf, epsilon)
    print(f"    gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]  "
          f"solid_frac = {1 - gamma.mean():.4f}")
    return gamma


# ── Initial analytic Gyroid gamma ─────────────────────────────────────────────

def initial_gamma(
    cell_centers_mm: np.ndarray,
    k_base:          float = 2.0 * math.pi / F_UNIT_SIZE,
    half_thickness:  float = 0.5 * F_WALL_THICKNESS,
    epsilon:         float = SDF_EPSILON,
) -> np.ndarray:
    """Uniform-frequency Gyroid gamma (zero RBF frequency perturbation everywhere)."""
    freq_mm = np.full_like(cell_centers_mm, k_base)   # (N, 3) all equal k_base
    sdf     = gyroid_sdf_batch(cell_centers_mm, freq_mm, half_thickness)
    return gamma_from_sdf(sdf, epsilon)


# ── Slice visualisation ───────────────────────────────────────────────────────

def plot_gamma_slice(
    cell_centers_mm: np.ndarray,
    gamma:           np.ndarray,
    axis:            str = 'z',
    slice_frac:      float = 0.5,
    save_path:       Path | None = None,
) -> None:
    """
    Render a 2-D slice through the gamma field using matplotlib scatter.

    axis        : 'x', 'y', or 'z'  — axis normal to the slice
    slice_frac  : fraction along the axis for the slice (0–1)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    ax_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    ax_min = cell_centers_mm[:, ax_idx].min()
    ax_max = cell_centers_mm[:, ax_idx].max()
    target = ax_min + slice_frac * (ax_max - ax_min)
    tol    = (ax_max - ax_min) * 0.03   # pick cells within 3 % of range

    mask = np.abs(cell_centers_mm[:, ax_idx] - target) < tol
    if mask.sum() < 5:
        print(f"  No cells near {axis}={target:.2f} mm; widening tolerance.")
        tol *= 3
        mask = np.abs(cell_centers_mm[:, ax_idx] - target) < tol

    remaining = [i for i in range(3) if i != ax_idx]
    h_idx, v_idx = remaining
    labels = ['X (mm)', 'Y (mm)', 'Z (mm)']

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        cell_centers_mm[mask, h_idx],
        cell_centers_mm[mask, v_idx],
        c=gamma[mask], cmap='RdBu_r', s=2, vmin=0, vmax=1
    )
    plt.colorbar(sc, ax=ax, label='gamma (1=fluid, 0=solid)')
    ax.set_xlabel(labels[h_idx])
    ax.set_ylabel(labels[v_idx])
    ax.set_title(f'Gyroid gamma  |  {axis.upper()}={target:.2f} mm slice')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Gyroid geometry helper for the MTO case.')
    parser.add_argument('--mode', choices=['init', 'vdb', 'plot'], default='init',
                        help='Operation mode: init | vdb | plot')
    parser.add_argument('--case',    default='app', help='OpenFOAM case directory')
    parser.add_argument('--vdb',     default=None,  help='Path to VDB file (mode=vdb)')
    parser.add_argument('--time',    default='0',   help='Time directory to write/read')
    parser.add_argument('--unit',    type=float, default=F_UNIT_SIZE,
                        help='TPMS cell size (mm)')
    parser.add_argument('--wall',    type=float, default=F_WALL_THICKNESS,
                        help='Wall thickness (mm)')
    parser.add_argument('--epsilon', type=float, default=SDF_EPSILON,
                        help='Smooth-Heaviside sharpness (mm)')
    parser.add_argument('--axis',    default='z',   help='Slice axis for plot mode')
    parser.add_argument('--frac',    type=float, default=0.5,
                        help='Slice position fraction 0-1 for plot mode')
    parser.add_argument('--save-plot', default=None,
                        help='Save plot to this path instead of showing interactively')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    case_dir   = (script_dir / args.case).resolve()

    if args.mode in ('init', 'vdb'):
        cc_mm = get_cell_centers_mm(case_dir)

        if args.mode == 'init':
            k              = 2.0 * math.pi / args.unit
            half_thickness = 0.5 * args.wall
            gamma          = initial_gamma(cc_mm, k, half_thickness, args.epsilon)
            print(f"  Uniform-frequency Gyroid  solid_frac = {1 - gamma.mean():.4f}")
        else:
            if args.vdb is None:
                sys.exit("ERROR: --vdb required for mode=vdb")
            gamma = gamma_from_vdb(Path(args.vdb), cc_mm, args.epsilon)

        out_path = case_dir / args.time / 'gamma'
        write_gamma_field(out_path, gamma, args.time)
        print(f"  Written → {out_path}")

    elif args.mode == 'plot':
        cc_mm  = get_cell_centers_mm(case_dir)
        gamma_path = case_dir / args.time / 'gamma'
        if not gamma_path.exists():
            sys.exit(f"ERROR: {gamma_path} not found. Run --mode init first.")
        gamma = read_scalar_field(gamma_path)
        save  = Path(args.save_plot) if args.save_plot else None
        plot_gamma_slice(cc_mm, gamma, axis=args.axis, slice_frac=args.frac,
                         save_path=save)


if __name__ == '__main__':
    main()
