#!/usr/bin/env python3
"""
Visualise the Gyroid SDF at a fixed z-slice.

Reads a gyroid_ctrl_pts_*.txt file, reconstructs the spatially-varying
frequency field via RBF interpolation, and plots:
  - Left  : |G| - half_thickness  (SDF, diverging colormap; red=solid, blue=fluid)
  - Right : smooth gamma field    (sigmoid of SDF; black=solid, white=fluid)

Both panels show the zero-crossing of the SDF (wall midline) as a contour.

Usage
-----
    python visualise_gyroid_slice.py gyroid_ctrl_pts_checkpoint.txt
    python visualise_gyroid_slice.py gyroid_ctrl_pts_checkpoint.txt --z 5.0 --nx 3000
    python visualise_gyroid_slice.py gyroid_ctrl_pts_checkpoint.txt --unit 1.2 --wall 0.25
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import RBFInterpolator
from scipy.special import expit

_GYROID_GRAD_MAX = math.sqrt(3)   # max |∇G|/k on the gyroid surface


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_ctrl_pts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (N,3), dk (N,3)) from a gyroid_ctrl_pts_*.txt file."""
    data = np.loadtxt(path, comments='#')
    return data[:, :3], data[:, 3:6]


# ── Frequency field ───────────────────────────────────────────────────────────

def rbf_freq_at_pts(query_pts: np.ndarray,
                    ctrl_pts:  np.ndarray,
                    dk_ctrl:   np.ndarray,
                    k_base:    float) -> np.ndarray:
    """
    Evaluate spatially-varying wavenumbers at query_pts (N,3).
    Returns freq (N,3) = k_base + RBF(dk)(query_pts) for each axis.
    Uses the same thin-plate-spline RBF as the optimizer.
    """
    freq = np.empty((len(query_pts), 3), dtype=np.float64)
    for ax in range(3):
        rbf = RBFInterpolator(ctrl_pts, dk_ctrl[:, ax],
                              kernel='thin_plate_spline', degree=1)
        freq[:, ax] = k_base + rbf(query_pts)
    return freq


# ── Gyroid SDF ────────────────────────────────────────────────────────────────

def gyroid_sdf(pts_mm: np.ndarray,
               freq_mm: np.ndarray,
               half_thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Gyroid SDF and raw G value.

    G(x,y,z) = sin(kx·x)·cos(ky·y) + sin(ky·y)·cos(kz·z) + sin(kz·z)·cos(kx·x)
    SDF       = |G| - half_thickness

    Returns (sdf (N,), G (N,)).
    """
    x = pts_mm[:, 0]; y = pts_mm[:, 1]; z = pts_mm[:, 2]
    kx = freq_mm[:, 0]; ky = freq_mm[:, 1]; kz = freq_mm[:, 2]
    G = (np.sin(kx * x) * np.cos(ky * y)
       + np.sin(ky * y) * np.cos(kz * z)
       + np.sin(kz * z) * np.cos(kx * x))
    return np.abs(G) - half_thickness, G


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='High-resolution Gyroid SDF slice visualiser.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('ctrl_pts',
                        help='gyroid_ctrl_pts_*.txt produced by the optimizer')
    parser.add_argument('--z',       type=float, default=8.0,
                        help='z-position of the slice (mm)')
    parser.add_argument('--xmin',    type=float, default=0.0,
                        help='x lower bound (mm)')
    parser.add_argument('--xmax',    type=float, default=4.0,
                        help='x upper bound (mm)')
    parser.add_argument('--ymin',    type=float, default=0.0,
                        help='y lower bound (mm)')
    parser.add_argument('--ymax',    type=float, default=2.5,
                        help='y upper bound (mm)')
    parser.add_argument('--nx',      type=int,   default=2000,
                        help='Grid resolution in x (y is set proportionally)')
    parser.add_argument('--unit',    type=float, default=1.5,
                        help='Gyroid cell size in mm → k_base = 2π/unit')
    parser.add_argument('--wall',    type=float, default=0.30,
                        help='Minimum physical wall thickness (mm)')
    parser.add_argument('--epsilon', type=float, default=0.04,
                        help='Smooth-Heaviside sharpness (mm)')
    parser.add_argument('--dpi',     type=int,   default=200,
                        help='Output DPI')
    parser.add_argument('--output',  default=None,
                        help='Output filename (default: sdf_slice_z<z>mm.png)')
    args = parser.parse_args()

    ctrl_path = Path(args.ctrl_pts)
    if not ctrl_path.exists():
        sys.exit(f'ERROR: file not found: {ctrl_path}')

    # ── Derived parameters ────────────────────────────────────────────────────
    k_base         = 2.0 * math.pi / args.unit
    half_thickness = 0.5 * args.wall * k_base * _GYROID_GRAD_MAX

    print(f"Parameters:")
    print(f"  unit size       = {args.unit:.3f} mm  →  k_base = {k_base:.4f} rad/mm")
    print(f"  wall_thickness  = {args.wall:.4f} mm  →  half_thickness = {half_thickness:.4f} (G-units)")
    print(f"  epsilon         = {args.epsilon:.4f} mm")
    print(f"  z-slice         = {args.z:.3f} mm")

    # ── Load control points ───────────────────────────────────────────────────
    ctrl_pts_mm, dk_ctrl = load_ctrl_pts(ctrl_path)
    n_ctrl = len(ctrl_pts_mm)
    print(f"\nLoaded {n_ctrl} control points from '{ctrl_path.name}'")
    print(f"  dk range: [{dk_ctrl.min():.4f}, {dk_ctrl.max():.4f}] rad/mm")

    # ── Build evaluation grid ─────────────────────────────────────────────────
    nx = args.nx
    aspect = (args.ymax - args.ymin) / (args.xmax - args.xmin)
    ny = max(4, round(nx * aspect))

    xs = np.linspace(args.xmin, args.xmax, nx)   # (nx,)
    ys = np.linspace(args.ymin, args.ymax, ny)   # (ny,)

    # Standard meshgrid: XX[j,i] = xs[i], YY[j,i] = ys[j]  →  shape (ny, nx)
    XX, YY = np.meshgrid(xs, ys)
    ZZ     = np.full_like(XX, args.z)
    pts    = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    total = nx * ny
    print(f"\nGrid: {nx} × {ny} = {total:,} points at z = {args.z} mm")
    print("  Building RBF frequency field ...")

    # ── Evaluate fields ───────────────────────────────────────────────────────
    freq_mm = rbf_freq_at_pts(pts, ctrl_pts_mm, dk_ctrl, k_base)

    print("  Computing Gyroid SDF ...")
    sdf_flat, G_flat = gyroid_sdf(pts, freq_mm, half_thickness)

    # Reshape to (ny, nx) — correct orientation for imshow/contour
    SDF   = sdf_flat.reshape(ny, nx)
    G_map = G_flat.reshape(ny, nx)
    gamma = expit(sdf_flat / args.epsilon).reshape(ny, nx)

    # Effective local wavenumber magnitude (useful diagnostic)
    k_eff = np.linalg.norm(freq_mm, axis=1).reshape(ny, nx)

    # ── Statistics ────────────────────────────────────────────────────────────
    solid_frac = float((gamma < 0.5).mean())
    print(f"\n  SDF range : [{SDF.min():.4f}, {SDF.max():.4f}]")
    print(f"  G   range : [{G_map.min():.4f}, {G_map.max():.4f}]")
    print(f"  k_eff range: [{k_eff.min():.4f}, {k_eff.max():.4f}] rad/mm")
    print(f"  solid fraction (γ<0.5): {solid_frac:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5.5))
    fig.patch.set_facecolor('#1a1a2e')

    axes = fig.subplots(1, 3)

    extent = [args.xmin, args.xmax, args.ymin, args.ymax]
    contour_kw = dict(levels=[0.0], colors='white', linewidths=0.9, alpha=0.85)

    # ── Panel 1: raw SDF ─────────────────────────────────────────────────────
    ax = axes[0]
    vabs = float(np.percentile(np.abs(SDF), 97))
    im1 = ax.imshow(SDF, origin='lower', extent=extent, aspect='equal',
                    cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                    interpolation='bilinear')
    ax.contour(xs, ys, SDF, **contour_kw)
    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label('|G| − t  (SDF)', color='white', fontsize=9)
    cb1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')
    ax.set_title(f'SDF = |G| − half_thickness\n(red=solid, blue=fluid)',
                 color='white', fontsize=10)

    # ── Panel 2: smooth gamma ─────────────────────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(gamma, origin='lower', extent=extent, aspect='equal',
                    cmap='bone', vmin=0, vmax=1,
                    interpolation='bilinear')
    ax.contour(xs, ys, SDF, **contour_kw)
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label('γ  (0=solid, 1=fluid)', color='white', fontsize=9)
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
    ax.set_title(f'Smooth gamma  (ε = {args.epsilon} mm)\nWhite contour = SDF = 0',
                 color='white', fontsize=10)

    # ── Panel 3: local wavenumber magnitude ───────────────────────────────────
    ax = axes[2]
    im3 = ax.imshow(k_eff, origin='lower', extent=extent, aspect='equal',
                    cmap='plasma', interpolation='bilinear')
    ax.contour(xs, ys, SDF, **contour_kw)
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label('|k| (rad/mm)', color='white', fontsize=9)
    cb3.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color='white')
    ax.set_title(f'Effective wavenumber |k|\n(k_base = {k_base:.3f} rad/mm)',
                 color='white', fontsize=10)

    # ── Shared axis styling ───────────────────────────────────────────────────
    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        ax.set_xlabel('x (mm)', color='white', fontsize=9)
        ax.set_ylabel('y (mm)', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#555577')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

    fig.suptitle(
        f"Gyroid TPMS slice  z = {args.z:.1f} mm  |  "
        f"cell size = {args.unit} mm  |  "
        f"wall = {args.wall} mm  |  "
        f"solid frac = {solid_frac:.3f}  |  "
        f"grid {nx}×{ny}",
        color='white', fontsize=11, y=1.01,
    )
    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    out = args.output or f'sdf_slice_z{args.z:.1f}mm.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}  ({nx}×{ny} grid, {args.dpi} DPI)")
    plt.close(fig)


if __name__ == '__main__':
    main()
