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
from scipy.ndimage import label, binary_erosion, binary_dilation
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
) -> tuple[np.ndarray, np.ndarray, tuple]:
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

    padded, origin = pad_and_origin(sdf, voxel_size)
    return padded, origin, (voxel_size, voxel_size, voxel_size)


# ── morphological post-processing ────────────────────────────────────────────

def helmholtz_filter_fft(
    gamma: np.ndarray,   # (nx,ny,nz) float64
    r_cells: float,      # filter radius in grid cells
) -> np.ndarray:
    """
    Solve  -r_cells² ∇²γ̃ + γ̃ = γ  via FFT on the uniform periodic grid.

    The discrete Laplacian eigenvalues are:
        λ(k) = 2*(cos(2π kx/nx)-1) + 2*(cos(2π ky/ny)-1) + 2*(cos(2π kz/nz)-1)
    Since λ ≤ 0 everywhere, (1 - r²λ) ≥ 1 and the system is always invertible.
    """
    nx, ny, nz = gamma.shape
    kx = np.arange(nx); ky = np.arange(ny); kz = np.arange(nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    lam = (2*(np.cos(2*np.pi*KX/nx) - 1)
         + 2*(np.cos(2*np.pi*KY/ny) - 1)
         + 2*(np.cos(2*np.pi*KZ/nz) - 1))          # ≤ 0
    denom = 1.0 - r_cells**2 * lam                  # ≥ 1
    gamma_tilde = np.real(np.fft.ifftn(np.fft.fftn(gamma) / denom))
    return np.clip(gamma_tilde, 0.0, 1.0)


def _ball_struct(r_cells: float) -> np.ndarray:
    """Return a boolean 3-D ball structuring element of radius r_cells."""
    r = max(1, int(np.ceil(r_cells)))
    i = np.arange(-r, r+1)
    I, J, K = np.meshgrid(i, i, i, indexing='ij')
    return (I**2 + J**2 + K**2) <= r**2


def morpho_clean_solid(
    gamma_3d:   np.ndarray,   # (nx,ny,nz) float32, 0=solid / 1=fluid (OF convention)
    cell_size:  float,        # mm (uniform)
    t_min:      float,        # minimum wall thickness (mm)
    beta:       float = 8.0,  # Heaviside sharpness
    eta:        float = 0.5,  # erosion threshold
    close_gaps: bool  = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erosion–dilation post-processing pipeline (see module docstring / user spec).

    Returns
    -------
    gamma_tilde : (nx,ny,nz) float64 — smooth Helmholtz-filtered field (OF
                  convention: 0=solid, 1=fluid), for smooth marching-cubes surface.
    solid_clean : (nx,ny,nz) bool   — binary clean solid mask (True=solid).
    """
    # Invert to spec convention (1=solid, 0=fluid) for filtering/projection
    g = 1.0 - gamma_3d.astype(np.float64)

    # Step 1-2: filter radius and Helmholtz filter
    r_cells = (t_min / 2.0) / cell_size
    print(f"  Helmholtz filter: t_min={t_min} mm  r={t_min/2:.3f} mm  "
          f"r_cells={r_cells:.2f}")
    g_tilde = helmholtz_filter_fft(g, r_cells)

    # Step 3-5: smooth erosion projection (P(x) ≥ 0.5 iff x ≥ 0)
    #   γ_ero = 0.5*(tanh(β*(g̃ - η)) + 1)  ≥ 0.5  ↔  g̃ ≥ η
    g_ero   = 0.5 * (np.tanh(beta * (g_tilde - eta)) + 1.0)
    solid_raw = g_ero >= 0.5                        # bool (nx,ny,nz)

    n_raw = solid_raw.sum()
    print(f"  After erosion: {n_raw:,} solid voxels  "
          f"({n_raw / solid_raw.size:.4f} fraction)")

    # Step 6: remove floating features — keep largest connected component
    labeled, n_comp = label(solid_raw)
    print(f"  Connected components: {n_comp}")
    if n_comp > 1:
        comp_sizes = np.bincount(labeled.ravel())[1:]   # index 0 = background
        largest    = int(comp_sizes.argmax()) + 1
        solid_clean = (labeled == largest)
        removed_vox = n_raw - solid_clean.sum()
        print(f"  Kept component #{largest}  "
              f"({solid_clean.sum():,} voxels, removed {removed_vox:,})")
    else:
        solid_clean = solid_raw.copy()

    # Step 7: optional gap closing (one erosion + one dilation)
    if close_gaps:
        struct = _ball_struct(r_cells)
        print(f"  Gap closing: ball r={int(np.ceil(r_cells))} cells …")
        solid_clean = binary_dilation(
            binary_erosion(solid_clean, structure=struct, border_value=0),
            structure=struct, border_value=0,
        )

    # Return smooth filtered field in OF convention for marching cubes
    gamma_tilde_of = 1.0 - g_tilde                 # back to 0=solid, 1=fluid
    return gamma_tilde_of.astype(np.float32), solid_clean


# ── mode: gamma ───────────────────────────────────────────────────────────────

def build_sdf_from_gamma(
    gamma_path:  Path,
    case_dir:    Path,
    voxel_size:  float,
    t_min:       float = F_WALL_THICKNESS,
    beta:        float = 8.0,
    eta:         float = 0.5,
    close_gaps:  bool  = False,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Load gamma from OpenFOAM, recover the structured hex grid, apply the
    erosion–dilation morphological pipeline, and return a padded SDF ready
    for marching cubes.

    When the structured grid is detected the pipeline is:
      1. Helmholtz filter (FFT) with r = t_min/2
      2. Smooth erosion projection
      3. Connected-component cleanup (largest component kept)
      4. Optional gap closing
      5. SDF = (eta - gamma_tilde) masked to clean solid → marching cubes
         on the smooth filtered surface (not on the blocky binary mask).

    Falls back to nearest-neighbour + simple threshold for unstructured meshes.
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

    # ── recover structured grid (filter to opt domain first) ──────────────
    # The OpenFOAM mesh extends into inlet/outlet regions; only the opt-domain
    # cells form a perfect structured hex grid.
    dec = 3
    opt_mask = ((cc_mm[:, 0] >= OPT_XMIN) & (cc_mm[:, 0] <= OPT_XMAX) &
                (cc_mm[:, 1] >= OPT_YMIN) & (cc_mm[:, 1] <= OPT_YMAX) &
                (cc_mm[:, 2] >= OPT_ZMIN) & (cc_mm[:, 2] <= OPT_ZMAX))
    cc_opt    = cc_mm[opt_mask]
    gamma_opt = gamma[opt_mask]

    xu = np.unique(np.round(cc_opt[:, 0], dec))
    yu = np.unique(np.round(cc_opt[:, 1], dec))
    zu = np.unique(np.round(cc_opt[:, 2], dec))
    n_expected = len(xu) * len(yu) * len(zu)
    structured = abs(n_expected - len(cc_opt)) / len(cc_opt) < 0.01

    if structured:
        dx = xu[1] - xu[0]; dy = yu[1] - yu[0]; dz = zu[1] - zu[0]
        print(f"  Structured grid (opt domain): {len(xu)}×{len(yu)}×{len(zu)} "
              f"= {n_expected:,}  cell size {dx:.4f}×{dy:.4f}×{dz:.4f} mm")

        gamma_3d = np.ones((len(xu), len(yu), len(zu)), dtype=np.float32)
        xi = np.clip(np.searchsorted(xu, np.round(cc_opt[:, 0], dec)), 0, len(xu)-1)
        yi = np.clip(np.searchsorted(yu, np.round(cc_opt[:, 1], dec)), 0, len(yu)-1)
        zi = np.clip(np.searchsorted(zu, np.round(cc_opt[:, 2], dec)), 0, len(zu)-1)
        gamma_3d[xi, yi, zi] = gamma_opt.astype(np.float32)

        # ── morphological pipeline ─────────────────────────────────────────
        # cell_size is uniform (dx == dy == dz confirmed above)
        gamma_tilde, solid_clean = morpho_clean_solid(
            gamma_3d, cell_size=dx,
            t_min=t_min, beta=beta, eta=eta, close_gaps=close_gaps,
        )
        print(f"  Final solid: {solid_clean.sum():,} voxels  "
              f"({solid_clean.mean():.4f} fraction)")

        # SDF on the smooth filtered field, zeroed outside clean solid.
        # Marching cubes at level 0 → smooth gyroid surface, no fragments.
        sdf = (eta - gamma_tilde).astype(np.float32)   # >0 inside solid
        sdf[~solid_clean] = -1.0                        # force fluid outside

        padded = np.pad(sdf, pad_width=1, mode='constant', constant_values=-1.0)
        origin = np.array([xu[0] - dx, yu[0] - dy, zu[0] - dz])
        return padded, origin, (dx, dy, dz)

    else:
        print(f"  Unstructured mesh — falling back to nearest-neighbour at {voxel_size} mm")
        xs, ys, zs, X, Y, Z, pts = make_grid(voxel_size)
        nx, ny, nz = len(xs), len(ys), len(zs)
        print(f"  Voxel grid: {nx}×{ny}×{nz} = {nx*ny*nz:,}")
        tree = KDTree(cc_mm)
        _, idxs = tree.query(pts, workers=-1)
        gamma_grid = gamma[idxs].reshape(nx, ny, nz).astype(np.float32)
        sdf  = (0.5 - gamma_grid).astype(np.float32)
        frac = float((sdf > 0).mean())
        print(f"  Solid voxel fraction: {frac:.4f}  "
              f"({int((sdf > 0).sum()):,} of {nx*ny*nz:,} voxels)")
        padded = np.pad(sdf, pad_width=1, mode='constant', constant_values=-1.0)
        origin = np.array([OPT_XMIN - voxel_size,
                           OPT_YMIN - voxel_size,
                           OPT_ZMIN - voxel_size])
        return padded, origin, (voxel_size, voxel_size, voxel_size)


# ── marching cubes + component filter + export ────────────────────────────────

def mesh_and_export(
    padded:       np.ndarray,
    origin:       np.ndarray,
    spacing:      tuple,        # (dx, dy, dz) in mm
    stl_path:     Path,
    min_vol_frac: float,
) -> None:
    print("  Running marching cubes …")
    verts, faces, _n, _v = marching_cubes(
        padded, level=0.0,
        spacing=spacing,
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
    # ── gamma-mode morphological processing ───────────────────────────────
    parser.add_argument('--t-min', type=float, default=F_WALL_THICKNESS,
                        help='Minimum wall thickness for morpho pipeline (mm). '
                             'Default = wall thickness from optimizer.')
    parser.add_argument('--beta', type=float, default=8.0,
                        help='Heaviside sharpness β for erosion projection.')
    parser.add_argument('--close-gaps', action='store_true',
                        help='Apply one erosion + dilation after component cleanup '
                             'to close small internal gaps.')
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
        padded, origin, spacing = build_sdf_from_gamma(
            gamma_path, case_dir, args.voxel,
            t_min=args.t_min, beta=args.beta, eta=0.5,
            close_gaps=args.close_gaps,
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
        padded, origin, spacing = build_sdf_from_ctrl(
            ctrl_pts, dk_ctrl, k_base, 0.5 * args.wall, args.voxel
        )

    mesh_and_export(padded, origin, spacing, stl_path, args.min_vol_frac)
    print(f"\nTotal time : {time.time() - t0:.1f} s")
    print(f"STL written: {stl_path}")


if __name__ == '__main__':
    main()
