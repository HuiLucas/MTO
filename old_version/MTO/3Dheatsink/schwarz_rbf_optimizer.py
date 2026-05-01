"""
Schwarz Primitive RBF Optimizer for the 3D heat-sink MTO case.

Replaces the internal SIMP/MMA density parameterisation with a Schwarz-P TPMS
geometry parameterised by RBF control-point displacements (identical to the
approach in LEAP71_version_KBE/notebooks/ImplicitRandomSchwarzPrimitive_Python.ipynb).

Design variables:
    displacement (dx, dy, dz) at each RBF control point on a regular grid.
    Total: n_ctrl * 3 scalars.

Each outer optimisation iteration:
    1. Given current control-point displacements, build an RBFDeformationField
       (thin-plate-spline, same as the notebook).
    2. Evaluate the Schwarz-P SDF at every mesh cell centre.
    3. Map SDF → gamma via a smooth Heaviside (sigmoid).  gamma=1 → fluid,
       gamma=0 → solid wall, matching the OpenFOAM convention in this case.
    4. Write the gamma field to the OpenFOAM case directory.
    5. Set controlDict startTime/endTime so the solver runs exactly ONE outer
       iteration, then exit.
    6. Run the OpenFOAM MTO_TF solver (serial or parallel).
    7. Read fsens = dJ/d(gamma) from the solver output.  fsens is written by
       MTO_TF.C (1-line modification: fsens.write() in the writeTime block).
    8. Chain-rule: dJ/d(ctrl_displacements) via analytic SDF derivatives and
       the RBF evaluation Jacobian.
    9. Feed (objective, gradient) to scipy L-BFGS-B for the next step.

Usage
-----
    python schwarz_rbf_optimizer.py [--case app] [--iters 50] [--parallel 20]

Dependencies
------------
    numpy, scipy (standard scientific Python)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid

# ── Physical / geometry parameters ─────────────────────────────────────────────
# blockMeshDict uses convertToMeters 0.01  →  mesh coords in metres
MESH_UNIT_TO_MM = 10.0        # 0.01 m * 1000 mm/m  (mesh unit → mm)

# Optimization domain bounding box (mm) – derived from blockMeshDict vertices
# Main opt block: x=[0,4], y=[0.5,2.375], z=[0,10]  mm  (blocks 1+2 combined)
OPT_XMIN, OPT_XMAX = 0.0,  4.0
OPT_YMIN, OPT_YMAX = 0.0,  2.5
OPT_ZMIN, OPT_ZMAX = 0.0, 10.0

# Schwarz-P TPMS parameters (mm)
F_UNIT_SIZE      = 1.5    # TPMS cell size – ~2-3 cells fit in 4 mm domain width
F_WALL_THICKNESS = 0.20   # solid wall thickness
SDF_EPSILON      = 0.04   # smooth-Heaviside sharpness (smaller = sharper, but less gradient)

# RBF control-point grid (mm)
CONTROL_SPACING  = 2.0    # spacing between control points on the regular grid
AMPLITUDE_INIT   = 0.0    # initial displacement (0 = undeformed Schwarz-P)
AMPLITUDE_BOUND  = 0.6    # ±bound for each displacement component (mm)

# ── OpenFOAM field I/O ─────────────────────────────────────────────────────────

_GAMMA_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     |                                                 |
|   \\\\  /    A nd           |                                                 |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volScalarField;
    location    "{location}";
    object      gamma;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
{n}
(
"""

_GAMMA_FOOTER = """\
)
;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            zeroGradient;
    }
    force
    {
        type            zeroGradient;
    }
    wall
    {
        type            zeroGradient;
    }
    sym
    {
        type            symmetry;
    }
}


// ************************************************************************* //
"""


def write_gamma_field(path: Path, values: np.ndarray, time_str: str) -> None:
    """Write a gamma volScalarField in OpenFOAM ASCII format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _GAMMA_HEADER.format(location=time_str, n=len(values))
    with open(path, 'w') as f:
        f.write(header)
        for v in values:
            f.write(f"{v:.6g}\n")
        f.write(_GAMMA_FOOTER)


def read_scalar_field(path: Path) -> np.ndarray:
    """
    Parse an OpenFOAM ASCII volScalarField and return the internalField values.
    Handles both 'uniform <val>' and 'nonuniform List<scalar> N (...)' formats.
    """
    text = path.read_text()

    # uniform
    m = re.search(r'internalField\s+uniform\s+([0-9Ee.+\-]+)', text)
    if m:
        # we need N: look for it in the header or infer
        n_match = re.search(r'nonuniform List<scalar>\s+(\d+)', text)
        if n_match:
            n = int(n_match.group(1))
        else:
            raise ValueError(f"Cannot determine cell count from {path}")
        return np.full(n, float(m.group(1)))

    # nonuniform
    m = re.search(
        r'internalField\s+nonuniform List<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)',
        text, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse internalField in {path}")
    n = int(m.group(1))
    vals = np.fromstring(m.group(2), sep='\n', count=n)
    if len(vals) != n:
        vals = np.array([float(x) for x in m.group(2).split()], dtype=float)
    return vals


def read_vector_field(path: Path) -> np.ndarray:
    """
    Parse an OpenFOAM ASCII volVectorField internalField.
    Returns shape (N, 3).

    Handles both formats (count on same or next line after 'List<vector>'):
        internalField   nonuniform List<vector> N   ← count inline
        (                                           OR
        internalField   nonuniform List<vector>
        N                                           ← count on own line
        (
        (x0 y0 z0)
        ...
        )
        ;
    """
    _VEC_RE = re.compile(
        r'\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)'
        r'\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)'
        r'\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s*\)'
    )

    # State machine: SEEK_HEADER → SEEK_COUNT → SEEK_OPEN → READ → DONE
    SEEK_HEADER, SEEK_COUNT, SEEK_OPEN, READ = range(4)
    state = SEEK_HEADER
    n     = 0
    rows: list[tuple] = []

    with open(path, 'r') as fh:
        for line in fh:
            s = line.strip()

            if state == SEEK_HEADER:
                if 'nonuniform List<vector>' not in s:
                    continue
                # count may be inline: "nonuniform List<vector> 516096"
                m = re.search(r'nonuniform List<vector>\s+(\d+)', s)
                if m:
                    n     = int(m.group(1))
                    state = SEEK_OPEN
                else:
                    state = SEEK_COUNT   # count on the next non-blank line

            elif state == SEEK_COUNT:
                if s.isdigit():
                    n     = int(s)
                    state = SEEK_OPEN

            elif state == SEEK_OPEN:
                if s == '(':
                    state = READ

            elif state == READ:
                if s.startswith(')'):
                    break                # closing ')' or ');\n'
                vm = _VEC_RE.search(s)
                if vm:
                    rows.append((float(vm.group(1)),
                                 float(vm.group(2)),
                                 float(vm.group(3))))

    if n == 0:
        raise ValueError(f"Cannot find 'nonuniform List<vector>' in {path}")
    if len(rows) != n:
        raise ValueError(f"Expected {n} vectors but parsed {len(rows)} from {path}")
    return np.array(rows, dtype=np.float64)


def update_control_dict(system_dir: Path, start_time: int, end_time: int,
                         write_interval: int = 1) -> None:
    """Patch controlDict to run exactly end_time-start_time outer iterations."""
    cd_path = system_dir / 'controlDict'
    text = cd_path.read_text()
    text = re.sub(r'(startTime\s+)\S+;', rf'\g<1>{start_time};', text)
    text = re.sub(r'(endTime\s+)\S+;',   rf'\g<1>{end_time};',   text)
    text = re.sub(r'(writeInterval\s+)\S+;', rf'\g<1>{write_interval};', text)
    cd_path.write_text(text)


def get_latest_time(case_dir: Path) -> int:
    """Return the largest numeric time directory present in case_dir."""
    times = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                times.append(int(d.name))
            except ValueError:
                pass
    return max(times) if times else 0


def get_cell_centers_mm(case_dir: Path, of_binary: str = 'postProcess') -> np.ndarray:
    """
    Generate cell-centre coordinates (mm) using postProcess -func writeCellCentres.
    Returns shape (N_cells, 3) in mm.
    Caches result to case_dir/cell_centers.npy.
    """
    cache = case_dir / 'cell_centers_mm.npy'
    if cache.exists():
        print(f"  Loading cached cell centres from {cache}")
        return np.load(cache)

    print("  Running postProcess -func writeCellCentres …")
    subprocess.run(
        [of_binary, '-func', 'writeCellCentres', '-time', '0', '-case', str(case_dir)],
        check=True, capture_output=True
    )

    # field may be at 0/C or 0/ccx etc. depending on OpenFOAM version
    cc_path = case_dir / '0' / 'C'
    if not cc_path.exists():
        # try cellCentre vector
        cc_path = next(case_dir.glob('0/ccx'), None)
        if cc_path is None:
            raise FileNotFoundError(
                "postProcess writeCellCentres did not produce 0/C; "
                "check your OpenFOAM installation."
            )

    cc_m = read_vector_field(cc_path)          # shape (N, 3) in metres
    cc_mm = cc_m * 1000.0                       # convert metres → mm
    np.save(cache, cc_mm)
    print(f"  Cell centres: {cc_mm.shape[0]:,} cells, range "
          f"x=[{cc_mm[:,0].min():.2f},{cc_mm[:,0].max():.2f}] "
          f"y=[{cc_mm[:,1].min():.2f},{cc_mm[:,1].max():.2f}] "
          f"z=[{cc_mm[:,2].min():.2f},{cc_mm[:,2].max():.2f}] mm")
    np.save(cache, cc_mm)
    return cc_mm


# ── RBF deformation field (mirrors notebook Cell 4) ───────────────────────────

class RBFDeformationField:
    """
    3-D displacement field: RBF thin-plate spline → dense baked grid → trilinear lookup.
    Identical to the class in ImplicitRandomSchwarzPrimitive_Python.ipynb.
    """

    def __init__(
        self,
        ctrl_pts:   np.ndarray,   # (N_ctrl, 3) control-point positions (mm)
        disp_ctrl:  np.ndarray,   # (N_ctrl, 3) displacements at control points (mm)
        bbox_min:   np.ndarray,   # (3,) field extent min (mm)
        bbox_max:   np.ndarray,   # (3,) field extent max (mm)
        bake_spacing: float = 0.5,
    ):
        self.ctrl_pts  = ctrl_pts
        self.disp_ctrl = disp_ctrl
        self.bbox_min  = np.asarray(bbox_min, dtype=float)
        self.bbox_max  = np.asarray(bbox_max, dtype=float)
        amplitude      = float(np.abs(disp_ctrl).max()) if disp_ctrl.size else 0.0

        # fit one RBF interpolator per axis
        self._rbf = [
            RBFInterpolator(ctrl_pts, disp_ctrl[:, ax],
                            kernel='thin_plate_spline', degree=1)
            for ax in range(3)
        ]

        # bake to dense grid for fast per-voxel lookup
        bake_axes = [np.arange(lo, hi + bake_spacing, bake_spacing)
                     for lo, hi in zip(self.bbox_min, self.bbox_max)]
        BX, BY, BZ = np.meshgrid(*bake_axes, indexing='ij')
        bake_pts = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
        nx, ny, nz = BX.shape
        baked = np.column_stack([rbf(bake_pts) for rbf in self._rbf])
        self._grid      = baked.reshape(nx, ny, nz, 3)
        self._bake_axes = bake_axes
        self._nx, self._ny, self._nz = nx, ny, nz
        self._step = np.array([a[1] - a[0] if len(a) > 1 else 1.0 for a in bake_axes])

    def get_displacement_batch(self, pts_mm: np.ndarray) -> np.ndarray:
        """Vectorised trilinear lookup; pts_mm is (N, 3) in mm, returns (N, 3)."""
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
        g  = self._grid
        return (g[ix,   iy,   iz  ] * (1-tx)*(1-ty)*(1-tz)
              + g[ix+1, iy,   iz  ] *    tx *(1-ty)*(1-tz)
              + g[ix,   iy+1, iz  ] * (1-tx)*   ty *(1-tz)
              + g[ix+1, iy+1, iz  ] *    tx *   ty *(1-tz)
              + g[ix,   iy,   iz+1] * (1-tx)*(1-ty)*   tz
              + g[ix+1, iy,   iz+1] *    tx *(1-ty)*   tz
              + g[ix,   iy+1, iz+1] * (1-tx)*   ty *   tz
              + g[ix+1, iy+1, iz+1] *    tx *   ty *   tz)


def build_field(ctrl_pts_mm: np.ndarray,
                disp_ctrl:   np.ndarray,
                bbox_min_mm: np.ndarray,
                bbox_max_mm: np.ndarray,
                bake_spacing: float = 0.5) -> RBFDeformationField:
    """Construct an RBFDeformationField from control-point displacements."""
    return RBFDeformationField(ctrl_pts_mm, disp_ctrl,
                               bbox_min_mm, bbox_max_mm, bake_spacing)


# ── Schwarz-P SDF and gamma ────────────────────────────────────────────────────

def schwarz_sdf_batch(pts_mm: np.ndarray,
                      disp_mm: np.ndarray,
                      k: float,
                      half_thickness: float) -> np.ndarray:
    """
    Vectorised Schwarz-P SDF at an array of points.

    SDF < 0  →  inside solid wall  →  gamma = 0 (solid in OpenFOAM convention)
    SDF > 0  →  fluid channel      →  gamma = 1 (fluid)

    Parameters
    ----------
    pts_mm        : (N, 3) cell-centre positions (mm)
    disp_mm       : (N, 3) RBF displacement field at each cell (mm)
    k             : 2*pi / unit_size  (rad/mm)
    half_thickness: 0.5 * wall_thickness (mm)
    """
    xd = pts_mm[:, 0] + disp_mm[:, 0]
    yd = pts_mm[:, 1] + disp_mm[:, 1]
    zd = pts_mm[:, 2] + disp_mm[:, 2]
    S  = np.cos(k * xd) + np.cos(k * yd) + np.cos(k * zd)
    return np.abs(S) - half_thickness


def gamma_from_sdf(sdf: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Smooth Heaviside: gamma = sigmoid(sdf / epsilon).
    gamma → 1 (fluid) for sdf >> 0, gamma → 0 (solid) for sdf << 0.
    """
    return expit(sdf / epsilon)


def dgamma_dsdf_vals(sdf: np.ndarray, epsilon: float) -> np.ndarray:
    """d(gamma)/d(sdf) = sigmoid' / epsilon."""
    g = gamma_from_sdf(sdf, epsilon)
    return g * (1.0 - g) / epsilon


# ── RBF Jacobian (one-time precomputation) ─────────────────────────────────────

def build_rbf_jacobian(ctrl_pts_mm: np.ndarray,
                       cell_centers_mm: np.ndarray,
                       bake_spacing: float = 0.5) -> np.ndarray:
    """
    Precompute the (N_cells, N_ctrl) linear evaluation matrix W such that:

        disp_a(cell_j) = sum_l  W[j, l] * ctrl_disp_a[l]

    for each displacement axis a independently (W is the same for all axes).

    Built by evaluating the RBF with a unit impulse at each control point.
    """
    n_ctrl = len(ctrl_pts_mm)
    n_cells = len(cell_centers_mm)
    bbox_min = ctrl_pts_mm.min(axis=0)
    bbox_max = ctrl_pts_mm.max(axis=0)

    print(f"  Building RBF Jacobian ({n_ctrl} ctrl pts × {n_cells:,} cells) …")
    W = np.empty((n_cells, n_ctrl), dtype=np.float64)
    I = np.eye(n_ctrl)
    for l in range(n_ctrl):
        rbf_l = RBFInterpolator(ctrl_pts_mm, I[:, l],
                                kernel='thin_plate_spline', degree=1)
        W[:, l] = rbf_l(cell_centers_mm).ravel()
    print("  RBF Jacobian ready.")
    return W


# ── Chain-rule gradient ────────────────────────────────────────────────────────

def chain_rule_gradient(
    fsens:           np.ndarray,   # (N,)  dJ/d(gamma)  from OpenFOAM
    pts_mm:          np.ndarray,   # (N,3) cell centres (mm)
    disp_mm:         np.ndarray,   # (N,3) RBF displacement at each cell (mm)
    sdf:             np.ndarray,   # (N,)  Schwarz SDF values
    k:               float,        # 2*pi / unit_size
    epsilon:         float,        # smooth-Heaviside sharpness
    W:               np.ndarray,   # (N, N_ctrl) RBF Jacobian
) -> np.ndarray:
    """
    Compute dJ/d(ctrl_displacements) via the full chain rule:

        dJ/d(p_la) = sum_j  fsens_j
                           * (d gamma_j / d SDF_j)
                           * (d SDF_j  / d disp_{a,j})
                           * W[j, l]

    Returns shape (N_ctrl, 3)  – gradient w.r.t. (dx, dy, dz) of each ctrl pt.
    """
    xd = pts_mm[:, 0] + disp_mm[:, 0]
    yd = pts_mm[:, 1] + disp_mm[:, 1]
    zd = pts_mm[:, 2] + disp_mm[:, 2]
    S  = np.cos(k * xd) + np.cos(k * yd) + np.cos(k * zd)
    signS = np.sign(S)

    # d(SDF)/d(disp_a)  for a = x, y, z
    dSDF_dx = signS * (-np.sin(k * xd)) * k
    dSDF_dy = signS * (-np.sin(k * yd)) * k
    dSDF_dz = signS * (-np.sin(k * zd)) * k

    # d(gamma)/d(SDF)
    dgds = dgamma_dsdf_vals(sdf, epsilon)           # (N,)

    # Combined weight at each cell
    wx = fsens * dgds * dSDF_dx                     # (N,)
    wy = fsens * dgds * dSDF_dy
    wz = fsens * dgds * dSDF_dz

    # Reduce over cells: dJ/d(ctrl) = W.T @ w
    grad_x = W.T @ wx                               # (N_ctrl,)
    grad_y = W.T @ wy
    grad_z = W.T @ wz

    return np.stack([grad_x, grad_y, grad_z], axis=1)  # (N_ctrl, 3)


# ── OpenFOAM runner ────────────────────────────────────────────────────────────

def run_openfoam_one_step(case_dir: Path,
                          start_time: int,
                          end_time: int,
                          solver: str = 'MTO_TF',
                          n_procs: int = 1,
                          iter_num: int = 0) -> None:
    """
    Run the MTO_TF solver for exactly (end_time - start_time) outer iterations.
    Handles serial or parallel execution.
    """
    system_dir = case_dir / 'system'
    update_control_dict(system_dir, start_time, end_time, write_interval=1)

    # Run all subprocesses from the case directory so that the C++ solver's
    # std::remove("meanT.txt") etc. in SIMP_initialize.H finds the files.
    cwd = str(case_dir)

    if n_procs > 1:
        # Parallel workflow: decomposePar → mpirun → reconstructPar
        print(f"  decomposePar (time {start_time}) …")
        r = subprocess.run(
            ['decomposePar', '-time', str(start_time), '-force', '-case', str(case_dir)],
            capture_output=True, cwd=cwd
        )
        if r.returncode != 0:
            log = case_dir / f'log.decomposePar.iter{iter_num:03d}'
            log.write_bytes(r.stdout + b'\n' + r.stderr)
            raise RuntimeError(
                f"decomposePar failed (exit {r.returncode}). Log: {log}\n"
                + r.stderr.decode(errors='replace')[-2000:]
            )

        log_path = case_dir / f'log.{solver}.iter{iter_num:03d}'
        print(f"  mpirun -n {n_procs} {solver} … (log → {log_path.name})")
        with open(log_path, 'wb') as lf:
            r = subprocess.run(
                ['mpirun', '--oversubscribe', '-n', str(n_procs),
                 solver, '-parallel', '-case', str(case_dir)],
                stdout=lf, stderr=subprocess.STDOUT,
                cwd=cwd,
                start_new_session=True,  # isolate mpirun's process group from Python
            )
        if r.returncode != 0:
            # print last 40 lines of solver log so the error is visible immediately
            tail = log_path.read_bytes()[-4000:].decode(errors='replace')
            raise RuntimeError(
                f"{solver} (parallel) failed (exit {r.returncode}).\n"
                f"Last output from {log_path.name}:\n{tail}"
            )

        print(f"  reconstructPar (time {end_time}) …")
        r = subprocess.run(
            ['reconstructPar', '-time', str(end_time), '-case', str(case_dir)],
            capture_output=True, cwd=cwd
        )
        if r.returncode != 0:
            log = case_dir / f'log.reconstructPar.iter{iter_num:03d}'
            log.write_bytes(r.stdout + b'\n' + r.stderr)
            raise RuntimeError(
                f"reconstructPar failed (exit {r.returncode}). Log: {log}\n"
                + r.stderr.decode(errors='replace')[-2000:]
            )
    else:
        print(f"  Running {solver} (serial) for t={start_time}→{end_time} …")
        result = subprocess.run(
            [solver, '-case', str(case_dir)],
            capture_output=False, cwd=cwd
        )
        if result.returncode != 0:
            raise RuntimeError(f"{solver} exited with code {result.returncode}")


def read_objective(case_dir: Path) -> tuple[float, float]:
    """
    Read the latest mean temperature (primary objective) and DissPower (constraint).
    Returns (meanT, DissPower).
    meanT is the actual temperature functional minimised by the adjoint/fsens.
    DissPower is reported for monitoring only.
    """
    def _last(fname):
        p = case_dir / fname
        if not p.exists():
            return float('nan')
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        return float(lines[-1]) if lines else float('nan')

    return _last('meanT.txt'), _last('Disspower.txt')


# ── Main optimizer class ───────────────────────────────────────────────────────

class SchwarzRBFOptimizer:
    """
    Outer optimisation loop: RBF control-point displacements → OpenFOAM → gradient.
    """

    def __init__(
        self,
        case_dir:        Path,
        k:               float = 2.0 * math.pi / F_UNIT_SIZE,
        wall_thickness:  float = F_WALL_THICKNESS,
        epsilon:         float = SDF_EPSILON,
        control_spacing: float = CONTROL_SPACING,
        amp_bound:       float = AMPLITUDE_BOUND,
        bake_spacing:    float = 0.4,
        solver:          str   = 'MTO_TF',
        n_procs:         int   = 1,
        of_binary:       str   = 'postProcess',
    ):
        self.case_dir       = case_dir
        self.k              = k
        self.half_thickness = 0.5 * wall_thickness
        self.epsilon        = epsilon
        self.amp_bound      = amp_bound
        self.solver         = solver
        self.n_procs        = n_procs
        self._iter          = 0
        self._history: list[dict] = []

        # Optimization domain bounding box (mm)
        opt_min = np.array([OPT_XMIN, OPT_YMIN, OPT_ZMIN])
        opt_max = np.array([OPT_XMAX, OPT_YMAX, OPT_ZMAX])

        # Grow field bbox for RBF evaluation stability (mirrors notebook)
        field_min = opt_min - amp_bound - 0.2
        field_max = opt_max + amp_bound + 0.2

        # Regular control-point grid (cover grown bbox)
        axes = [np.arange(lo, hi + control_spacing, control_spacing)
                for lo, hi in zip(field_min, field_max)]
        GX, GY, GZ = np.meshgrid(*axes, indexing='ij')
        self.ctrl_pts_mm = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
        self.n_ctrl = len(self.ctrl_pts_mm)
        self.field_min = field_min
        self.field_max = field_max
        self.bake_spacing = bake_spacing

        print(f"Control points : {self.n_ctrl}  "
              f"(grid {GX.shape[0]}×{GX.shape[1]}×{GX.shape[2]})")

        # Cell centres
        self.cell_centers_mm = get_cell_centers_mm(case_dir, of_binary)
        self.n_cells = len(self.cell_centers_mm)

        # Precompute RBF Jacobian (expensive once, free afterwards)
        self.W = build_rbf_jacobian(self.ctrl_pts_mm, self.cell_centers_mm, bake_spacing)

        # Design variable bounds
        self.bounds = [(-amp_bound, amp_bound)] * (self.n_ctrl * 3)

        print(f"Optimiser ready. {self.n_ctrl * 3} design variables "
              f"(±{amp_bound} mm each).\n")

    # ── internals ─────────────────────────────────────────────────────────────

    def _displacements_to_field(self, x: np.ndarray) -> tuple[RBFDeformationField,
                                                               np.ndarray]:
        """Reshape flat x vector → (N_ctrl, 3), build field, return (field, disp_ctrl)."""
        disp_ctrl = x.reshape(self.n_ctrl, 3)
        field = build_field(
            self.ctrl_pts_mm, disp_ctrl,
            self.field_min, self.field_max, self.bake_spacing
        )
        return field, disp_ctrl

    def _gamma_from_x(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gamma, sdf, disp_mm from flat design vector."""
        field, _ = self._displacements_to_field(x)
        disp_mm  = field.get_displacement_batch(self.cell_centers_mm)
        sdf      = schwarz_sdf_batch(self.cell_centers_mm, disp_mm,
                                     self.k, self.half_thickness)
        gamma    = gamma_from_sdf(sdf, self.epsilon)
        return gamma, sdf, disp_mm

    # ── objective + gradient (called by scipy) ────────────────────────────────

    def objective_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        1. Evaluate Schwarz gamma from x.
        2. Write gamma to OpenFOAM.
        3. Run one OpenFOAM outer iteration.
        4. Read fsens and objective.
        5. Chain-rule → gradient w.r.t. x.
        """
        t0 = time.time()
        self._iter += 1
        print(f"\n{'─'*60}")
        print(f"Outer iteration {self._iter}")
        print(f"{'─'*60}")

        # Always run from time 0 → 1 so that p/U/T initial conditions (which
        # only exist in app/0/) are always available to the parallel solver.
        # gamma is written to app/0/, decomposed, then the solver runs 0→1.
        start_t = 0
        end_t   = 1

        # 1+2. Build and write gamma
        gamma, sdf, disp_mm = self._gamma_from_x(x)
        print(f"  gamma:  min={gamma.min():.3f}  max={gamma.max():.3f}  "
              f"mean={gamma.mean():.3f}  solid_frac={1-gamma.mean():.3f}")

        gamma_path = self.case_dir / '0' / 'gamma'
        write_gamma_field(gamma_path, gamma, '0')
        print(f"  gamma written → {gamma_path}")

        # 3. Run OpenFOAM
        run_openfoam_one_step(
            self.case_dir, start_t, end_t,
            self.solver, self.n_procs, iter_num=self._iter
        )

        # 4. Read fsens and objective
        fsens_path = self.case_dir / '1' / 'fsens'
        if not fsens_path.exists():
            raise FileNotFoundError(
                f"fsens not found at {fsens_path}.\n"
                "Make sure MTO_TF.C has been modified to call fsens.write() "
                "and the solver has been recompiled."
            )
        fsens = read_scalar_field(fsens_path)

        # meanT is the primary objective minimised by the adjoint (fsens = dJ/dgamma).
        # DissPower is a constraint, not the objective — using it caused
        # the gradient and function value to be inconsistent.
        meanT, dissPower = read_objective(self.case_dir)

        vol_use = 1.0 - gamma.mean()
        print(f"  J (meanT)      = {meanT:.6g}")
        print(f"  DissPower      = {dissPower:.6g}  (constraint, not minimised)")
        print(f"  solid_fraction = {vol_use:.4f}")
        print(f"  ||fsens||_inf  = {np.abs(fsens).max():.4g}")

        # 5. Chain-rule gradient
        # Normalise fsens the same way the C++ MMA does:  dfdx = fsens/(scalef*nallcells)
        # This keeps the gradient O(1) per design variable, independent of mesh size,
        # and consistent with the scale of meanT.
        scale = float(np.abs(fsens).max()) * len(fsens)
        if scale > 0:
            fsens_scaled = fsens / scale
        else:
            fsens_scaled = fsens

        grad_ctrl = chain_rule_gradient(
            fsens_scaled, self.cell_centers_mm, disp_mm, sdf,
            self.k, self.epsilon, self.W
        )                                           # (N_ctrl, 3)
        grad_flat = grad_ctrl.ravel()               # (N_ctrl * 3,)

        elapsed = time.time() - t0
        print(f"  elapsed = {elapsed:.1f} s  ||grad||_2 = {np.linalg.norm(grad_flat):.4g}")

        self._history.append(dict(
            iter=self._iter, J=meanT, dissPower=dissPower, vol=vol_use,
            grad_norm=float(np.linalg.norm(grad_flat)), elapsed=elapsed
        ))
        self._save_history()

        return float(meanT), grad_flat

    def _save_history(self) -> None:
        hist_path = self.case_dir / 'schwarz_opt_history.txt'
        with open(hist_path, 'w') as f:
            f.write("iter  J_meanT      DissPower    solid_frac  grad_norm  elapsed_s\n")
            for h in self._history:
                f.write(f"{h['iter']:4d}  {h['J']:12.6g}  "
                        f"{h.get('dissPower', float('nan')):12.6g}  "
                        f"{h['vol']:10.4f}  {h['grad_norm']:10.4g}  "
                        f"{h['elapsed']:8.1f}\n")

    def save_ctrl_pts(self, x: np.ndarray, tag: str = '') -> None:
        """Save current control-point positions + displacements to a file."""
        disp_ctrl = x.reshape(self.n_ctrl, 3)
        out_path  = self.case_dir / f'schwarz_ctrl_pts{tag}.txt'
        with open(out_path, 'w') as f:
            f.write("# x_mm  y_mm  z_mm  dx_mm  dy_mm  dz_mm\n")
            for (px, py, pz), (dx, dy, dz) in zip(self.ctrl_pts_mm, disp_ctrl):
                f.write(f"{px:.4f} {py:.4f} {pz:.4f} "
                        f"{dx:.6g} {dy:.6g} {dz:.6g}\n")
        print(f"  Control points saved → {out_path}")

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        n_iters:   int   = 50,
        x0:        np.ndarray | None = None,
        load_ctrl: Path | None = None,
    ) -> np.ndarray:
        """
        Optimise RBF control-point displacements.

        Parameters
        ----------
        n_iters   : maximum number of outer L-BFGS-B steps
        x0        : initial flat design vector (defaults to zero displacement)
        load_ctrl : path to a previous schwarz_ctrl_pts.txt to warm-start

        Returns
        -------
        x_opt : optimised flat design vector
        """
        # Initial design variables
        if x0 is not None:
            x_init = x0
        elif load_ctrl is not None:
            data   = np.loadtxt(load_ctrl)
            x_init = data[:, 3:6].ravel()   # columns 3-5 are dx, dy, dz
            print(f"  Warm-started from {load_ctrl}")
        else:
            x_init = np.zeros(self.n_ctrl * 3)

        print(f"\nStarting L-BFGS-B optimisation  ({n_iters} max outer iters)\n")
        self.save_ctrl_pts(x_init, tag='_init')

        result = minimize(
            self.objective_and_gradient,
            x_init,
            method='L-BFGS-B',
            jac=True,                  # objective_and_gradient returns (f, g)
            bounds=self.bounds,
            options=dict(maxiter=n_iters, ftol=1e-10, gtol=1e-6, iprint=1),
        )

        print(f"\nOptimisation finished: {result.message}")
        print(f"  Final J = {result.fun:.6g}   nit = {result.nit}")

        x_opt = result.x
        self.save_ctrl_pts(x_opt, tag='_optimised')

        # Write the final gamma field for inspection
        gamma_final, _, _ = self._gamma_from_x(x_opt)
        final_path = self.case_dir / '1' / 'gamma_schwarz_final'
        write_gamma_field(final_path, gamma_final, '1')
        print(f"  Final Schwarz gamma written → {final_path}")

        return x_opt


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Schwarz Primitive RBF optimiser for the 3D heat-sink MTO case.'
    )
    parser.add_argument('--case',     default='app',
                        help='Path to OpenFOAM case directory (default: app)')
    parser.add_argument('--iters',    type=int, default=50,
                        help='Max L-BFGS-B outer iterations (default: 50)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of MPI processes (1 = serial, default: 1)')
    parser.add_argument('--solver',   default='MTO_TF',
                        help='OpenFOAM solver executable (default: MTO_TF)')
    parser.add_argument('--spacing',  type=float, default=CONTROL_SPACING,
                        help=f'RBF control-point spacing in mm (default: {CONTROL_SPACING})')
    parser.add_argument('--unit',     type=float, default=F_UNIT_SIZE,
                        help=f'Schwarz-P TPMS cell size in mm (default: {F_UNIT_SIZE})')
    parser.add_argument('--wall',     type=float, default=F_WALL_THICKNESS,
                        help=f'Schwarz-P wall thickness in mm (default: {F_WALL_THICKNESS})')
    parser.add_argument('--epsilon',  type=float, default=SDF_EPSILON,
                        help=f'Smooth-Heaviside sharpness in mm (default: {SDF_EPSILON})')
    parser.add_argument('--warmstart', default=None,
                        help='Path to schwarz_ctrl_pts.txt for warm-start')
    parser.add_argument('--postprocess', default='postProcess',
                        help='OpenFOAM postProcess binary (default: postProcess)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    case_dir   = (script_dir / args.case).resolve()
    if not case_dir.is_dir():
        sys.exit(f"ERROR: case directory not found: {case_dir}")

    warm = Path(args.warmstart) if args.warmstart else None

    opt = SchwarzRBFOptimizer(
        case_dir        = case_dir,
        k               = 2.0 * math.pi / args.unit,
        wall_thickness  = args.wall,
        epsilon         = args.epsilon,
        control_spacing = args.spacing,
        solver          = args.solver,
        n_procs         = args.parallel,
        of_binary       = args.postprocess,
    )
    opt.run(n_iters=args.iters, load_ctrl=warm)


if __name__ == '__main__':
    main()
