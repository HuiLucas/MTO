"""
Gyroid RBF Optimizer for the 3D heat-sink MTO case.

Replaces the internal SIMP/MMA density parameterisation with a Gyroid TPMS
geometry parameterised by RBF control-point spatial-frequency perturbations.

Design variables:
    frequency perturbation (dk_x, dk_y, dk_z) at each RBF control point on a
    regular grid.  The spatially-varying frequency at a cell centre is:
        k_a(x,y,z) = k_base + RBF(dk_a_ctrl)(x,y,z)
    Total: n_ctrl * 3 scalars.

Gyroid TPMS implicit surface:
    G(x,y,z) = sin(kx*x)*cos(ky*y)
              + sin(ky*y)*cos(kz*z)
              + sin(kz*z)*cos(kx*x)  = 0
    SDF = |G| - half_thickness
    gamma = sigmoid(SDF / epsilon)   (1 → fluid, 0 → solid)

Each outer optimisation iteration:
    1. Given current control-point frequency perturbations dk, build an
       RBFFrequencyField (thin-plate-spline).
    2. Evaluate the Gyroid SDF at every mesh cell centre.
    3. Map SDF → gamma via a smooth Heaviside (sigmoid).
    4. Write the gamma field to the OpenFOAM case directory.
    5. Set controlDict startTime/endTime so the solver runs exactly ONE outer
       iteration, then exits.
    6. Run the OpenFOAM MTO_TF solver (serial or parallel).
    7. Read fsens = dJ/d(gamma) from the solver output.
    8. Chain-rule: dJ/d(dk_ctrl) via analytic Gyroid derivatives and the RBF
       evaluation Jacobian.
    9. Feed (objective, gradient) to scipy L-BFGS-B for the next step.

Usage
-----
    python gyroid_rbf_optimizer.py [--case app] [--iters 50] [--parallel 20]

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

from am_constraints import AMConstraints

# ── Physical / geometry parameters ─────────────────────────────────────────────
# blockMeshDict uses convertToMeters 0.01  →  mesh coords in metres
MESH_UNIT_TO_MM = 10.0        # 0.01 m * 1000 mm/m  (mesh unit → mm)

# Optimization domain bounding box (mm) – derived from blockMeshDict vertices
OPT_XMIN, OPT_XMAX = 0.0,  4.0
OPT_YMIN, OPT_YMAX = 0.0,  2.5
OPT_ZMIN, OPT_ZMAX = 0.0, 10.0

# Gyroid TPMS parameters
F_UNIT_SIZE      = 1.5    # TPMS cell size (mm) – sets base frequency k_base
F_WALL_THICKNESS = 0.30   # solid wall thickness (mm)
SDF_EPSILON      = 0.04   # smooth-Heaviside sharpness (mm)

# RBF control-point grid
CONTROL_SPACING  = 2.0    # spacing between control points (mm)
K_AMP_BOUND      = 2.0    # ±bound on each dk component (rad/mm);
                           # ensures k_base ± K_AMP_BOUND stays positive for
                           # default k_base ≈ 4.19 rad/mm

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

    m = re.search(r'internalField\s+uniform\s+([0-9Ee.+\-]+)', text)
    if m:
        n_match = re.search(r'nonuniform List<scalar>\s+(\d+)', text)
        if n_match:
            n = int(n_match.group(1))
        else:
            raise ValueError(f"Cannot determine cell count from {path}")
        return np.full(n, float(m.group(1)))

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
    """
    _VEC_RE = re.compile(
        r'\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)'
        r'\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)'
        r'\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s*\)'
    )

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
                m = re.search(r'nonuniform List<vector>\s+(\d+)', s)
                if m:
                    n     = int(m.group(1))
                    state = SEEK_OPEN
                else:
                    state = SEEK_COUNT

            elif state == SEEK_COUNT:
                if s.isdigit():
                    n     = int(s)
                    state = SEEK_OPEN

            elif state == SEEK_OPEN:
                if s == '(':
                    state = READ

            elif state == READ:
                if s.startswith(')'):
                    break
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
    Caches result to case_dir/cell_centers_mm.npy.
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

    cc_path = case_dir / '0' / 'C'
    if not cc_path.exists():
        cc_path = next(case_dir.glob('0/ccx'), None)
        if cc_path is None:
            raise FileNotFoundError(
                "postProcess writeCellCentres did not produce 0/C; "
                "check your OpenFOAM installation."
            )

    cc_m  = read_vector_field(cc_path)   # (N, 3) in metres
    cc_mm = cc_m * 1000.0                 # → mm
    np.save(cache, cc_mm)
    print(f"  Cell centres: {cc_mm.shape[0]:,} cells, range "
          f"x=[{cc_mm[:,0].min():.2f},{cc_mm[:,0].max():.2f}] "
          f"y=[{cc_mm[:,1].min():.2f},{cc_mm[:,1].max():.2f}] "
          f"z=[{cc_mm[:,2].min():.2f},{cc_mm[:,2].max():.2f}] mm")
    np.save(cache, cc_mm)
    return cc_mm


# ── RBF spatial-frequency field ────────────────────────────────────────────────

class RBFFrequencyField:
    """
    3-D spatial-frequency perturbation field:
        RBF thin-plate spline → dense baked grid → trilinear lookup.

    Stores and interpolates (dk_x, dk_y, dk_z) – the per-axis deviations from
    k_base.  The actual wavenumber at any point is k_base + dk(x,y,z).
    """

    def __init__(
        self,
        ctrl_pts:     np.ndarray,   # (N_ctrl, 3) control-point positions (mm)
        dk_ctrl:      np.ndarray,   # (N_ctrl, 3) frequency perturbations (rad/mm)
        bbox_min:     np.ndarray,   # (3,) field extent min (mm)
        bbox_max:     np.ndarray,   # (3,) field extent max (mm)
        bake_spacing: float = 0.5,
    ):
        self.ctrl_pts = ctrl_pts
        self.dk_ctrl  = dk_ctrl
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)

        self._rbf = [
            RBFInterpolator(ctrl_pts, dk_ctrl[:, ax],
                            kernel='thin_plate_spline', degree=1)
            for ax in range(3)
        ]

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

    def get_dk_batch(self, pts_mm: np.ndarray) -> np.ndarray:
        """Vectorised trilinear lookup; pts_mm is (N, 3) in mm, returns (N, 3) dk."""
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


def build_freq_field(ctrl_pts_mm: np.ndarray,
                     dk_ctrl:     np.ndarray,
                     bbox_min_mm: np.ndarray,
                     bbox_max_mm: np.ndarray,
                     bake_spacing: float = 0.5) -> RBFFrequencyField:
    """Construct an RBFFrequencyField from control-point frequency perturbations."""
    return RBFFrequencyField(ctrl_pts_mm, dk_ctrl,
                             bbox_min_mm, bbox_max_mm, bake_spacing)


# ── Gyroid SDF and gamma ───────────────────────────────────────────────────────

def gyroid_sdf_batch(pts_mm:         np.ndarray,
                     freq_mm:         np.ndarray,
                     half_thickness:  float) -> np.ndarray:
    """
    Vectorised Gyroid SDF at an array of points.

    G(x,y,z) = sin(kx*x)*cos(ky*y)
              + sin(ky*y)*cos(kz*z)
              + sin(kz*z)*cos(kx*x)
    SDF = |G| - half_thickness

    SDF < 0  →  inside solid wall  →  gamma = 0
    SDF > 0  →  fluid channel      →  gamma = 1

    Parameters
    ----------
    pts_mm        : (N, 3) cell-centre positions (mm)
    freq_mm       : (N, 3) spatially-varying frequencies [kx, ky, kz] (rad/mm)
    half_thickness: 0.5 * wall_thickness (mm)
    """
    x  = pts_mm[:, 0];  y  = pts_mm[:, 1];  z  = pts_mm[:, 2]
    kx = freq_mm[:, 0]; ky = freq_mm[:, 1]; kz = freq_mm[:, 2]
    G  = (np.sin(kx*x) * np.cos(ky*y)
        + np.sin(ky*y) * np.cos(kz*z)
        + np.sin(kz*z) * np.cos(kx*x))
    return np.abs(G) - half_thickness


def gamma_from_sdf(sdf: np.ndarray, epsilon: float) -> np.ndarray:
    """Smooth Heaviside: gamma = sigmoid(sdf / epsilon)."""
    return expit(sdf / epsilon)


def dgamma_dsdf_vals(sdf: np.ndarray, epsilon: float) -> np.ndarray:
    """d(gamma)/d(sdf) = sigmoid(sdf/eps) * (1 - sigmoid(sdf/eps)) / eps."""
    g = gamma_from_sdf(sdf, epsilon)
    return g * (1.0 - g) / epsilon


# ── RBF Jacobian (one-time precomputation) ─────────────────────────────────────

def build_rbf_jacobian(ctrl_pts_mm:     np.ndarray,
                       cell_centers_mm: np.ndarray,
                       bake_spacing:    float = 0.5) -> np.ndarray:
    """
    Precompute the (N_cells, N_ctrl) linear evaluation matrix W such that:

        dk_a(cell_j) = sum_l  W[j, l] * dk_a_ctrl[l]

    for each frequency axis a independently (W is axis-independent).

    Built by evaluating the RBF with a unit impulse at each control point.
    """
    n_ctrl  = len(ctrl_pts_mm)
    n_cells = len(cell_centers_mm)

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
    fsens:          np.ndarray,   # (N,)   dJ/d(gamma)  from OpenFOAM
    pts_mm:         np.ndarray,   # (N,3)  cell centres (mm)
    freq_mm:        np.ndarray,   # (N,3)  [kx, ky, kz] at each cell (rad/mm)
    sdf:            np.ndarray,   # (N,)   Gyroid SDF values
    epsilon:        float,        # smooth-Heaviside sharpness
    W:              np.ndarray,   # (N, N_ctrl) RBF Jacobian
) -> np.ndarray:
    """
    Compute dJ/d(dk_ctrl) via the full chain rule:

        dJ/d(dk_{a,l}) = sum_j  fsens_j
                               * d(gamma_j)/d(SDF_j)
                               * d(SDF_j)/d(G_j)
                               * d(G_j)/d(k_{a,j})
                               * W[j, l]

    where:
        d(SDF)/d(G) = sign(G)

        d(G)/d(kx) = x * ( cos(kx*x)*cos(ky*y) - sin(kz*z)*sin(kx*x) )
        d(G)/d(ky) = y * ( cos(ky*y)*cos(kz*z) - sin(kx*x)*sin(ky*y) )
        d(G)/d(kz) = z * ( cos(kz*z)*cos(kx*x) - sin(ky*y)*sin(kz*z) )

    Returns shape (N_ctrl, 3) – gradient w.r.t. (dk_x, dk_y, dk_z) of each ctrl pt.
    """
    x  = pts_mm[:, 0];  y  = pts_mm[:, 1];  z  = pts_mm[:, 2]
    kx = freq_mm[:, 0]; ky = freq_mm[:, 1]; kz = freq_mm[:, 2]

    G     = (np.sin(kx*x) * np.cos(ky*y)
           + np.sin(ky*y) * np.cos(kz*z)
           + np.sin(kz*z) * np.cos(kx*x))
    signG = np.sign(G)

    dG_dkx = x * (np.cos(kx*x)*np.cos(ky*y) - np.sin(kz*z)*np.sin(kx*x))
    dG_dky = y * (np.cos(ky*y)*np.cos(kz*z) - np.sin(kx*x)*np.sin(ky*y))
    dG_dkz = z * (np.cos(kz*z)*np.cos(kx*x) - np.sin(ky*y)*np.sin(kz*z))

    dgds = dgamma_dsdf_vals(sdf, epsilon)   # (N,)

    wx = fsens * dgds * signG * dG_dkx     # (N,)
    wy = fsens * dgds * signG * dG_dky
    wz = fsens * dgds * signG * dG_dkz

    grad_kx = W.T @ wx                     # (N_ctrl,)
    grad_ky = W.T @ wy
    grad_kz = W.T @ wz

    return np.stack([grad_kx, grad_ky, grad_kz], axis=1)  # (N_ctrl, 3)


# ── OpenFOAM runner ────────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], cwd: str,
             log_path: Path | None = None,
             capture: bool = False,
             mpi: bool = False,
             timeout: int = 7200,
             env: dict | None = None) -> subprocess.CompletedProcess:
    """
    Run a command, optionally writing stdout+stderr to log_path.

    mpi=True  → start_new_session=True (detaches from controlling terminal;
                 required for mpirun to function correctly in non-interactive
                 sessions; with setsid the new PID == PGID so we can still
                 killpg on cleanup).
    mpi=False → capture=True/False as requested, no special session handling.
    """
    kwargs: dict = dict(cwd=cwd)
    if env is not None:
        kwargs['env'] = env

    if mpi:
        # setsid: new session, no controlling terminal, new process group.
        # proc.pid == pgid after setsid, so killpg(proc.pid, 9) kills all ranks.
        kwargs['start_new_session'] = True
    else:
        # For short-lived helpers (decomposePar, reconstructPar) just inherit session.
        pass

    if capture:
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.PIPE
        lf = None
    elif log_path is not None:
        lf = open(log_path, 'wb')
        kwargs['stdout'] = lf
        kwargs['stderr'] = subprocess.STDOUT
    else:
        lf = None

    proc = subprocess.Popen(cmd, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, Exception):
        # Kill job: with start_new_session proc.pid IS the pgid.
        # For non-mpi procs fall back to plain terminate.
        try:
            if mpi:
                os.killpg(proc.pid, 9)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        if lf:
            lf.close()
        raise
    finally:
        if lf:
            lf.close()

    return subprocess.CompletedProcess(
        cmd, proc.returncode,
        stdout=stdout, stderr=stderr,
    )


def _kill_mpi_orphans(solver: str = 'MTO_TF') -> None:
    """Kill any leftover solver / MPI processes from a previous crashed run."""
    for name in (solver, 'orted', 'orterun'):
        subprocess.run(['pkill', '-SIGTERM', name],
                       capture_output=True)
    time.sleep(0.5)
    # Remove stale OpenMPI session files that block new connections
    import glob
    for p in glob.glob('/tmp/ompi.*') + glob.glob('/tmp/openmpi-sessions-*'):
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def _mpirun_is_healthy(solver: str = 'MTO_TF', probe_timeout: float = 8.0) -> bool:
    """
    Quick sanity-check: launch 'mpirun -n 1 hostname' and confirm it produces
    output within probe_timeout seconds.  Returns False if mpirun hangs.
    """
    try:
        devnull = open(os.devnull, 'rb')
        proc = subprocess.Popen(
            ['mpirun', '--oversubscribe', '-n', '1', 'hostname'],
            stdin=devnull, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        devnull.close()
        try:
            out, _ = proc.communicate(timeout=probe_timeout)
            return proc.returncode == 0 and len(out.strip()) > 0
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                pass
            proc.wait()
            return False
    except Exception:
        return False


def run_openfoam_one_step(case_dir: Path,
                          start_time: int,
                          end_time: int,
                          solver: str = 'MTO_TF',
                          n_procs: int = 1,
                          iter_num: int = 0,
                          solver_timeout: int = 7200) -> None:
    """
    Run the MTO_TF solver for exactly (end_time - start_time) outer iterations.
    If n_procs > 1 but mpirun is unresponsive, falls back to serial automatically.
    """
    system_dir = case_dir / 'system'
    update_control_dict(system_dir, start_time, end_time, write_interval=1)
    cwd = str(case_dir)

    # Auto-detect mpirun health on first parallel attempt
    use_parallel = n_procs > 1
    if use_parallel and iter_num == 1:
        print("  Probing mpirun … ", end='', flush=True)
        if _mpirun_is_healthy(solver):
            print("OK")
        else:
            print("HUNG – falling back to serial for all iterations")
            use_parallel = False
            # Persist the fallback decision so later iterations skip the probe
            _mpirun_fallback_flag = case_dir / '.mpirun_broken'
            _mpirun_fallback_flag.touch()

    # Check persistent fallback flag (set by iter 1 or a previous session)
    if use_parallel and (case_dir / '.mpirun_broken').exists():
        use_parallel = False

    if use_parallel:
        # Clean up any orphan processes / stale MPI state before starting
        _kill_mpi_orphans(solver)

        print(f"  decomposePar (time {start_time}) …")
        r = _run_cmd(
            ['decomposePar', '-time', str(start_time), '-force', '-case', str(case_dir)],
            cwd=cwd, capture=True,
        )
        if r.returncode != 0:
            log = case_dir / f'log.decomposePar.iter{iter_num:03d}'
            log.write_bytes((r.stdout or b'') + b'\n' + (r.stderr or b''))
            raise RuntimeError(
                f"decomposePar failed (exit {r.returncode}). Log: {log}\n"
                + (r.stderr or b'').decode(errors='replace')[-2000:]
            )

        log_path = case_dir / f'log.{solver}.iter{iter_num:03d}'
        print(f"  mpirun -n {n_procs} {solver} … (log → {log_path.name})")
        r = _run_cmd(
            ['mpirun', '--oversubscribe', '-n', str(n_procs),
             solver, '-parallel', '-case', str(case_dir)],
            cwd=cwd, log_path=log_path, mpi=True, timeout=solver_timeout,
        )
        if r.returncode != 0:
            tail = log_path.read_bytes()[-4000:].decode(errors='replace')
            raise RuntimeError(
                f"{solver} (parallel) failed (exit {r.returncode}).\n"
                f"Last output from {log_path.name}:\n{tail}"
            )

        print(f"  reconstructPar (time {end_time}) …")
        r = _run_cmd(
            ['reconstructPar', '-time', str(end_time), '-case', str(case_dir)],
            cwd=cwd, capture=True,
        )
        if r.returncode != 0:
            log = case_dir / f'log.reconstructPar.iter{iter_num:03d}'
            log.write_bytes((r.stdout or b'') + b'\n' + (r.stderr or b''))
            raise RuntimeError(
                f"reconstructPar failed (exit {r.returncode}). Log: {log}\n"
                + (r.stderr or b'').decode(errors='replace')[-2000:]
            )
        
        # Immediately backup the unified time directory before any potential cleanup
        backup_latest_fluid_state(case_dir)
    else:
        # Serial mode: MTO_TF is MPI-compiled so it needs MPI symbols even with nProcs=1.
        # If the real mpirun is broken, inject a single-process MPI shim via LD_PRELOAD.
        shim = Path(__file__).parent / 'fake_mpi.so'
        env  = None
        if shim.exists():
            import os as _os
            env = _os.environ.copy()
            existing = env.get('LD_PRELOAD', '')
            env['LD_PRELOAD'] = (str(shim) + ':' + existing).strip(':')

        print(f"  Running {solver} (serial+MPI-shim) for t={start_time}→{end_time} …")
        r = _run_cmd([solver, '-case', str(case_dir)], cwd=cwd,
                     timeout=solver_timeout, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"{solver} exited with code {r.returncode}")
        
        # Immediately backup the time directory before any potential cleanup
        backup_latest_fluid_state(case_dir)


def read_objective(case_dir: Path) -> tuple[float, float]:
    """Read the latest meanT and DissPower from text files."""
    def _last(fname):
        p = case_dir / fname
        if not p.exists():
            return float('nan')
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        return float(lines[-1]) if lines else float('nan')
    return _last('meanT.txt'), _last('Disspower.txt')


def backup_latest_fluid_state(case_dir: Path, backup_dir: Path | None = None) -> Path:
    """
    Copy the latest time-step output directory to a persistent backup location.
    Always overwrites the backup, ensuring there is always a saved fluid state.
    
    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory
    backup_dir : Path | None
        Backup directory name (default: case_dir / 'latest_fluid_state')
        
    Returns
    -------
    Path
        Path to the backed-up time directory
    """
    if backup_dir is None:
        backup_dir = case_dir / 'latest_fluid_state'

    latest_time = get_latest_time(case_dir)
    if latest_time == 0:
        print(f"  WARNING: No numeric time directories found to backup in {case_dir}")
        return backup_dir

    latest_time_dir = case_dir / str(latest_time)

    # Remove old backup if it exists
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Preferred: copy the unified root time directory if present
    copied_any = False
    if latest_time_dir.exists():
        try:
            shutil.copytree(latest_time_dir, backup_dir, dirs_exist_ok=True)
            print(f"  Copied root time directory: {latest_time_dir.name} → {backup_dir.name}/")
            copied_any = True
        except Exception:
            # Fall through to copying processor subdirs if copytree fails
            pass

    # If root time dir did not contain expected fields, also copy per-processor time dirs
    proc_dirs = sorted([d for d in case_dir.iterdir() if d.is_dir() and d.name.startswith('processor')],
                       key=lambda p: int(p.name.replace('processor', '')))
    for proc in proc_dirs:
        proc_time = proc / str(latest_time)
        if proc_time.exists():
            dest = backup_dir / proc.name
            try:
                shutil.copytree(proc_time, dest, dirs_exist_ok=True)
                print(f"  Copied processor time: {proc.name}/{latest_time} → {dest}/")
                copied_any = True
            except Exception:
                # Try per-field copy as fallback
                dest.mkdir(parents=True, exist_ok=True)
                for fname in ('p', 'U', 'T'):
                    srcf = proc_time / fname
                    if srcf.exists():
                        try:
                            shutil.copy2(srcf, dest / fname)
                            copied_any = True
                        except Exception:
                            pass

    # As a last-resort, copy specific fields from the root time dir if present
    if not copied_any and latest_time_dir.exists():
        for fname in ('p', 'U', 'T', 'gamma', 'fsens'):
            srcf = latest_time_dir / fname
            if srcf.exists():
                try:
                    shutil.copy2(srcf, backup_dir / fname)
                    copied_any = True
                except Exception:
                    pass

    if copied_any:
        print(f"  Fluid state backed up for time {latest_time} → {backup_dir}")
    else:
        print(f"  WARNING: No fluid state files found for time {latest_time} to backup.")

    return backup_dir


# ── Main optimizer class ───────────────────────────────────────────────────────

class GyroidRBFOptimizer:
    """
    Outer optimisation loop: RBF control-point frequency perturbations → OpenFOAM → gradient.

    Design variables x = [dk_x_ctrl ; dk_y_ctrl ; dk_z_ctrl] (flat, length N_ctrl*3).
    k_a(cell) = k_base + W @ dk_a_ctrl,  where W is the RBF evaluation matrix.
    """

    def __init__(
        self,
        case_dir:        Path,
        k_base:          float = 2.0 * math.pi / F_UNIT_SIZE,
        wall_thickness:  float = F_WALL_THICKNESS,
        epsilon:         float = SDF_EPSILON,
        control_spacing: float = CONTROL_SPACING,
        k_amp_bound:     float = K_AMP_BOUND,
        bake_spacing:    float = 0.4,
        solver:          str   = 'MTO_TF',
        n_procs:         int   = 1,
        of_binary:       str   = 'postProcess',
        # ── AM constraint options ──────────────────────────────────────────────
        am_r_filter:     float = 0.15,   # Helmholtz filter radius (mm)
        am_theta_max:    float = math.pi / 4.0,
        am_P_bar:        float = 0.01,
        am_Phi_o:        float = 0.01,
        am_mu_overhang:  float = 1.0,
        am_mu_thickness: float = 10.0,
        use_overhang:    bool  = True,
        use_thickness:   bool  = True,
    ):
        self.case_dir       = case_dir
        self.k_base         = k_base
        self.half_thickness = 0.5 * wall_thickness
        self.epsilon        = epsilon
        self.k_amp_bound    = k_amp_bound
        self.solver         = solver
        self.n_procs        = n_procs
        self._iter          = 0
        self._history: list[dict] = []

        if k_base - k_amp_bound <= 0:
            raise ValueError(
                f"k_amp_bound ({k_amp_bound:.3f}) ≥ k_base ({k_base:.3f}); "
                "negative frequencies would occur. Reduce k_amp_bound."
            )

        opt_min = np.array([OPT_XMIN, OPT_YMIN, OPT_ZMIN])
        opt_max = np.array([OPT_XMAX, OPT_YMAX, OPT_ZMAX])

        field_min = opt_min - 0.5
        field_max = opt_max + 0.5

        axes = [np.arange(lo, hi + control_spacing, control_spacing)
                for lo, hi in zip(field_min, field_max)]
        GX, GY, GZ = np.meshgrid(*axes, indexing='ij')
        self.ctrl_pts_mm = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
        self.n_ctrl      = len(self.ctrl_pts_mm)
        self.field_min   = field_min
        self.field_max   = field_max
        self.bake_spacing = bake_spacing

        print(f"Control points : {self.n_ctrl}  "
              f"(grid {GX.shape[0]}×{GX.shape[1]}×{GX.shape[2]})")
        print(f"k_base = {k_base:.4f} rad/mm  (unit size = {2*math.pi/k_base:.3f} mm)")
        print(f"k_amp_bound = ±{k_amp_bound:.4f} rad/mm  "
              f"→ k in [{k_base-k_amp_bound:.3f}, {k_base+k_amp_bound:.3f}] rad/mm")

        self.cell_centers_mm = get_cell_centers_mm(case_dir, of_binary)
        self.n_cells = len(self.cell_centers_mm)

        self.W = build_rbf_jacobian(self.ctrl_pts_mm, self.cell_centers_mm, bake_spacing)

        self.bounds = [(-k_amp_bound, k_amp_bound)] * (self.n_ctrl * 3)

        # ── AM constraints ────────────────────────────────────────────────────
        if use_overhang or use_thickness:
            print("\nBuilding AM constraint operators …")
            self.am = AMConstraints(
                pts_mm          = self.cell_centers_mm,
                r_filter_mm     = am_r_filter,
                theta_max       = am_theta_max,
                P_bar           = am_P_bar,
                Phi_o           = am_Phi_o,
                mu_overhang     = am_mu_overhang,
                mu_thickness    = am_mu_thickness,
                use_overhang    = use_overhang,
                use_thickness   = use_thickness,
            )
        else:
            self.am = None

        print(f"\nOptimiser ready. {self.n_ctrl * 3} design variables "
              f"(±{k_amp_bound:.4f} rad/mm each).\n")

    # ── internals ─────────────────────────────────────────────────────────────

    def _dk_to_field(self, x: np.ndarray) -> tuple[RBFFrequencyField, np.ndarray]:
        """Reshape flat x vector → (N_ctrl, 3), build field, return (field, dk_ctrl)."""
        dk_ctrl = x.reshape(self.n_ctrl, 3)
        field   = build_freq_field(
            self.ctrl_pts_mm, dk_ctrl,
            self.field_min, self.field_max, self.bake_spacing
        )
        return field, dk_ctrl

    def _gamma_from_x(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (gamma, sdf, freq_mm) from flat design vector."""
        field, _ = self._dk_to_field(x)
        dk_mm    = field.get_dk_batch(self.cell_centers_mm)       # (N, 3)
        freq_mm  = self.k_base + dk_mm                            # (N, 3) actual k_x,k_y,k_z
        sdf      = gyroid_sdf_batch(self.cell_centers_mm, freq_mm, self.half_thickness)
        gamma    = gamma_from_sdf(sdf, self.epsilon)
        return gamma, sdf, freq_mm

    # ── objective + gradient (called by scipy) ────────────────────────────────

    def objective_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        1. Evaluate Gyroid gamma from x.
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

        start_t = 0
        end_t   = 1

        gamma, sdf, freq_mm = self._gamma_from_x(x)
        print(f"  gamma:  min={gamma.min():.3f}  max={gamma.max():.3f}  "
              f"mean={gamma.mean():.3f}  solid_frac={1-gamma.mean():.3f}")
        dk_mm = freq_mm - self.k_base
        print(f"  dk:     min={dk_mm.min():.4f}  max={dk_mm.max():.4f} rad/mm")

        gamma_path = self.case_dir / '0' / 'gamma'
        write_gamma_field(gamma_path, gamma, '0')
        print(f"  gamma written → {gamma_path}")

        try:
            run_openfoam_one_step(
                self.case_dir, start_t, end_t,
                self.solver, self.n_procs, iter_num=self._iter
            )
        finally:
            # Always backup the latest fluid state, even if solver fails or is interrupted
            backup_latest_fluid_state(self.case_dir)

        fsens_path = self.case_dir / '1' / 'fsens'
        if not fsens_path.exists():
            raise FileNotFoundError(
                f"fsens not found at {fsens_path}.\n"
                "Make sure MTO_TF.C has been modified to call fsens.write() "
                "and the solver has been recompiled."
            )
        fsens = read_scalar_field(fsens_path)

        meanT, dissPower = read_objective(self.case_dir)

        vol_use = 1.0 - gamma.mean()
        print(f"  J (meanT)      = {meanT:.6g}")
        print(f"  DissPower      = {dissPower:.6g}  (constraint, not minimised)")
        print(f"  solid_fraction = {vol_use:.4f}")
        print(f"  ||fsens||_inf  = {np.abs(fsens).max():.4g}")

        if not hasattr(self, '_fsens_ref'):
            self._fsens_ref = float(np.abs(fsens).max())
            if self._fsens_ref == 0:
                self._fsens_ref = 1.0
        fsens_norm = fsens / self._fsens_ref

        # ── AM constraint penalties ───────────────────────────────────────────
        am_info = {}
        J_aug   = float(meanT)
        if self.am is not None:
            J_aug, fsens_aug, am_info = self.am.apply(
                gamma, J_aug, fsens_norm, iteration=self._iter,
            )
            self.am.update_penalties(am_info)
            print(f"  g_overhang = {am_info.get('g_oh', 0.0):.4g}  "
                  f"(limit {self.am.P_bar:.3g}, pen={am_info.get('pen_oh',0):.3g})")
            print(f"  g_thickness= {am_info.get('g_th', 0.0):.4g}  "
                  f"(limit {self.am.Phi_o:.3g}, pen={am_info.get('pen_th',0):.3g})  "
                  f"β_proj={am_info.get('beta_proj',1):.1f}")
        else:
            fsens_aug = fsens_norm

        grad_ctrl = chain_rule_gradient(
            fsens_aug, self.cell_centers_mm, freq_mm, sdf, self.epsilon, self.W
        )                                            # (N_ctrl, 3)
        grad_flat = grad_ctrl.ravel()                # (N_ctrl * 3,)

        elapsed = time.time() - t0
        print(f"  elapsed = {elapsed:.1f} s  ||grad||_2 = {np.linalg.norm(grad_flat):.4g}  "
              f"fsens_ref = {self._fsens_ref:.4g}")

        self._history.append(dict(
            iter=self._iter, J=meanT, J_aug=J_aug,
            dissPower=dissPower, vol=vol_use,
            g_oh=am_info.get('g_oh', 0.0),
            g_th=am_info.get('g_th', 0.0),
            grad_norm=float(np.linalg.norm(grad_flat)), elapsed=elapsed,
        ))
        self._save_history()
        self.save_ctrl_pts(x, tag='_checkpoint')   # always overwritten; safe restart point

        return J_aug, grad_flat

    def _save_history(self) -> None:
        hist_path = self.case_dir / 'gyroid_opt_history.txt'
        with open(hist_path, 'w') as f:
            f.write("iter  J_meanT      J_aug        DissPower    solid_frac  "
                    "g_oh      g_th      grad_norm  elapsed_s\n")
            for h in self._history:
                f.write(f"{h['iter']:4d}  {h['J']:12.6g}  "
                        f"{h.get('J_aug', h['J']):12.6g}  "
                        f"{h.get('dissPower', float('nan')):12.6g}  "
                        f"{h['vol']:10.4f}  "
                        f"{h.get('g_oh', 0.0):9.4g}  "
                        f"{h.get('g_th', 0.0):9.4g}  "
                        f"{h['grad_norm']:10.4g}  "
                        f"{h['elapsed']:8.1f}\n")

    def save_ctrl_pts(self, x: np.ndarray, tag: str = '') -> None:
        """Save current control-point positions + frequency perturbations to a file."""
        dk_ctrl  = x.reshape(self.n_ctrl, 3)
        out_path = self.case_dir / f'gyroid_ctrl_pts{tag}.txt'
        with open(out_path, 'w') as f:
            f.write("# x_mm  y_mm  z_mm  dk_x_radmm  dk_y_radmm  dk_z_radmm\n")
            for (px, py, pz), (dkx, dky, dkz) in zip(self.ctrl_pts_mm, dk_ctrl):
                f.write(f"{px:.4f} {py:.4f} {pz:.4f} "
                        f"{dkx:.6g} {dky:.6g} {dkz:.6g}\n")
        print(f"  Control points saved → {out_path}")

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        n_iters:   int            = 50,
        x0:        np.ndarray | None = None,
        load_ctrl: Path | None    = None,
    ) -> np.ndarray:
        """
        Optimise RBF control-point frequency perturbations.

        Parameters
        ----------
        n_iters   : maximum number of outer L-BFGS-B steps
        x0        : initial flat design vector (defaults to zero, i.e. uniform gyroid)
        load_ctrl : path to a previous gyroid_ctrl_pts.txt to warm-start

        Returns
        -------
        x_opt : optimised flat design vector (dk_x, dk_y, dk_z at each ctrl pt)
        """
        if x0 is not None:
            x_init = x0
        elif load_ctrl is not None:
            data   = np.loadtxt(load_ctrl)
            x_init = data[:, 3:6].ravel()   # columns 3-5 are dk_x, dk_y, dk_z
            print(f"  Warm-started from {load_ctrl}")
        else:
            x_init = np.zeros(self.n_ctrl * 3)

        print(f"\nStarting L-BFGS-B optimisation  ({n_iters} max outer iters)\n")
        self.save_ctrl_pts(x_init, tag='_init')

        result = minimize(
            self.objective_and_gradient,
            x_init,
            method='L-BFGS-B',
            jac=True,
            bounds=self.bounds,
            options=dict(maxiter=n_iters, ftol=1e-30, gtol=1e-4, iprint=1),
        )

        print(f"\nOptimisation finished: {result.message}")
        print(f"  Final J = {result.fun:.6g}   nit = {result.nit}")

        x_opt = result.x
        self.save_ctrl_pts(x_opt, tag='_optimised')

        gamma_final, _, _ = self._gamma_from_x(x_opt)
        final_path = self.case_dir / '1' / 'gamma_gyroid_final'
        write_gamma_field(final_path, gamma_final, '1')
        print(f"  Final Gyroid gamma written → {final_path}")

        return x_opt


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Gyroid RBF optimiser for the 3D heat-sink MTO case.'
    )
    parser.add_argument('--case',      default='app',
                        help='Path to OpenFOAM case directory (default: app)')
    parser.add_argument('--iters',     type=int, default=50,
                        help='Max L-BFGS-B outer iterations (default: 50)')
    parser.add_argument('--parallel',  type=int, default=1,
                        help='Number of MPI processes (1 = serial, default: 1)')
    parser.add_argument('--solver',    default='MTO_TF',
                        help='OpenFOAM solver executable (default: MTO_TF)')
    parser.add_argument('--spacing',   type=float, default=CONTROL_SPACING,
                        help=f'RBF control-point spacing in mm (default: {CONTROL_SPACING})')
    parser.add_argument('--unit',      type=float, default=F_UNIT_SIZE,
                        help=f'Gyroid cell size in mm – sets k_base (default: {F_UNIT_SIZE})')
    parser.add_argument('--wall',      type=float, default=F_WALL_THICKNESS,
                        help=f'Gyroid wall thickness in mm (default: {F_WALL_THICKNESS})')
    parser.add_argument('--epsilon',   type=float, default=SDF_EPSILON,
                        help=f'Smooth-Heaviside sharpness in mm (default: {SDF_EPSILON})')
    parser.add_argument('--kbound',    type=float, default=K_AMP_BOUND,
                        help=f'±bound on dk control variables in rad/mm (default: {K_AMP_BOUND})')
    parser.add_argument('--warmstart', default=None,
                        help='Path to gyroid_ctrl_pts.txt for warm-start')
    parser.add_argument('--postprocess', default='postProcess',
                        help='OpenFOAM postProcess binary (default: postProcess)')
    # ── AM constraint arguments ────────────────────────────────────────────────
    parser.add_argument('--am-filter',    type=float, default=0.15,
                        help='Helmholtz filter radius in mm (default: 0.15)')
    parser.add_argument('--am-theta',     type=float, default=45.0,
                        help='Max overhang angle in degrees (default: 45)')
    parser.add_argument('--am-P-bar',     type=float, default=0.01,
                        help='Overhang constraint bound (default: 0.01)')
    parser.add_argument('--am-Phi-o',     type=float, default=0.01,
                        help='Thickness constraint bound (default: 0.01)')
    parser.add_argument('--mu-overhang',  type=float, default=1.0,
                        help='Initial penalty weight for overhang (default: 1.0)')
    parser.add_argument('--mu-thickness', type=float, default=10.0,
                        help='Initial penalty weight for thickness (default: 10.0)')
    parser.add_argument('--no-overhang',  action='store_true',
                        help='Disable the overhang angle constraint')
    parser.add_argument('--no-thickness', action='store_true',
                        help='Disable the wall-thickness constraint')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    case_dir   = (script_dir / args.case).resolve()
    if not case_dir.is_dir():
        sys.exit(f"ERROR: case directory not found: {case_dir}")

    warm = Path(args.warmstart) if args.warmstart else None

    opt = GyroidRBFOptimizer(
        case_dir        = case_dir,
        k_base          = 2.0 * math.pi / args.unit,
        wall_thickness  = args.wall,
        epsilon         = args.epsilon,
        control_spacing = args.spacing,
        k_amp_bound     = args.kbound,
        solver          = args.solver,
        n_procs         = args.parallel,
        of_binary       = args.postprocess,
        am_r_filter     = args.am_filter,
        am_theta_max    = math.radians(args.am_theta),
        am_P_bar        = args.am_P_bar,
        am_Phi_o        = args.am_Phi_o,
        am_mu_overhang  = args.mu_overhang,
        am_mu_thickness = args.mu_thickness,
        use_overhang    = not args.no_overhang,
        use_thickness   = not args.no_thickness,
    )
    opt.run(n_iters=args.iters, load_ctrl=warm)


if __name__ == '__main__':
    main()
