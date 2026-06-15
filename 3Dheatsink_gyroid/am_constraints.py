"""
Analytic gyroid overhang constraint.

The outward solid normal is computed analytically from the gyroid implicit surface.

Two gyroid parameterisations are supported:

  Standard (rot_matrix is None):
    G = sin(kx·x)cos(ky·y) + sin(ky·y)cos(kz·z) + sin(kz·z)cos(kx·x)

  Rotated / flow-aligned (rot_matrix = R, a 3×3 orthogonal matrix):
    p = kx·x,  q = ky·y,  r = kz·z
    [u, v, w] = R @ [p, q, r]
    G = cos(u)cos(v) + sin(v)cos(w) − sin(w)sin(u)

    R is built by gyroid_rotation_matrix(a) which maps the x-axis to the
    unit flow-direction vector a.  The pi/2 phase shift (cos instead of sin
    in the first term) centres a gyroid hole along the flow axis.

  outward solid normal  =  sign(G) × ∇G / |∇G|

An overhang violation occurs wherever the normal points too far downward:
    viol = max(0,  −cos(θ_max) − (n̂ · b̂))

Gradients w.r.t. RBF frequency control-point perturbations dk_ctrl are
computed fully analytically via the chain rule through the rotation.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ── Rotation matrix ───────────────────────────────────────────────────────────

def gyroid_rotation_matrix(a: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the x-axis to unit vector a.

    Uses the closed-form Rodrigues expression:
        R[0,:] = [a0,  a1,  a2]
        R[1,:] = [-a1, (a0·a1² + a2²)/D,  (a0−1)·a1·a2/D]
        R[2,:] = [-a2, (a0−1)·a1·a2/D,    (a0·a2² + a1²)/D]
    where D = a1² + a2².

    Degenerate cases (a parallel to ±x) are handled explicitly.
    """
    a = np.asarray(a, dtype=float)
    a = a / np.linalg.norm(a)
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    D = a1 ** 2 + a2 ** 2
    if D < 1e-10:
        if a0 >= 0.0:
            return np.eye(3)
        # a ≈ (−1, 0, 0): rotate 180° around z
        return np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    return np.array([
        [a0,  a1,                    a2],
        [-a1, (a0 * a1**2 + a2**2) / D, (a0 - 1.0) * a1 * a2 / D],
        [-a2, (a0 - 1.0) * a1 * a2 / D, (a0 * a2**2 + a1**2) / D],
    ])


# ── Internal geometry helper ──────────────────────────────────────────────────

def _gyroid_geometry(
    pts_mm:   np.ndarray,   # (N, 3)
    freq_mm:  np.ndarray,   # (N, 3)  [kx, ky, kz]
    gamma:    np.ndarray,   # (N,)
    epsilon:  float,
    rot_matrix: np.ndarray | None,
) -> dict:
    """
    Compute G, spatial gradient (Gx, Gy, Gz), dG/dkα, and mixed partials
    dGx/dkα … dGz/dkα for the appropriate gyroid formula.

    Returns a dict with keys:
        G, sG, Gx, Gy, Gz, ng, inv_ng,
        dG_dkx, dG_dky, dG_dkz,
        dGx_dkx, dGx_dky, dGx_dkz,
        dGy_dkx, dGy_dky, dGy_dkz,
        dGz_dkx, dGz_dky, dGz_dkz,
        w, dw_dkx, dw_dky, dw_dkz
    """
    eps_reg = 1e-6
    x = pts_mm[:, 0]; y = pts_mm[:, 1]; z = pts_mm[:, 2]
    kx = freq_mm[:, 0]; ky = freq_mm[:, 1]; kz = freq_mm[:, 2]

    if rot_matrix is not None:
        R = rot_matrix
        p = kx * x; q = ky * y; r = kz * z
        u = R[0, 0]*p + R[0, 1]*q + R[0, 2]*r
        v = R[1, 0]*p + R[1, 1]*q + R[1, 2]*r
        w_coord = R[2, 0]*p + R[2, 1]*q + R[2, 2]*r

        su = np.sin(u); cu = np.cos(u)
        sv = np.sin(v); cv = np.cos(v)
        sw = np.sin(w_coord); cw = np.cos(w_coord)

        G  = cu*cv + sv*cw - sw*su

        # First derivatives of G_shifted w.r.t. rotated coords u,v,w
        dG_du = -su*cv - sw*cu
        dG_dv = -cu*sv + cv*cw
        dG_dw = -sv*sw - cw*su

        # Effective spatial gradient factors: Aα = R[:,α]·[dG_du, dG_dv, dG_dw]
        Ax = R[0, 0]*dG_du + R[1, 0]*dG_dv + R[2, 0]*dG_dw
        Ay = R[0, 1]*dG_du + R[1, 1]*dG_dv + R[2, 1]*dG_dw
        Az = R[0, 2]*dG_du + R[1, 2]*dG_dv + R[2, 2]*dG_dw

        Gx = kx * Ax
        Gy = ky * Ay
        Gz = kz * Az

        # dG/dkα = coord · Aα
        dG_dkx = x * Ax
        dG_dky = y * Ay
        dG_dkz = z * Az

        # Hessian of G_shifted w.r.t. (u, v, w) – symmetric
        H00 = -cu*cv + sw*su
        H01 = su*sv
        H02 = -cw*cu
        H11 = -cu*cv - sv*cw
        H12 = -cv*sw
        H22 = -sv*cw + sw*su

        # Quadratic form: M_{i,j} = R[:,i]^T H_s R[:,j]
        # Helper: (H_s @ r_b), components
        def _hr(rb):
            hr0 = H00*rb[0] + H01*rb[1] + H02*rb[2]
            hr1 = H01*rb[0] + H11*rb[1] + H12*rb[2]
            hr2 = H02*rb[0] + H12*rb[1] + H22*rb[2]
            return hr0, hr1, hr2

        r0 = (R[0, 0], R[1, 0], R[2, 0])
        r1 = (R[0, 1], R[1, 1], R[2, 1])
        r2 = (R[0, 2], R[1, 2], R[2, 2])

        def _M(ra, rb):
            hr0, hr1, hr2 = _hr(rb)
            return ra[0]*hr0 + ra[1]*hr1 + ra[2]*hr2

        M00 = _M(r0, r0); M01 = _M(r0, r1); M02 = _M(r0, r2)
        M10 = _M(r1, r0); M11 = _M(r1, r1); M12 = _M(r1, r2)
        M20 = _M(r2, r0); M21 = _M(r2, r1); M22 = _M(r2, r2)

        # Mixed partials ∂(Gα)/∂kβ
        dGx_dkx = Ax + kx * x * M00
        dGx_dky = kx * y * M01
        dGx_dkz = kx * z * M02

        dGy_dkx = ky * x * M10
        dGy_dky = Ay + ky * y * M11
        dGy_dkz = ky * z * M12

        dGz_dkx = kz * x * M20
        dGz_dky = kz * y * M21
        dGz_dkz = Az + kz * z * M22

    else:
        sx = np.sin(kx*x); cx = np.cos(kx*x)
        sy = np.sin(ky*y); cy = np.cos(ky*y)
        sz = np.sin(kz*z); cz = np.cos(kz*z)

        G  = sx*cy + sy*cz + sz*cx
        Ax = cx*cy - sz*sx
        Ay = cy*cz - sx*sy
        Az = cz*cx - sy*sz
        Gx = kx*Ax; Gy = ky*Ay; Gz = kz*Az

        dG_dkx = x * Ax
        dG_dky = y * Ay
        dG_dkz = z * Az

        dGx_dkx = Ax - kx * x * (sx*cy + sz*cx)
        dGx_dky = -kx * y * cx * sy
        dGx_dkz = -kx * z * cz * sx

        dGy_dkx = -ky * x * cx * sy
        dGy_dky = Ay - ky * y * (sy*cz + sx*cy)
        dGy_dkz = -ky * z * cy * sz

        dGz_dkx = -kz * x * cz * sx
        dGz_dky = -kz * y * cy * sz
        dGz_dkz = Az - kz * z * (sz*cx + sy*cz)

    sG = np.sign(G)
    sG[sG == 0] = 1.0

    ng2 = Gx**2 + Gy**2 + Gz**2 + eps_reg**2
    ng  = np.sqrt(ng2)
    inv_ng = 1.0 / ng

    w = gamma * (1.0 - gamma) / epsilon

    # dw/dkα = (1−2γ)·w/ε · sign(G) · dG/dkα
    dw_factor = (1.0 - 2.0 * gamma) * w / epsilon * sG
    dw_dkx = dw_factor * dG_dkx
    dw_dky = dw_factor * dG_dky
    dw_dkz = dw_factor * dG_dkz

    return dict(
        G=G, sG=sG,
        Gx=Gx, Gy=Gy, Gz=Gz, ng=ng, inv_ng=inv_ng,
        dG_dkx=dG_dkx, dG_dky=dG_dky, dG_dkz=dG_dkz,
        dGx_dkx=dGx_dkx, dGx_dky=dGx_dky, dGx_dkz=dGx_dkz,
        dGy_dkx=dGy_dkx, dGy_dky=dGy_dky, dGy_dkz=dGy_dkz,
        dGz_dkx=dGz_dkx, dGz_dky=dGz_dky, dGz_dkz=dGz_dkz,
        w=w, dw_dkx=dw_dkx, dw_dky=dw_dky, dw_dkz=dw_dkz,
    )


# ── Smooth morphological erosion (bridging length) ───────────────────────────
#
# LPBF printers can self-support ("bridge") unsupported spans up to a length
# L_bridge.  Small islands of overhang-violating *surface* (smaller than the
# bridge length) are therefore manufacturable for free and should not be
# penalised; only large connected violating surface regions matter.
#
# `viol = max(0, -cos_max - n_b)` is a property of the gyroid level-set's
# gradient *direction*, defined (and roughly periodic) everywhere in 3D space
# — not just on the printed G≈0 surface.  Averaging it unweighted over a voxel
# mixes in the surrounding bulk volume, whose gradient-direction statistics
# are an almost scale-invariant geometric constant unrelated to the printed
# surface's bridgeability.  To erode the *surface's* overhang pattern, the
# per-voxel statistics are restricted to the interface using the band weight
# w = gamma*(1-gamma)/epsilon (large only near G≈0, ~0 in the bulk):
#
#   1. Bin cell centres into a uniform auxiliary voxel grid of pitch
#      r = L_bridge/2 (independent of the unstructured / graded FOAM mesh).
#   2. Per voxel v: vbar_v = sum(w_i*viol_i) / sum(w_i)  — the local,
#      interface-weighted average violation; omega_v = sum(w_i) is the
#      voxel's interface "mass".
#   3. Erode vbar with an omega-weighted soft-min (LogSumExp, sharpness
#      1/bridge_eps) over the 3x3x3-voxel neighbourhood (box-filter,
#      separable). Voxels with omega≈0 (no surface nearby) contribute ~0 to
#      both the numerator and denominator of the soft-min ratio — they are
#      excluded, not treated as "clean" — so a thin printed shell is not
#      spuriously eroded just because it is thin through-thickness.
#   4. Gather the eroded value back to each cell  ->  m.
#
# A surface island survives erosion only if its core (more than ~r from its
# edge) is itself fully violating; sub-bridge islands erode to ~0.  `m` then
# replaces `viol` everywhere (pen = w*m**2, g_oh = sum(w*m)/sum(w)).
#
# Both vbar_v and omega_v depend on dk_ctrl (via viol_i(k) and w_i(k)), so
# dm_i/dk has two channels: through dn_b,i/dk (via viol) and through
# dw_dk_i (via w). `_erode_violation_field` returns `m` together with a
# `redistribute` callable
#     R_nb, R_w = redistribute(Q)
# implementing the analytic gradient operator
#     sum_i Q_i*dm_i/dk = sum_i (R_nb[i]*dn_b,i/dk + R_w[i]*dw_dk_i)
# so that, for any per-cell array Q,
#     d(sum_i w_i*m_i**2)/dk  uses Q = 2*m*w
#     d(sum_i w_i*m_i)   /dk  uses Q = w
# `r_bridge<=0` disables erosion (`m=viol`,
# `redistribute(Q) = (-Q*H_viol, 0)`), exactly recovering the un-eroded
# penalty (dm_i/dk = dviol_i/dk = -H_viol_i*dn_b,i/dk).


def _voxel_grid_index(pts_mm: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Bin point positions into a uniform voxel grid of the given pitch.

    Returns (vox_idx (N,) int — linear voxel index per point,
             grid_shape (3,) int, n_vox).
    """
    mins = pts_mm.min(axis=0)
    maxs = pts_mm.max(axis=0)
    grid_shape = np.maximum(1, np.ceil((maxs - mins) / voxel_size).astype(int) + 1)
    idx = np.floor((pts_mm - mins) / voxel_size).astype(int)
    idx = np.clip(idx, 0, grid_shape - 1)
    vox_idx = idx[:, 0] + grid_shape[0] * (idx[:, 1] + grid_shape[1] * idx[:, 2])
    n_vox = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
    return vox_idx, grid_shape, n_vox


def _box_filter_3x3x3(field_vox: np.ndarray) -> np.ndarray:
    """Sum each voxel with its 26 face/edge/corner neighbours (zero-padded at
    grid edges), via three separable 1-D ±1-window passes."""
    out = field_vox
    for axis in range(3):
        plus  = np.zeros_like(out)
        minus = np.zeros_like(out)
        dst, src = [slice(None)] * 3, [slice(None)] * 3
        dst[axis], src[axis] = slice(1, None), slice(None, -1)
        plus[tuple(dst)] = out[tuple(src)]
        dst, src = [slice(None)] * 3, [slice(None)] * 3
        dst[axis], src[axis] = slice(None, -1), slice(1, None)
        minus[tuple(dst)] = out[tuple(src)]
        out = out + plus + minus
    return out


def _erode_violation_field(
    pts_mm:     np.ndarray,  # (N, 3)
    viol:       np.ndarray,  # (N,)  >= 0
    w:          np.ndarray,  # (N,)  interface weight
    r_bridge:   float,       # erosion radius (mm), = L_bridge / 2
    bridge_eps: float,       # soft-min sharpness scale (violation units)
) -> tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """Smooth, interface-weighted morphological erosion of `viol` by radius
    `r_bridge`.

    Returns (m, redistribute) — see module section docstring above.
    """
    H_viol = (viol > 0.0).astype(float)

    if r_bridge <= 0.0:
        zeros = np.zeros_like(viol)

        def redistribute(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return -Q * H_viol, zeros

        return viol, redistribute

    TINY = 1e-300

    vox_idx, grid_shape, n_vox = _voxel_grid_index(pts_mm, r_bridge)
    # vox_idx = idx_x + gx*(idx_y + gy*idx_z) is the C-order flat index of an
    # array shaped (gz, gy, gx) — reverse grid_shape (which is (gx,gy,gz))
    # before reshape so that .reshape(...).ravel() round-trips through
    # vox_idx correctly (matters whenever gx != gz).
    grid_shape_zyx = tuple(grid_shape[::-1])

    sum_w     = np.bincount(vox_idx, weights=w,        minlength=n_vox)
    sum_wviol = np.bincount(vox_idx, weights=w * viol, minlength=n_vox)
    vbar = sum_wviol / (sum_w + TINY)

    beta = 1.0 / bridge_eps
    E = np.exp(-beta * vbar)

    omega_grid = sum_w.reshape(grid_shape_zyx)
    E_grid     = E.reshape(grid_shape_zyx)
    S = _box_filter_3x3x3(E_grid * omega_grid).ravel()
    C = _box_filter_3x3x3(omega_grid).ravel()

    ratio = np.where(C > TINY, S / np.maximum(C, TINY), 1.0)
    m_vox = -(1.0 / beta) * np.log(ratio)
    m = m_vox[vox_idx]

    inv_S = np.where(S > TINY, 1.0 / S, 0.0)
    inv_C = np.where(C > TINY, 1.0 / C, 0.0)

    def redistribute(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Qv = np.bincount(vox_idx, weights=Q, minlength=n_vox)
        T  = _box_filter_3x3x3((Qv * inv_S).reshape(grid_shape_zyx)).ravel()
        U  = _box_filter_3x3x3((Qv * inv_C).reshape(grid_shape_zyx)).ravel()

        alpha = E * T
        gamma_ = (1.0 / beta) * (alpha - U)

        alpha_i = alpha[vox_idx]
        gamma_i = gamma_[vox_idx]
        vbar_i  = vbar[vox_idx]

        R_nb = -alpha_i * w * H_viol
        R_w  = alpha_i * (viol - vbar_i) - gamma_i
        return R_nb, R_w

    return m, redistribute


# ── Analytic overhang computation ────────────────────────────────────────────

def compute_gyroid_overhang(
    pts_mm:  np.ndarray,   # (N, 3)  cell-centre positions in mm
    freq_mm: np.ndarray,   # (N, 3)  [kx, ky, kz] at each cell (rad/mm)
    gamma:   np.ndarray,   # (N,)    γ = sigmoid(SDF/ε)
    sdf:     np.ndarray,   # (N,)    |G| − half_thickness (kept for API)
    epsilon: float,        # sigmoid sharpness (mm)
    cos_max: float,        # cos(θ_max), e.g. cos(45°) = 1/√2
    b_vec:   np.ndarray,   # (3,)    build-direction unit vector (normalised)
    mu_oh:   float,        # penalty multiplier
    W:       np.ndarray,   # (N, N_ctrl)  RBF evaluation Jacobian
    rot_matrix: np.ndarray | None = None,  # 3×3 gyroid rotation matrix (or None)
    eps_reg: float = 1e-6,
    r_bridge:   float = 0.75,  # bridging-length erosion radius (mm) = L_bridge/2
    bridge_eps: float = 0.02,  # erosion soft-min sharpness (violation units)
    p_agg:      float = 2.0,   # Lp-norm exponent for J_oh's voxel aggregation (>=2)
) -> tuple[float, np.ndarray, dict]:
    """
    Analytic gyroid overhang penalty and gradient.

    Before forming the penalty, the per-cell violation field `viol` is passed
    through a smooth morphological erosion (radius `r_bridge`, see
    `_erode_violation_field`) so that islands of violation smaller than the
    LPBF bridge length (`2*r_bridge`) contribute ~0 penalty.  `r_bridge<=0`
    disables this and recovers the original per-cell penalty exactly.

    J_oh aggregates the eroded per-voxel violation `m_vox` (broadcast to
    cells as `m`) via an interface-weighted Lp-norm:
        S_p  = sum_i w_i * m_i**p_agg
        J_oh = (mu_oh/N) * S_p**(2/p_agg)
    `p_agg=2` (default) reproduces the original mean-of-squares penalty
    exactly. `p_agg>2` makes J_oh increasingly dominated by the worst
    (largest-m_vox) voxels, up to J_oh -> (mu_oh/N)*max(m_vox)**2 as
    p_agg -> inf — so a single severe, unbridgeable hotspot can no longer be
    averaged away by improvements elsewhere. `g_oh` (below, and the value
    returned by `compute_gyroid_overhang_raw`) remains the plain
    interface-weighted mean.

    Returns
    -------
    J_oh       : float        total penalty value
    grad_ctrl  : (N_ctrl, 3)  dJ_oh/d(dk_ctrl)
    info       : dict         {'g_oh', 'pen_oh'}
    """
    N = len(pts_mm)
    geo = _gyroid_geometry(pts_mm, freq_mm, gamma, epsilon, rot_matrix)

    bx, by, bz = b_vec
    Gx = geo['Gx']; Gy = geo['Gy']; Gz = geo['Gz']
    sG = geo['sG']; inv_ng = geo['inv_ng']

    b_dot_gradG = bx*Gx + by*Gy + bz*Gz
    n_b = sG * b_dot_gradG * inv_ng

    viol = np.maximum(0.0, -cos_max - n_b)
    w    = geo['w']

    m, redistribute = _erode_violation_field(pts_mm, viol, w, r_bridge, bridge_eps)

    TINY = 1e-300
    m_nn = np.maximum(m, 0.0)   # guard against ~0 negative FP noise (m_vox >= 0 analytically)
    m_p  = m_nn ** p_agg
    S_p  = float(np.sum(w * m_p))
    S_p_safe = max(S_p, TINY)
    J_oh = mu_oh / N * S_p_safe ** (2.0 / p_agg)

    w_sum = float(np.sum(w)) + 1e-30
    g_oh  = float(np.sum(w * m)) / w_sum

    # ── Gradient ──────────────────────────────────────────────────────────────
    # ∂n_b/∂kα = sG·∂(b·∇G)/∂kα·inv_ng − n_b·∂|∇G|/∂kα·inv_ng
    # ∂(b·∇G)/∂kα = bx·dGx_dkα + by·dGy_dkα + bz·dGz_dkα
    # ∂|∇G|/∂kα = (Gx·dGx_dkα + Gy·dGy_dkα + Gz·dGz_dkα)·inv_ng

    dBG_dkx = bx*geo['dGx_dkx'] + by*geo['dGy_dkx'] + bz*geo['dGz_dkx']
    dBG_dky = bx*geo['dGx_dky'] + by*geo['dGy_dky'] + bz*geo['dGz_dky']
    dBG_dkz = bx*geo['dGx_dkz'] + by*geo['dGy_dkz'] + bz*geo['dGz_dkz']

    Gx_v = geo['Gx']; Gy_v = geo['Gy']; Gz_v = geo['Gz']
    dng_dkx = (Gx_v*geo['dGx_dkx'] + Gy_v*geo['dGy_dkx'] + Gz_v*geo['dGz_dkx']) * inv_ng
    dng_dky = (Gx_v*geo['dGx_dky'] + Gy_v*geo['dGy_dky'] + Gz_v*geo['dGz_dky']) * inv_ng
    dng_dkz = (Gx_v*geo['dGx_dkz'] + Gy_v*geo['dGy_dkz'] + Gz_v*geo['dGz_dkz']) * inv_ng

    dnb_dkx = sG * dBG_dkx * inv_ng - n_b * dng_dkx * inv_ng
    dnb_dky = sG * dBG_dky * inv_ng - n_b * dng_dky * inv_ng
    dnb_dkz = sG * dBG_dkz * inv_ng - n_b * dng_dkz * inv_ng

    dw_dkx = geo['dw_dkx']; dw_dky = geo['dw_dky']; dw_dkz = geo['dw_dkz']
    c_p = (mu_oh / N) * (2.0 / p_agg) * S_p_safe ** (2.0 / p_agg - 1.0)
    Q = p_agg * m_nn ** (p_agg - 1.0) * w   # d(sum_i w_i*m_i**p_agg)/dm_i, per cell
    R_nb, R_w = redistribute(Q)
    sk_x = c_p * ((m_p + R_w) * dw_dkx + R_nb * dnb_dkx)
    sk_y = c_p * ((m_p + R_w) * dw_dky + R_nb * dnb_dky)
    sk_z = c_p * ((m_p + R_w) * dw_dkz + R_nb * dnb_dkz)

    grad_ctrl = np.stack([W.T @ sk_x, W.T @ sk_y, W.T @ sk_z], axis=1)

    return J_oh, grad_ctrl, {'g_oh': g_oh, 'pen_oh': J_oh}


def compute_gyroid_overhang_raw(
    pts_mm:  np.ndarray,
    freq_mm: np.ndarray,
    gamma:   np.ndarray,
    sdf:     np.ndarray,
    epsilon: float,
    cos_max: float,
    b_vec:   np.ndarray,
    W:       np.ndarray,
    rot_matrix: np.ndarray | None = None,
    eps_reg: float = 1e-6,
    r_bridge:   float = 0.75,  # bridging-length erosion radius (mm) = L_bridge/2
    bridge_eps: float = 0.02,  # erosion soft-min sharpness (violation units)
) -> tuple[float, np.ndarray, dict]:

    N = len(pts_mm)
    geo = _gyroid_geometry(pts_mm, freq_mm, gamma, epsilon, rot_matrix)

    bx, by, bz = b_vec
    Gx = geo['Gx']; Gy = geo['Gy']; Gz = geo['Gz']
    sG = geo['sG']; inv_ng = geo['inv_ng']

    b_dot_gradG = bx*Gx + by*Gy + bz*Gz
    n_b = sG * b_dot_gradG * inv_ng

    viol = np.maximum(0.0, -cos_max - n_b)
    w    = geo['w']

    m, redistribute = _erode_violation_field(pts_mm, viol, w, r_bridge, bridge_eps)

    w_sum = float(np.sum(w)) + 1e-30
    g_oh  = float(np.sum(w * m)) / w_sum

    dBG_dkx = bx*geo['dGx_dkx'] + by*geo['dGy_dkx'] + bz*geo['dGz_dkx']
    dBG_dky = bx*geo['dGx_dky'] + by*geo['dGy_dky'] + bz*geo['dGz_dky']
    dBG_dkz = bx*geo['dGx_dkz'] + by*geo['dGy_dkz'] + bz*geo['dGz_dkz']

    Gx_v = geo['Gx']; Gy_v = geo['Gy']; Gz_v = geo['Gz']
    dng_dkx = (Gx_v*geo['dGx_dkx'] + Gy_v*geo['dGy_dkx'] + Gz_v*geo['dGz_dkx']) * inv_ng
    dng_dky = (Gx_v*geo['dGx_dky'] + Gy_v*geo['dGy_dky'] + Gz_v*geo['dGz_dky']) * inv_ng
    dng_dkz = (Gx_v*geo['dGx_dkz'] + Gy_v*geo['dGy_dkz'] + Gz_v*geo['dGz_dkz']) * inv_ng

    dnb_dkx = sG * dBG_dkx * inv_ng - n_b * dng_dkx * inv_ng
    dnb_dky = sG * dBG_dky * inv_ng - n_b * dng_dky * inv_ng
    dnb_dkz = sG * dBG_dkz * inv_ng - n_b * dng_dkz * inv_ng

    dw_dkx = geo['dw_dkx']; dw_dky = geo['dw_dky']; dw_dkz = geo['dw_dkz']
    c = 1.0 / w_sum
    R_nb, R_w = redistribute(w)   # d(sum_i w_i*m_i)/dk channels
    sk_x = c * ((m - g_oh + R_w) * dw_dkx + R_nb * dnb_dkx)
    sk_y = c * ((m - g_oh + R_w) * dw_dky + R_nb * dnb_dky)
    sk_z = c * ((m - g_oh + R_w) * dw_dkz + R_nb * dnb_dkz)

    grad_ctrl = np.stack([W.T @ sk_x, W.T @ sk_y, W.T @ sk_z], axis=1)

    return g_oh, grad_ctrl, {'g_oh': g_oh}


# ── Parameter container ───────────────────────────────────────────────────────

class AMConstraints:

    def __init__(
        self,
        theta_max:       float              = math.pi / 4.0,
        build_direction: np.ndarray | None  = None,
        P_bar:           float              = 0.01,
        mu_overhang:     float              = 1.0,
        use_overhang:    bool               = True,
        L_bridge_mm:     float              = 1.5,
        bridge_eps:      float              = 0.02,
        p_agg:           float              = 2.0,
    ):
        self.cos_max      = math.cos(theta_max)
        self.b_vec        = (np.array([0.0, 0.0, 1.0], dtype=float)
                             if build_direction is None
                             else np.asarray(build_direction, dtype=float))
        self.b_vec       /= np.linalg.norm(self.b_vec)
        self.P_bar        = P_bar
        self.mu_oh        = mu_overhang
        # Bridging-length erosion: islands of overhang violation smaller than
        # L_bridge erode to ~0 and contribute no penalty (see
        # _erode_violation_field). r_bridge<=0 disables erosion.
        self.L_bridge_mm  = L_bridge_mm
        self.bridge_eps   = bridge_eps
        self.r_bridge     = L_bridge_mm / 2.0
        self.use_overhang = use_overhang
        self.p_agg        = p_agg

    def update_penalties(self, info: dict, violation_tol: float = 1e-4) -> None:
        """Increase mu_oh if overhang violation exceeds P_bar."""
        if self.use_overhang and info.get('g_oh', 0.0) > self.P_bar + violation_tol:
            self.mu_oh = min(self.mu_oh * 1.2, 1e9)
