"""
Analytic gyroid overhang constraint.

The outward solid normal is computed analytically from the gyroid implicit surface:

    G(x,y,z; kx,ky,kz) = sin(kx·x)cos(ky·y)
                        + sin(ky·y)cos(kz·z)
                        + sin(kz·z)cos(kx·x)

    outward solid normal  =  sign(G) × ∇G / |∇G|

An overhang violation occurs wherever the normal points too far downward (below
the self-supporting threshold θ_max from vertical):

    viol = max(0,  −cos(θ_max) − (n̂ · b̂))

where b̂ is the build direction (+z by default).

The penalty and its gradient w.r.t. the RBF frequency control-point
perturbations dk_ctrl are computed fully analytically — no Helmholtz filter,
no finite-difference stencil, no discrete gradient operator.

Surface localisation uses the existing smooth sigmoid:

    w = γ (1 − γ) / ε           (peaks at the solid/fluid interface, SDF ≈ 0)

so only cells near the gyroid wall surface contribute to the penalty.
"""

from __future__ import annotations

import math
import numpy as np


# ── Analytic overhang computation ────────────────────────────────────────────

def compute_gyroid_overhang(
    pts_mm:  np.ndarray,   # (N, 3)  cell-centre positions in mm
    freq_mm: np.ndarray,   # (N, 3)  [kx, ky, kz] at each cell (rad/mm)
    gamma:   np.ndarray,   # (N,)    γ = sigmoid(SDF/ε)
    sdf:     np.ndarray,   # (N,)    |G| − half_thickness (unused but kept for API)
    epsilon: float,        # sigmoid sharpness (mm)
    cos_max: float,        # cos(θ_max), e.g. cos(45°) = 1/√2
    b_vec:   np.ndarray,   # (3,)    build-direction unit vector (normalised)
    mu_oh:   float,        # penalty multiplier
    W:       np.ndarray,   # (N, N_ctrl)  RBF evaluation Jacobian
    eps_reg: float = 1e-6, # |∇G| regularisation (prevents division by zero)
) -> tuple[float, np.ndarray, dict]:
    """
    Analytic gyroid overhang penalty and gradient.

    Returns
    -------
    J_oh       : float        total penalty value (add directly to J_aug)
    grad_ctrl  : (N_ctrl, 3)  dJ_oh/d(dk_ctrl); add to the thermal grad_ctrl
    info       : dict         {'g_oh', 'pen_oh'} for logging / penalty updates
    """
    N = len(pts_mm)
    x = pts_mm[:, 0];  y = pts_mm[:, 1];  z = pts_mm[:, 2]
    kx = freq_mm[:, 0]; ky = freq_mm[:, 1]; kz = freq_mm[:, 2]
    bx, by, bz = b_vec

    # ── Trig values ─────────────────────────────────────────────────────────
    sx = np.sin(kx * x);  cx = np.cos(kx * x)
    sy = np.sin(ky * y);  cy = np.cos(ky * y)
    sz = np.sin(kz * z);  cz = np.cos(kz * z)

    # ── Gyroid value and spatial-gradient factors ────────────────────────────
    # G = sx·cy + sy·cz + sz·cx
    # ∂G/∂x = kx · Ax,   Ax = cx·cy − sz·sx
    # ∂G/∂y = ky · Ay,   Ay = cy·cz − sx·sy
    # ∂G/∂z = kz · Az,   Az = cz·cx − sy·sz
    G  = sx * cy + sy * cz + sz * cx
    Ax = cx * cy - sz * sx
    Ay = cy * cz - sx * sy
    Az = cz * cx - sy * sz
    Gx = kx * Ax
    Gy = ky * Ay
    Gz = kz * Az

    # ── Regularised gradient magnitude ──────────────────────────────────────
    ng2 = Gx**2 + Gy**2 + Gz**2 + eps_reg**2
    ng  = np.sqrt(ng2)
    inv_ng = 1.0 / ng

    # ── Outward solid normal · build direction ───────────────────────────────
    # n̂ = sign(G) × ∇G / |∇G|
    # n_b = n̂ · b̂ = sign(G) × (b · ∇G) / |∇G|
    sG = np.sign(G)
    sG[sG == 0] = 1.0   # G=0 is mid-wall; w≈0 there so sign choice is irrelevant
    b_dot_gradG = bx * Gx + by * Gy + bz * Gz
    n_b = sG * b_dot_gradG * inv_ng

    # ── Overhang violation and surface weight ────────────────────────────────
    # viol > 0 when surface points more than θ_max below horizontal
    viol = np.maximum(0.0, -cos_max - n_b)
    w    = gamma * (1.0 - gamma) / epsilon   # surface indicator ≥ 0

    # ── Penalty ─────────────────────────────────────────────────────────────
    pen_j = w * viol**2
    J_oh  = mu_oh / N * float(np.sum(pen_j))
    # Monitoring value: surface-weighted mean violation (comparable to old g_oh)
    w_sum = float(np.sum(w)) + 1e-30
    g_oh  = float(np.sum(w * np.maximum(0.0, -cos_max - n_b))) / w_sum

    # ── Gradient dJ_oh/dk_{α,l} = Σ_j [∂J_oh/∂kα_j] × W[j,l] ─────────────
    #
    # ∂J_oh/∂kα_j = (μ/N) [ viol_j² · ∂w_j/∂kα_j
    #                        − 2 w_j viol_j · ∂n_b_j/∂kα_j ]
    #
    # Non-zero only where viol_j > 0 (both viol² and viol kill inactive cells).

    # ∂G/∂kα  (needed for the ∂w/∂kα term)
    dG_dkx = x * Ax
    dG_dky = y * Ay
    dG_dkz = z * Az

    # Mixed partials ∂(∂G/∂coord)/∂kα  (9 terms, all analytic)
    # Gx = kx(cx·cy − sz·sx)
    dGx_dkx = Ax - kx * x * (sx * cy + sz * cx)
    dGx_dky = -kx * y * cx * sy
    dGx_dkz = -kx * z * cz * sx

    # Gy = ky(cy·cz − sx·sy)
    dGy_dkx = -ky * x * cx * sy
    dGy_dky = Ay - ky * y * (sy * cz + sx * cy)
    dGy_dkz = -ky * z * cy * sz

    # Gz = kz(cz·cx − sy·sz)
    dGz_dkx = -kz * x * cz * sx
    dGz_dky = -kz * y * cy * sz
    dGz_dkz = Az - kz * z * (sz * cx + sy * cz)

    # ∂(b·∇G)/∂kα
    dBG_dkx = bx * dGx_dkx + by * dGy_dkx + bz * dGz_dkx
    dBG_dky = bx * dGx_dky + by * dGy_dky + bz * dGz_dky
    dBG_dkz = bx * dGx_dkz + by * dGy_dkz + bz * dGz_dkz

    # ∂|∇G|/∂kα = (Gx·∂Gx/∂kα + Gy·∂Gy/∂kα + Gz·∂Gz/∂kα) / |∇G|
    dng_dkx = (Gx * dGx_dkx + Gy * dGy_dkx + Gz * dGz_dkx) * inv_ng
    dng_dky = (Gx * dGx_dky + Gy * dGy_dky + Gz * dGz_dky) * inv_ng
    dng_dkz = (Gx * dGx_dkz + Gy * dGy_dkz + Gz * dGz_dkz) * inv_ng

    # ∂n_b/∂kα = sG · ∂(b·∇G)/∂kα / |∇G|  −  n_b · ∂|∇G|/∂kα / |∇G|
    dnb_dkx = sG * dBG_dkx * inv_ng - n_b * dng_dkx * inv_ng
    dnb_dky = sG * dBG_dky * inv_ng - n_b * dng_dky * inv_ng
    dnb_dkz = sG * dBG_dkz * inv_ng - n_b * dng_dkz * inv_ng

    # ∂w/∂kα = (1−2γ)·γ(1−γ)/ε² · sign(G) · ∂G/∂kα
    #        = (1−2γ)·w/ε · sign(G) · ∂G/∂kα
    dw_factor = (1.0 - 2.0 * gamma) * w / epsilon * sG
    dw_dkx = dw_factor * dG_dkx
    dw_dky = dw_factor * dG_dky
    dw_dkz = dw_factor * dG_dkz

    # Combine: sk_α[j] = ∂J_oh/∂kα_j
    c = mu_oh / N
    sk_x = c * (viol**2 * dw_dkx - 2.0 * w * viol * dnb_dkx)
    sk_y = c * (viol**2 * dw_dky - 2.0 * w * viol * dnb_dky)
    sk_z = c * (viol**2 * dw_dkz - 2.0 * w * viol * dnb_dkz)

    # Project through RBF Jacobian: grad_ctrl[l, α] = Σ_j sk_α[j] · W[j,l]
    grad_ctrl = np.stack([W.T @ sk_x, W.T @ sk_y, W.T @ sk_z], axis=1)  # (N_ctrl, 3)

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
    eps_reg: float = 1e-6,
) -> tuple[float, np.ndarray, dict]:
    """
    Analytic overhang constraint value g_oh and its gradient d(g_oh)/d(dk_ctrl).

    Unlike compute_gyroid_overhang, this returns the raw surface-weighted mean
    violation (not the penalised squared version) and its exact gradient, suitable
    for use as a proper inequality constraint in trust-constr optimisers.

    g_oh = sum_j(w_j * max(0, -cos_max - n_b_j)) / sum_j(w_j)

    d(g_oh)/d(k_alpha_j) = (1/w_sum) *
        sum_j [(viol_j - g_oh) * dw_j/dk_alpha  -  w_j * H(viol_j) * dn_b_j/dk_alpha]

    where H is the Heaviside indicator (1 when viol > 0, else 0).

    Returns
    -------
    g_oh       : float        surface-weighted mean overhang violation
    grad_ctrl  : (N_ctrl, 3)  d(g_oh)/d(dk_ctrl); negate and pass to scipy constraint jac
    info       : dict         {'g_oh'}
    """
    N = len(pts_mm)
    x = pts_mm[:, 0];  y = pts_mm[:, 1];  z = pts_mm[:, 2]
    kx = freq_mm[:, 0]; ky = freq_mm[:, 1]; kz = freq_mm[:, 2]
    bx, by, bz = b_vec

    sx = np.sin(kx * x);  cx = np.cos(kx * x)
    sy = np.sin(ky * y);  cy = np.cos(ky * y)
    sz = np.sin(kz * z);  cz = np.cos(kz * z)

    G  = sx * cy + sy * cz + sz * cx
    Ax = cx * cy - sz * sx
    Ay = cy * cz - sx * sy
    Az = cz * cx - sy * sz
    Gx = kx * Ax
    Gy = ky * Ay
    Gz = kz * Az

    ng2 = Gx**2 + Gy**2 + Gz**2 + eps_reg**2
    ng  = np.sqrt(ng2)
    inv_ng = 1.0 / ng

    sG = np.sign(G)
    sG[sG == 0] = 1.0
    b_dot_gradG = bx * Gx + by * Gy + bz * Gz
    n_b = sG * b_dot_gradG * inv_ng

    viol  = np.maximum(0.0, -cos_max - n_b)
    H_viol = (viol > 0.0).astype(float)
    w     = gamma * (1.0 - gamma) / epsilon

    w_sum = float(np.sum(w)) + 1e-30
    g_oh  = float(np.sum(w * viol)) / w_sum

    # ── Gradient of G w.r.t. k_alpha (spatial coords) ──────────────────────
    dG_dkx = x * Ax
    dG_dky = y * Ay
    dG_dkz = z * Az

    # Mixed partials ∂(∂G/∂coord)/∂k_alpha
    dGx_dkx = Ax - kx * x * (sx * cy + sz * cx)
    dGx_dky = -kx * y * cx * sy
    dGx_dkz = -kx * z * cz * sx

    dGy_dkx = -ky * x * cx * sy
    dGy_dky = Ay - ky * y * (sy * cz + sx * cy)
    dGy_dkz = -ky * z * cy * sz

    dGz_dkx = -kz * x * cz * sx
    dGz_dky = -kz * y * cy * sz
    dGz_dkz = Az - kz * z * (sz * cx + sy * cz)

    dBG_dkx = bx * dGx_dkx + by * dGy_dkx + bz * dGz_dkx
    dBG_dky = bx * dGx_dky + by * dGy_dky + bz * dGz_dky
    dBG_dkz = bx * dGx_dkz + by * dGy_dkz + bz * dGz_dkz

    dng_dkx = (Gx * dGx_dkx + Gy * dGy_dkx + Gz * dGz_dkx) * inv_ng
    dng_dky = (Gx * dGx_dky + Gy * dGy_dky + Gz * dGz_dky) * inv_ng
    dng_dkz = (Gx * dGx_dkz + Gy * dGy_dkz + Gz * dGz_dkz) * inv_ng

    dnb_dkx = sG * dBG_dkx * inv_ng - n_b * dng_dkx * inv_ng
    dnb_dky = sG * dBG_dky * inv_ng - n_b * dng_dky * inv_ng
    dnb_dkz = sG * dBG_dkz * inv_ng - n_b * dng_dkz * inv_ng

    # ∂w/∂k_alpha = (1−2γ)·w/ε · sign(G) · ∂G/∂k_alpha
    dw_factor = (1.0 - 2.0 * gamma) * w / epsilon * sG
    dw_dkx = dw_factor * dG_dkx
    dw_dky = dw_factor * dG_dky
    dw_dkz = dw_factor * dG_dkz

    # Per-cell weight: d(g_oh)/dk_alpha_j = (1/w_sum) * [
    #   (viol_j - g_oh) * dw_j/dk_alpha  -  w_j * H(viol_j) * dnb_j/dk_alpha ]
    c = 1.0 / w_sum
    sk_x = c * ((viol - g_oh) * dw_dkx - w * H_viol * dnb_dkx)
    sk_y = c * ((viol - g_oh) * dw_dky - w * H_viol * dnb_dky)
    sk_z = c * ((viol - g_oh) * dw_dkz - w * H_viol * dnb_dkz)

    grad_ctrl = np.stack([W.T @ sk_x, W.T @ sk_y, W.T @ sk_z], axis=1)  # (N_ctrl, 3)

    return g_oh, grad_ctrl, {'g_oh': g_oh}


# ── Parameter container ───────────────────────────────────────────────────────

class AMConstraints:
    """
    Stores overhang constraint parameters and the adaptive penalty weight.
    The computation itself is in compute_gyroid_overhang().
    """

    def __init__(
        self,
        theta_max:       float              = math.pi / 4.0,
        build_direction: np.ndarray | None  = None,
        P_bar:           float              = 0.01,
        mu_overhang:     float              = 1.0,
        use_overhang:    bool               = True,
    ):
        self.cos_max      = math.cos(theta_max)
        self.b_vec        = (np.array([0.0, 0.0, 1.0], dtype=float)
                             if build_direction is None
                             else np.asarray(build_direction, dtype=float))
        self.b_vec       /= np.linalg.norm(self.b_vec)
        self.P_bar        = P_bar
        self.mu_oh        = mu_overhang
        self.use_overhang = use_overhang

    def update_penalties(self, info: dict, violation_tol: float = 1e-4) -> None:
        """Increase mu_oh if overhang violation exceeds P_bar."""
        if self.use_overhang and info.get('g_oh', 0.0) > self.P_bar + violation_tol:
            self.mu_oh = min(self.mu_oh * 1.2, 1e6)
