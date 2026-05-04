"""
Additive Manufacturing Constraints for the Gyroid RBF Topology Optimizer.

Implements:
  1. HelmholtzFilter   – PDE smoother: -R² Δγ̃ + γ̃ = γ_raw
  2. modified_heaviside / d_modified_heaviside – robust dilation/erosion projection
  3. compute_overhang_constraint  – self-support angle integral + sensitivity
  4. compute_thickness_constraint – minimum solid wall thickness integral + sensitivity

All sensitivities are analytical (adjoint-consistent).

References
----------
Wang, Lazarov & Sigmund (2011) – Robust topology optimization for minimum length scale
Lazarov & Sigmund (2016) – Filters in topology optimization based on Helmholtz-type PDEs
Guest et al. (2004) – Achieving minimum length scale in topology optimization
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg


# ── 1. Helmholtz PDE filter ─────────────────────────────────────────────────────

class HelmholtzFilter:
    """
    Node-based Helmholtz PDE filter on an unstructured cell-centre mesh.

    Solves: (I + R² L) γ̃ = γ_raw
    where L is the distance-weighted graph Laplacian (positive semi-definite).

    The physical filter radius r and PDE kernel R are related by r = R/(2√3),
    equivalently R = r · 2√3.

    Parameters
    ----------
    pts_mm      : (N, 3) cell-centre positions in mm
    r_filter_mm : physical filter radius in mm (sets the minimum length scale)
    n_neighbors : nearest cells used to build the graph Laplacian (default 6)
    cg_tol      : iterative solver tolerance
    """

    def __init__(self,
                 pts_mm: np.ndarray,
                 r_filter_mm: float,
                 n_neighbors: int = 6,
                 cg_tol: float = 1e-6):
        self.N      = len(pts_mm)
        self.cg_tol = cg_tol
        R = r_filter_mm * 2.0 * np.sqrt(3.0)   # PDE kernel radius
        self._build(pts_mm, R, n_neighbors)

    def _build(self, pts_mm: np.ndarray, R: float, k: int) -> None:
        N = self.N
        print(f"  [Helmholtz] Building filter matrix  N={N:,}  R={R:.4f} mm  k={k} …")
        tree  = KDTree(pts_mm)
        _, ix = tree.query(pts_mm, k=k + 1)   # (N, k+1) – col 0 is self
        nbr   = ix[:, 1:]                      # (N, k)

        dp    = pts_mm[nbr] - pts_mm[:, np.newaxis, :]   # (N, k, 3)
        dist2 = np.maximum(np.sum(dp**2, axis=2), 1e-30) # (N, k)
        w     = 1.0 / dist2                               # (N, k) weights

        # System matrix H = I + R² L
        # H[i,j] = -R² w_ij  (off-diagonal)
        # H[i,i] = 1 + R² Σ_j w_ij
        row_i   = np.repeat(np.arange(N), k)
        col_j   = nbr.ravel()
        off     = -R**2 * w.ravel()
        diag    = 1.0 + R**2 * w.sum(axis=1)

        rows = np.concatenate([row_i,        np.arange(N)])
        cols = np.concatenate([col_j,        np.arange(N)])
        vals = np.concatenate([off,          diag])
        self._H = csr_matrix((vals, (rows, cols)), shape=(N, N))
        print(f"  [Helmholtz] Done  nnz={self._H.nnz:,}")

    def apply(self, gamma_raw: np.ndarray) -> np.ndarray:
        """Forward solve: H γ̃ = γ_raw  →  γ̃."""
        sol, info = cg(self._H, gamma_raw, rtol=self.cg_tol, maxiter=1000)
        if info != 0:
            print(f"  [Helmholtz] forward CG did not converge (info={info})")
        return np.clip(sol, 0.0, 1.0)

    def apply_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Adjoint solve (H is symmetric, so identical to forward): H λ = rhs."""
        sol, info = cg(self._H, rhs, rtol=self.cg_tol, maxiter=1000)
        if info != 0:
            print(f"  [Helmholtz] adjoint CG did not converge (info={info})")
        return sol


# ── 2. Modified Heaviside projection ────────────────────────────────────────────

def modified_heaviside(t: np.ndarray, eta_d: float, beta: float) -> np.ndarray:
    """
    Smooth threshold projection (Wang et al. 2011).

    if t <= eta_d:
        P = eta_d * (exp(-β(1 - t/η)) - (1 - t/η) exp(-β))
    else:
        P = (1-η) * (1 - exp(-β(t-η)/(1-η))) + ((t-η)/(1-η)) exp(-β) + η

    eta_d = 0.05  →  dilation  (expands solid: maps most values toward 1)
    eta_d = 0.95  →  erosion   (contracts solid: maps most values toward 0)
    """
    eb = np.exp(-beta)
    ed = max(eta_d, 1e-12)
    eu = max(1.0 - eta_d, 1e-12)

    r_lo = (1.0 - t / ed)
    P_lo = ed * (np.exp(-beta * r_lo) - r_lo * eb)

    r_hi = (t - eta_d) / eu
    P_hi = eu * (1.0 - np.exp(-beta * r_hi)) + r_hi * eb + eta_d

    return np.where(t <= eta_d, P_lo, P_hi)


def d_modified_heaviside(t: np.ndarray, eta_d: float, beta: float) -> np.ndarray:
    """Analytical derivative dP/dt of modified_heaviside."""
    eb = np.exp(-beta)
    ed = max(eta_d, 1e-12)
    eu = max(1.0 - eta_d, 1e-12)

    r_lo  = (1.0 - t / ed)
    dP_lo = beta * np.exp(-beta * r_lo) + eb   # chain rule through r_lo = ... /ed

    r_hi  = (t - eta_d) / eu
    dP_hi = beta * np.exp(-beta * r_hi) + eb / eu

    return np.where(t <= eta_d, dP_lo, dP_hi)


def beta_schedule(iteration: int,
                  beta_init: float = 1.0,
                  double_every: int = 40,
                  beta_max: float = 16.0) -> float:
    """Projection steepness that doubles every `double_every` iterations."""
    n_doublings = iteration // double_every
    return min(beta_init * (2.0 ** n_doublings), beta_max)


# ── 3. Mesh gradient operator (weighted least-squares) ──────────────────────────

class MeshGradientOperator:
    """
    Sparse gradient matrices Gx, Gy, Gz computed via inverse-distance-weighted
    least squares over k nearest neighbours.

    (Gx @ f)[i] ≈ ∂f/∂x at cell i,  similarly Gy, Gz.
    """

    def __init__(self, pts_mm: np.ndarray, n_neighbors: int = 6):
        N = len(pts_mm)
        print(f"  [GradOp] Building gradient operators  N={N:,}  k={n_neighbors} …")
        tree  = KDTree(pts_mm)
        _, ix = tree.query(pts_mm, k=n_neighbors + 1)
        nbr   = ix[:, 1:]          # (N, k)
        k     = nbr.shape[1]

        dp    = pts_mm[nbr] - pts_mm[:, np.newaxis, :]   # (N, k, 3)
        dist2 = np.sum(dp**2, axis=2)                     # (N, k)
        w     = 1.0 / np.maximum(dist2, 1e-20)            # (N, k)

        # Weighted least squares: min_g Σ_j w_j (dp_j·g - Δf_j)²
        # Solution: g = (A^T W A)^{-1} A^T W Δf
        # Where A = dp  (k×3),  Δf = f[nbr] - f[i]
        AtWA  = np.einsum('nki,nk,nkj->nij', dp, w, dp) + 1e-10 * np.eye(3)  # (N,3,3)
        AtWA_inv = np.linalg.inv(AtWA)                                          # (N,3,3)
        dpt_w    = dp.transpose(0, 2, 1) * w[:, np.newaxis, :]                 # (N,3,k)
        AtW      = np.einsum('nij,njk->nik', AtWA_inv, dpt_w)                  # (N,3,k)

        # g_i = AtW[i] @ (f[nbr[i]] - f[i])
        #      = AtW[i] @ f[nbr[i]]  -  sum(AtW[i], axis=1) * f[i]
        AtW_sum = AtW.sum(axis=2)   # (N, 3) — self coefficient (negative)

        row_i = np.repeat(np.arange(N), k)
        col_j = nbr.ravel()
        rows  = np.concatenate([row_i,       np.arange(N)])
        cols  = np.concatenate([col_j,       np.arange(N)])

        self.Gx = csr_matrix(
            (np.concatenate([AtW[:, 0, :].ravel(), -AtW_sum[:, 0]]), (rows, cols)),
            shape=(N, N))
        self.Gy = csr_matrix(
            (np.concatenate([AtW[:, 1, :].ravel(), -AtW_sum[:, 1]]), (rows, cols)),
            shape=(N, N))
        self.Gz = csr_matrix(
            (np.concatenate([AtW[:, 2, :].ravel(), -AtW_sum[:, 2]]), (rows, cols)),
            shape=(N, N))
        print(f"  [GradOp] Done  nnz={self.Gx.nnz:,}")

    def gradient(self, f: np.ndarray):
        """Return (∂f/∂x, ∂f/∂y, ∂f/∂z), each shape (N,)."""
        return self.Gx @ f, self.Gy @ f, self.Gz @ f


# ── 4. Overhang angle constraint ────────────────────────────────────────────────

def compute_overhang_constraint(
    gamma: np.ndarray,
    grad_op: MeshGradientOperator,
    vol_per_cell: np.ndarray,
    b_vec: np.ndarray,
    theta_max: float = np.pi / 4.0,
    beta_q: float    = 16.0,
    eps_reg: float   = 1e-8,
) -> tuple[float, np.ndarray]:
    """
    Overhang constraint via density-gradient alignment integral.

    g_oh = Σ_i V_i · f(a_i) · a_i  ≤  P_bar

    where
        a_i   = (b · ∇γ_i) / (|∇γ_i| + ε)          alignment factor
        f(a)  = sigmoid(-2 β_q (a - cos θ_max))       soft angle penalty

    Returns
    -------
    g_val       : scalar constraint value
    dg_dgamma   : (N,) sensitivity ∂g/∂γ (before filter adjoint)
    """
    bx, by, bz = b_vec
    cos_t = np.cos(theta_max)

    gx, gy, gz   = grad_op.gradient(gamma)
    b_dg         = bx * gx + by * gy + bz * gz                   # b · ∇γ
    norm_g       = np.sqrt(gx**2 + gy**2 + gz**2 + eps_reg**2)  # |∇γ| regularised
    a            = b_dg / norm_g                                  # alignment

    # Soft Heaviside penalty:  q = -2β(a - cos_t),  f = sigmoid(q)
    # df/da = (df/dq)*(dq/da) = f*(1-f) * (-2β)   ← negative above cos_t
    q      = -2.0 * beta_q * (a - cos_t)
    f      = 1.0 / (1.0 + np.exp(-q))
    df_da  = -2.0 * beta_q * f * (1.0 - f)                       # ∂f/∂a < 0 for a > cos_t

    g_val = float(np.dot(vol_per_cell, f * a))

    # ∂g/∂a_i = V_i (df/da · a + f)
    p = vol_per_cell * (df_da * a + f)   # (N,)

    # ∂a/∂(gx) = bx/n - a·gx/n²  etc.
    n2 = norm_g**2
    wa_x = p * (bx / norm_g - a * gx / n2)
    wa_y = p * (by / norm_g - a * gy / n2)
    wa_z = p * (bz / norm_g - a * gz / n2)

    # ∂g/∂γ = Gx^T wa_x + Gy^T wa_y + Gz^T wa_z
    dg_dgamma = grad_op.Gx.T @ wa_x + grad_op.Gy.T @ wa_y + grad_op.Gz.T @ wa_z

    return g_val, dg_dgamma


# ── 5. Minimum solid wall-thickness constraint ──────────────────────────────────

def compute_thickness_constraint(
    gamma_tilde: np.ndarray,
    vol_per_cell: np.ndarray,
    beta_proj: float,
    eta_dil: float = 0.05,
    eta_ero: float = 0.95,
) -> tuple[float, np.ndarray]:
    """
    Minimum solid wall-thickness constraint (robust dilation/erosion formulation).

    Uses the solid density ρ = 1 - γ̃  (ρ≈1 solid, ρ≈0 fluid).

    ρ_dil = P(ρ, η_dil, β)   — solid dilated (expands solid into fluid)
    ρ_ero = P(ρ, η_ero, β)   — solid eroded  (contracts solid)

    g_th = Σ_i V_i · ρ_dil_i · (1 - ρ_ero_i) / V_total  ≤  Φ_o

    Interpretation:
        Thin solid wall  → ρ_dil≈1 but ρ_ero≈0  →  large integrand  →  constraint violated
        Thick solid wall → both ρ_dil≈1, ρ_ero≈1 →  integrand ≈ 0   →  satisfied
        Fluid region     → both ≈0               →  integrand ≈ 0   →  satisfied

    Returns
    -------
    g_val           : scalar constraint value
    dg_dgamma_tilde : (N,) sensitivity ∂g/∂γ̃ (before filter adjoint)
    """
    rho       = 1.0 - gamma_tilde              # solid density
    rho_dil   = modified_heaviside(rho, eta_dil, beta_proj)
    rho_ero   = modified_heaviside(rho, eta_ero, beta_proj)
    drho_dil  = d_modified_heaviside(rho, eta_dil, beta_proj)
    drho_ero  = d_modified_heaviside(rho, eta_ero, beta_proj)

    V_total = vol_per_cell.sum()
    integrand = rho_dil * (1.0 - rho_ero)
    g_val     = float(np.dot(vol_per_cell, integrand)) / V_total

    # ∂g/∂ρ_i = V_i/V ( dρ_dil·(1-ρ_ero) - ρ_dil·dρ_ero )
    dg_drho = vol_per_cell / V_total * (
        drho_dil * (1.0 - rho_ero) - rho_dil * drho_ero
    )

    # ρ = 1 - γ̃  →  ∂g/∂γ̃ = -∂g/∂ρ
    dg_dgamma_tilde = -dg_drho

    return g_val, dg_dgamma_tilde


# ── 6. Unified AM constraint evaluator ──────────────────────────────────────────

class AMConstraints:
    """
    Wraps all AM constraints and provides a single call interface for the
    optimizer.  Pre-builds all mesh operators (KD-trees, sparse matrices) once.

    Parameters
    ----------
    pts_mm          : (N, 3) cell-centre positions in mm
    r_filter_mm     : Helmholtz filter radius (sets min length scale), mm
    build_direction : unit vector pointing away from build plate (default +Z)
    theta_max       : max overhang half-angle, radians (default π/4 = 45°)
    beta_q          : steepness of overhang Heaviside (default 16)
    P_bar           : overhang constraint upper bound (default 0.01)
    Phi_o           : thickness constraint upper bound (default 0.01)
    mu_overhang     : initial augmented-Lagrangian penalty weight for overhang
    mu_thickness    : initial augmented-Lagrangian penalty weight for thickness
    use_overhang    : whether to activate the overhang constraint (default True)
    use_thickness   : whether to activate the thickness constraint (default True)
    """

    def __init__(
        self,
        pts_mm:          np.ndarray,
        r_filter_mm:     float       = 0.15,
        build_direction: np.ndarray  = None,
        theta_max:       float       = np.pi / 4.0,
        beta_q:          float       = 16.0,
        P_bar:           float       = 0.01,
        Phi_o:           float       = 0.01,
        mu_overhang:     float       = 1.0,
        mu_thickness:    float       = 10.0,
        use_overhang:    bool        = True,
        use_thickness:   bool        = True,
    ):
        self.r_filter     = r_filter_mm
        self.b_vec        = (np.array([0.0, 0.0, 1.0])
                             if build_direction is None
                             else np.asarray(build_direction, dtype=float))
        self.b_vec       /= np.linalg.norm(self.b_vec)
        self.theta_max    = theta_max
        self.beta_q       = beta_q
        self.P_bar        = P_bar
        self.Phi_o        = Phi_o
        self.mu_oh        = mu_overhang
        self.mu_th        = mu_thickness
        self.use_overhang = use_overhang
        self.use_thickness = use_thickness

        N              = len(pts_mm)
        self.N         = N
        # Uniform cell volumes (mesh is nearly isotropic; normalised for integrals)
        self.vol       = np.ones(N) / N

        # Build mesh operators once
        self.helmholtz = HelmholtzFilter(pts_mm, r_filter_mm, n_neighbors=6)
        if use_overhang:
            self.grad_op = MeshGradientOperator(pts_mm, n_neighbors=6)
        else:
            self.grad_op = None

    # ── Main entry: evaluate constraints + add penalty to (J, grad_J) ──────────

    def apply(
        self,
        gamma: np.ndarray,        # (N,) raw gamma from gyroid (1=fluid, 0=solid)
        J: float,                 # current objective value
        grad_J: np.ndarray,       # (N,) dJ/dgamma from OpenFOAM fsens
        iteration: int,
    ) -> tuple[float, np.ndarray, dict]:
        """
        1. Helmholtz-filter gamma → gamma_tilde.
        2. Compute overhang constraint on gamma_tilde.
        3. Compute thickness constraint on gamma_tilde (dilation/erosion of solid).
        4. Add quadratic penalty terms to J and grad_J.
        5. Return augmented (J_aug, grad_aug, info_dict).

        The penalty form is:
            J_aug = J + μ_oh · max(0, g_oh - P_bar)²
                      + μ_th · max(0, g_th - Φ_o)²
        """
        beta_proj = beta_schedule(iteration)

        # ── Filter ──────────────────────────────────────────────────────────────
        gamma_tilde = self.helmholtz.apply(gamma)

        info = {'beta_proj': beta_proj,
                'g_oh': 0.0, 'g_th': 0.0,
                'pen_oh': 0.0, 'pen_th': 0.0}

        # grad_J is ∂J/∂γ at each cell; we augment it with constraint sensitivities.
        # The filter maps γ → γ̃, so chain rule through adjoint:
        #   ∂(penalty)/∂γ = H⁻¹ ∂(penalty)/∂γ̃
        grad_aug = grad_J.copy()
        J_aug    = J

        # ── Overhang constraint ──────────────────────────────────────────────────
        if self.use_overhang:
            g_oh, dg_oh_dgt = compute_overhang_constraint(
                gamma_tilde, self.grad_op, self.vol,
                self.b_vec, self.theta_max, self.beta_q,
            )
            viol_oh = max(0.0, g_oh - self.P_bar)
            pen_oh  = self.mu_oh * viol_oh**2
            J_aug  += pen_oh
            info['g_oh']   = g_oh
            info['pen_oh'] = pen_oh

            if viol_oh > 0.0:
                # ∂pen_oh/∂γ̃ = 2 μ viol · ∂g_oh/∂γ̃
                dp_oh_dgt = 2.0 * self.mu_oh * viol_oh * dg_oh_dgt
                # Chain through Helmholtz filter: ∂/∂γ = H⁻¹ ∂/∂γ̃
                dp_oh_dg  = self.helmholtz.apply_adjoint(dp_oh_dgt)
                grad_aug += dp_oh_dg

        # ── Thickness constraint ─────────────────────────────────────────────────
        if self.use_thickness:
            g_th, dg_th_dgt = compute_thickness_constraint(
                gamma_tilde, self.vol, beta_proj,
            )
            viol_th = max(0.0, g_th - self.Phi_o)
            pen_th  = self.mu_th * viol_th**2
            J_aug  += pen_th
            info['g_th']   = g_th
            info['pen_th'] = pen_th

            if viol_th > 0.0:
                dp_th_dgt = 2.0 * self.mu_th * viol_th * dg_th_dgt
                dp_th_dg  = self.helmholtz.apply_adjoint(dp_th_dgt)
                grad_aug += dp_th_dg

        return J_aug, grad_aug, info

    def update_penalties(self, info: dict, violation_tol: float = 1e-3) -> None:
        """
        Increase penalty weights if constraints are still violated.
        Call once per outer iteration after `apply`.
        """
        if self.use_overhang and info['g_oh'] > self.P_bar + violation_tol:
            self.mu_oh = min(self.mu_oh * 2.0, 1e6)
        if self.use_thickness and info['g_th'] > self.Phi_o + violation_tol:
            self.mu_th = min(self.mu_th * 2.0, 1e6)
