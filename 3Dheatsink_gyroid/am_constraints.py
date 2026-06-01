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
        w, dw_dkx, dw_dky, dw_dkz,
        eps_reg
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
) -> tuple[float, np.ndarray, dict]:
    """
    Analytic gyroid overhang penalty and gradient.

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
    pen_j = w * viol**2
    J_oh  = mu_oh / N * float(np.sum(pen_j))

    w_sum = float(np.sum(w)) + 1e-30
    g_oh  = float(np.sum(w * np.maximum(0.0, -cos_max - n_b))) / w_sum

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
    c = mu_oh / N
    sk_x = c * (viol**2 * dw_dkx - 2.0 * w * viol * dnb_dkx)
    sk_y = c * (viol**2 * dw_dky - 2.0 * w * viol * dnb_dky)
    sk_z = c * (viol**2 * dw_dkz - 2.0 * w * viol * dnb_dkz)

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
) -> tuple[float, np.ndarray, dict]:
    """
    Raw surface-weighted mean overhang violation g_oh and its exact gradient,
    suitable for use as a proper inequality constraint in trust-constr optimisers.

    g_oh = Σ_j(w_j · max(0, −cos_max − n_b_j)) / Σ_j(w_j)
    """
    N = len(pts_mm)
    geo = _gyroid_geometry(pts_mm, freq_mm, gamma, epsilon, rot_matrix)

    bx, by, bz = b_vec
    Gx = geo['Gx']; Gy = geo['Gy']; Gz = geo['Gz']
    sG = geo['sG']; inv_ng = geo['inv_ng']

    b_dot_gradG = bx*Gx + by*Gy + bz*Gz
    n_b = sG * b_dot_gradG * inv_ng

    viol   = np.maximum(0.0, -cos_max - n_b)
    H_viol = (viol > 0.0).astype(float)
    w      = geo['w']

    w_sum = float(np.sum(w)) + 1e-30
    g_oh  = float(np.sum(w * viol)) / w_sum

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
    sk_x = c * ((viol - g_oh) * dw_dkx - w * H_viol * dnb_dkx)
    sk_y = c * ((viol - g_oh) * dw_dky - w * H_viol * dnb_dky)
    sk_z = c * ((viol - g_oh) * dw_dkz - w * H_viol * dnb_dkz)

    grad_ctrl = np.stack([W.T @ sk_x, W.T @ sk_y, W.T @ sk_z], axis=1)

    return g_oh, grad_ctrl, {'g_oh': g_oh}


# ── Parameter container ───────────────────────────────────────────────────────

class AMConstraints:
    """
    Stores overhang constraint parameters and the adaptive penalty weight.
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
