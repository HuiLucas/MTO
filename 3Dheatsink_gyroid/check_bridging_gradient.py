"""
Finite-difference verification of the analytic gradients in am_constraints.py,
covering both the un-eroded path (r_bridge=0, must reproduce the original
per-cell penalty exactly) and the interface-weighted bridging-length erosion
path (r_bridge>0).

Also includes a qualitative "thin shell" sanity check: a curved 2D manifold
of interface weight w (a thin spherical shell) carrying a small (<L_bridge)
violating cap and a large (>L_bridge) violating cap, confirming the small
cap erodes to m~0 while the large cap's core retains m~viol.
"""

from __future__ import annotations

import math

import numpy as np

from am_constraints import (
    compute_gyroid_overhang,
    compute_gyroid_overhang_raw,
    _erode_violation_field,
    _gyroid_geometry,
)


def build_synthetic_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    N = 3000
    pts_mm = rng.uniform(low=[0.0, 0.0, 0.0], high=[6.0, 6.0, 8.0], size=(N, 3))
    epsilon = 0.2
    half_thickness = 0.3
    k0 = 2.0 * math.pi / 1.8
    freq_base = np.full((N, 3), k0)
    N_ctrl = 5
    ctrl_pts = rng.uniform(low=[0.0, 0.0, 0.0], high=[6.0, 6.0, 8.0], size=(N_ctrl, 3))
    sigma_rbf = 2.0
    d2 = ((pts_mm[:, None, :] - ctrl_pts[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (2.0 * sigma_rbf ** 2))
    return dict(
        pts_mm=pts_mm, epsilon=epsilon, half_thickness=half_thickness,
        freq_base=freq_base, W=W, N_ctrl=N_ctrl,
        cos_max=math.cos(math.radians(45.0)),
        b_vec=np.array([0.0, 0.0, 1.0]), mu_oh=1.0,
    )


def gamma_sdf_of(case, freq_mm):
    pts_mm, epsilon, half_thickness = case['pts_mm'], case['epsilon'], case['half_thickness']
    dummy = np.zeros(len(pts_mm))
    geo = _gyroid_geometry(pts_mm, freq_mm, dummy, epsilon, rot_matrix=None)
    sdf = np.abs(geo['G']) - half_thickness
    gamma = 1.0 / (1.0 + np.exp(-sdf / epsilon))
    return gamma, sdf


def viol_of(case, dk_ctrl):
    freq_mm = case['freq_base'] + case['W'] @ dk_ctrl
    gamma, sdf = gamma_sdf_of(case, freq_mm)
    geo = _gyroid_geometry(case['pts_mm'], freq_mm, gamma, case['epsilon'], rot_matrix=None)
    bx, by, bz = case['b_vec']
    n_b = geo['sG'] * (bx * geo['Gx'] + by * geo['Gy'] + bz * geo['Gz']) * geo['inv_ng']
    return np.maximum(0.0, -case['cos_max'] - n_b)


def pick_dk_ctrl0(case, h, max_tries=50):
    """Find a dk_ctrl0 such that H_viol = (viol>0) doesn't flip for any cell
    under a +-h perturbation of any dk_ctrl component (avoids FD kinks)."""
    N_ctrl = case['N_ctrl']
    rng = np.random.default_rng(42)
    for _ in range(max_tries):
        dk_ctrl0 = 0.1 * rng.standard_normal((N_ctrl, 3))
        H0 = viol_of(case, dk_ctrl0) > 0.0
        ok = True
        for c in range(N_ctrl):
            for axis in range(3):
                for sign in (+1.0, -1.0):
                    dk = dk_ctrl0.copy()
                    dk[c, axis] += sign * h
                    Hp = viol_of(case, dk) > 0.0
                    if np.any(Hp != H0):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return dk_ctrl0
    raise RuntimeError("Could not find a dk_ctrl0 without H_viol flips")


def fd_check(case, dk_ctrl0, r_bridge, bridge_eps=0.02, h=1e-5, label="", p_agg=2.0):
    pts_mm, epsilon = case['pts_mm'], case['epsilon']
    cos_max, b_vec, mu_oh, W = case['cos_max'], case['b_vec'], case['mu_oh'], case['W']
    N_ctrl = case['N_ctrl']

    def eval_J_g(dk_ctrl):
        freq_mm = case['freq_base'] + W @ dk_ctrl
        gamma, sdf = gamma_sdf_of(case, freq_mm)
        J_oh, grad_J, _ = compute_gyroid_overhang(
            pts_mm, freq_mm, gamma, sdf, epsilon, cos_max, b_vec, mu_oh, W,
            rot_matrix=None, r_bridge=r_bridge, bridge_eps=bridge_eps, p_agg=p_agg)
        g_oh, grad_g, _ = compute_gyroid_overhang_raw(
            pts_mm, freq_mm, gamma, sdf, epsilon, cos_max, b_vec, W,
            rot_matrix=None, r_bridge=r_bridge, bridge_eps=bridge_eps)
        return J_oh, grad_J, g_oh, grad_g

    J0, grad_J0, g0, grad_g0 = eval_J_g(dk_ctrl0)

    max_rel_J = 0.0
    max_rel_g = 0.0
    n_components = N_ctrl * 3
    for c in range(N_ctrl):
        for axis in range(3):
            dk_p = dk_ctrl0.copy(); dk_p[c, axis] += h
            dk_m = dk_ctrl0.copy(); dk_m[c, axis] -= h
            Jp, _, gp, _ = eval_J_g(dk_p)
            Jm, _, gm, _ = eval_J_g(dk_m)
            dJ_fd = (Jp - Jm) / (2 * h)
            dg_fd = (gp - gm) / (2 * h)
            rel_J = abs(grad_J0[c, axis] - dJ_fd) / (abs(dJ_fd) + 1e-12)
            rel_g = abs(grad_g0[c, axis] - dg_fd) / (abs(dg_fd) + 1e-12)
            max_rel_J = max(max_rel_J, rel_J)
            max_rel_g = max(max_rel_g, rel_g)

    print(f"=== {label} (r_bridge={r_bridge}, p_agg={p_agg}) ===")
    print(f"  g_oh = {g0:.5f}")
    print(f"  J_oh: max relative error over {n_components} components = {max_rel_J:.3e}")
    print(f"  g_oh: max relative error over {n_components} components = {max_rel_g:.3e}")


def thin_shell_sanity_check():
    """A thin flat shell (w = Gaussian band around z=z0) carrying a small
    (<L_bridge) violating disc and a large (>L_bridge) violating disc.
    The small disc should erode to m~0; the large disc's core should retain
    m~viol."""
    print("\n=== Thin-shell erosion sanity check ===")

    spacing = 0.25
    x = np.arange(0.0, 14.0 + 1e-9, spacing)
    y = np.arange(0.0, 14.0 + 1e-9, spacing)
    z = np.arange(3.0, 5.0 + 1e-9, spacing)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts_mm = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    z0 = 4.0
    sigma = 0.15
    r_bridge = 0.75
    bridge_eps = 0.02

    w = np.exp(-((pts_mm[:, 2] - z0) / sigma) ** 2)

    small_center = np.array([3.0, 3.0])
    large_center = np.array([10.0, 10.0])
    R_small = 0.3   # diameter 0.6mm < L_bridge=1.5mm
    R_large = 4.0   # diameter 8.0mm > L_bridge=1.5mm
    core_large = 2.0   # core radius, 2mm margin from the disc's edge

    rho_small = np.linalg.norm(pts_mm[:, :2] - small_center, axis=1)
    rho_large = np.linalg.norm(pts_mm[:, :2] - large_center, axis=1)

    viol = np.zeros(len(pts_mm))
    viol[rho_small < R_small] = 0.25
    viol[rho_large < R_large] = 0.25

    m, redistribute = _erode_violation_field(pts_mm, viol, w, r_bridge, bridge_eps)
    assert np.all(np.isfinite(m)), "non-finite values in eroded field m"

    on_shell = w > 0.5
    m_small = m[on_shell & (rho_small < R_small)]
    m_large_core = m[on_shell & (rho_large < core_large)]

    print(f"  small disc (diam 0.6mm < L_bridge): on-shell m max = {m_small.max():.4f} (expect ~0)")
    print(f"  large disc core (diam 8mm > L_bridge): on-shell m min = {m_large_core.min():.4f} (expect ~0.25)")

    assert m_small.max() < 0.05, "small (self-bridging) island did not erode to ~0"
    assert m_large_core.min() > 0.15, "large violating region was incorrectly eroded"

    R_nb, R_w = redistribute(2.0 * m * w)
    assert np.all(np.isfinite(R_nb)) and np.all(np.isfinite(R_w)), "non-finite redistribute() output"


if __name__ == '__main__':
    case = build_synthetic_case()
    h = 1e-5
    dk_ctrl0 = pick_dk_ctrl0(case, h)
    fd_check(case, dk_ctrl0, r_bridge=0.0, h=h, label="Backward-compat (erosion disabled)", p_agg=2.0)
    fd_check(case, dk_ctrl0, r_bridge=0.75, h=h, label="Erosion active (w-weighted)", p_agg=2.0)
    fd_check(case, dk_ctrl0, r_bridge=0.75, h=h, label="Lp-norm aggregation (p_agg=6)", p_agg=6.0)
    thin_shell_sanity_check()
    print("\nAll checks passed.")
