#!/usr/bin/env python3
"""
gyroid_to_obj.py  –  Extract the gyroid wall surface as a quad-dominant OBJ mesh
                     using Dual Contouring on an adaptive octree grid.

The isosurface extracted is:

    |G(x,y,z)| - half_thickness = 0       (i.e. the solid/fluid interface)

where
    G  = sin(kx·x)·cos(ky·y) + sin(ky·y)·cos(kz·z) + sin(kz·z)·cos(kx·x)
    kx, ky, kz  =  k_base  +  RBF(dk_ctrl)(x, y, z)

Usage
-----
    python gyroid_to_obj.py \\
        [--ctrl  app/gyroid_ctrl_pts_checkpoint.txt] \\
        [--out   gyroid_surface.obj] \\
        [--unit  1.5]    # gyroid cell size in mm  → sets k_base
        [--wall  0.60]   # wall thickness in mm
        [--res   0.10]   # coarsest octree cell size in mm
        [--depth 4]      # max octree refinement depth
        [--xmin 0] [--xmax 4] [--ymin 0] [--ymax 2.5] [--zmin 0] [--zmax 10]
        [--config gyroid_case_config.yaml]

Dependencies
------------
    numpy, scipy  (pip install numpy scipy)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.interpolate import RBFInterpolator


# ── Gyroid rotation (mirrors gyroid_to_stl.py) ───────────────────────────────

def _gyroid_rotation_matrix(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a / np.linalg.norm(a)
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    D = a1**2 + a2**2
    if D < 1e-10:
        if a0 >= 0.0:
            return np.eye(3)
        return np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    return np.array([
        [a0,  a1,                    a2],
        [-a1, (a0*a1**2 + a2**2)/D,  (a0 - 1.0)*a1*a2/D],
        [-a2, (a0 - 1.0)*a1*a2/D,   (a0*a2**2 + a1**2)/D],
    ])


# ── RBF field ────────────────────────────────────────────────────────────────

class _BakedRBF:
    def __init__(self, ctrl_pts: np.ndarray, dk_ctrl: np.ndarray,
                 bbox_min: np.ndarray, bbox_max: np.ndarray,
                 bake_spacing: float = 0.3):
        self._bbox_min = bbox_min
        self._bbox_max = bbox_max
        rbfs = [RBFInterpolator(ctrl_pts, dk_ctrl[:, ax],
                                kernel='thin_plate_spline', degree=1)
                for ax in range(3)]
        axes = [np.arange(lo, hi + bake_spacing, bake_spacing)
                for lo, hi in zip(bbox_min, bbox_max)]
        BX, BY, BZ = np.meshgrid(*axes, indexing='ij')
        pts = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
        baked = np.column_stack([rbf(pts) for rbf in rbfs])
        nx, ny, nz = BX.shape
        self._grid = baked.reshape(nx, ny, nz, 3)
        self._axes = axes
        self._step = np.array([a[1] - a[0] if len(a) > 1 else 1.0 for a in axes])
        self._shape = (nx, ny, nz)

    def __call__(self, pts_mm: np.ndarray) -> np.ndarray:
        pts = np.clip(pts_mm, self._bbox_min, self._bbox_max)
        nx, ny, nz = self._shape
        gx = (pts[:, 0] - self._axes[0][0]) / self._step[0]
        gy = (pts[:, 1] - self._axes[1][0]) / self._step[1]
        gz = (pts[:, 2] - self._axes[2][0]) / self._step[2]
        ix = np.clip(gx.astype(int), 0, nx - 2)
        iy = np.clip(gy.astype(int), 0, ny - 2)
        iz = np.clip(gz.astype(int), 0, nz - 2)
        tx = (gx - ix)[:, None]; ty = (gy - iy)[:, None]; tz = (gz - iz)[:, None]
        g = self._grid
        return (g[ix,   iy,   iz  ] * (1-tx)*(1-ty)*(1-tz)
              + g[ix+1, iy,   iz  ] *    tx *(1-ty)*(1-tz)
              + g[ix,   iy+1, iz  ] * (1-tx)*   ty *(1-tz)
              + g[ix+1, iy+1, iz  ] *    tx *   ty *(1-tz)
              + g[ix,   iy,   iz+1] * (1-tx)*(1-ty)*   tz
              + g[ix+1, iy,   iz+1] *    tx *(1-ty)*   tz
              + g[ix,   iy+1, iz+1] * (1-tx)*   ty *   tz
              + g[ix+1, iy+1, iz+1] *    tx *   ty *   tz)


# ── Gyroid scalar field ───────────────────────────────────────────────────────

def gyroid_G(pts_mm: np.ndarray, k_base: float,
             rbf_field: _BakedRBF | None,
             rot_matrix: np.ndarray | None = None) -> np.ndarray:
    dk = rbf_field(pts_mm) if rbf_field is not None else np.zeros((len(pts_mm), 3))
    x = pts_mm[:, 0]; y = pts_mm[:, 1]; z = pts_mm[:, 2]
    kx = k_base + dk[:, 0]; ky = k_base + dk[:, 1]; kz = k_base + dk[:, 2]
    if rot_matrix is not None:
        R = rot_matrix
        p = kx*x; q = ky*y; r = kz*z
        u = R[0,0]*p + R[0,1]*q + R[0,2]*r
        v = R[1,0]*p + R[1,1]*q + R[1,2]*r
        w = R[2,0]*p + R[2,1]*q + R[2,2]*r
        return np.cos(u)*np.cos(v) + np.sin(v)*np.cos(w) - np.sin(w)*np.sin(u)
    return (np.sin(kx*x)*np.cos(ky*y)
          + np.sin(ky*y)*np.cos(kz*z)
          + np.sin(kz*z)*np.cos(kx*x))


def sdf_gyroid(pts_mm: np.ndarray, k_base: float, half_t: float,
               rbf_field: _BakedRBF | None,
               rot_matrix: np.ndarray | None = None) -> np.ndarray:
    """Evaluate |G| - half_thickness (both wall surfaces at once)."""
    G = gyroid_G(pts_mm, k_base, rbf_field, rot_matrix)
    return np.abs(G) - half_t


def sdf_sheet(pts_mm: np.ndarray, k_base: float, half_t: float,
              rbf_field: _BakedRBF | None, rot_matrix: np.ndarray | None,
              sheet: int) -> np.ndarray:
    """
    SDF for one wall surface, with consistent inside-negative orientation.
      sheet=+1 : F = G  - half_t   (zero set: G = +half_t)
      sheet=-1 : F = -G - half_t   (zero set: G = -half_t)
    Both are negative inside the solid wall and positive outside,
    so sign-change detection and normal computation work identically.
    """
    G = gyroid_G(pts_mm, k_base, rbf_field, rot_matrix)
    return (G - half_t) if sheet > 0 else (-G - half_t)


# ── QEF solver (Tikhonov-regularized) ────────────────────────────────────────

def solve_qef(intersections: np.ndarray, normals: np.ndarray,
              cell_min: np.ndarray, cell_max: np.ndarray,
              lam: float = 1e-2) -> np.ndarray:
    """
    Solve the regularized QEF:
        min  Σ (n_i · (x - p_i))²  +  λ ||x - x_c||²
    where x_c is the cell centroid (Tikhonov term pulls jittery saddle-region
    vertices toward the cell center instead of letting them fly).

    Falls back to the centroid when fewer than 3 constraints exist.
    Result is clamped to the cell bounds.
    """
    centroid = 0.5 * (cell_min + cell_max)
    if len(intersections) < 3:
        return centroid

    A = normals                                          # (K, 3)
    b = np.einsum('ki,ki->k', normals, intersections)   # (K,)

    # Augment with Tikhonov rows: sqrt(λ)·I and sqrt(λ)·centroid
    sq_lam = math.sqrt(lam)
    A_aug = np.vstack([A, sq_lam * np.eye(3)])
    b_aug = np.concatenate([b, sq_lam * centroid])

    try:
        x, _, rank, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    except np.linalg.LinAlgError:
        return centroid

    if rank < 1:
        return centroid

    return np.clip(x, cell_min, cell_max)


def _gradient_fd(pt: np.ndarray, k_base: float, half_t: float,
                 rbf_field, rot_matrix, h: float, sheet: int) -> np.ndarray:
    """
    Central-difference gradient of sdf_sheet at a single point.
    h should be 0.05–0.10 × min(cell_size) — never a fixed absolute value.
    Returns the unnormalized gradient vector.
    """
    p = pt[np.newaxis, :]
    dx = sdf_sheet(p + [[h, 0, 0]], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
    mx = sdf_sheet(p - [[h, 0, 0]], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
    dy = sdf_sheet(p + [[0, h, 0]], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
    my = sdf_sheet(p - [[0, h, 0]], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
    dz = sdf_sheet(p + [[0, 0, h]], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
    mz = sdf_sheet(p - [[0, 0, h]], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
    return np.array([(dx - mx), (dy - my), (dz - mz)]) / (2.0 * h)


def _normalized_gradient(pt: np.ndarray, k_base: float, half_t: float,
                         rbf_field, rot_matrix, h: float, sheet: int) -> np.ndarray:
    """Normalized gradient (surface normal direction) at pt."""
    g = _gradient_fd(pt, k_base, half_t, rbf_field, rot_matrix, h, sheet)
    n = np.linalg.norm(g)
    return g / n if n > 1e-12 else g


# ── Newton projection — snap a point onto F=0 ────────────────────────────────

def _newton_project(pt: np.ndarray, k_base: float, half_t: float,
                    rbf_field, rot_matrix, h: float, sheet: int,
                    cell_min: np.ndarray, cell_max: np.ndarray,
                    max_iters: int = 5, tol: float = 1e-7) -> np.ndarray:
    """
    Newton iteration:  x ← x − F(x)/|∇F(x)|² · ∇F(x)
    Snaps an approximate QEF vertex onto the exact zero level set.
    Result is clamped to cell bounds so it cannot escape.
    """
    x = pt.copy()
    for _ in range(max_iters):
        f = sdf_sheet(x[np.newaxis], k_base, half_t, rbf_field, rot_matrix, sheet)[0]
        if abs(f) < tol:
            break
        g = _gradient_fd(x, k_base, half_t, rbf_field, rot_matrix, h, sheet)
        gg = float(np.dot(g, g))
        if gg < 1e-24:
            break
        x = x - (f / gg) * g
    return np.clip(x, cell_min, cell_max)


# ── Octree ────────────────────────────────────────────────────────────────────

# 8 corner offsets for a unit cube: [0,1]^3 in (i,j,k) order
_CORNER_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=float)

# 12 edges as pairs of corner indices
_CELL_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),  # along x
    (0, 2), (1, 3), (4, 6), (5, 7),  # along y
    (0, 4), (1, 5), (2, 6), (3, 7),  # along z
]


class OctreeCell(NamedTuple):
    """Leaf cell of the octree carrying its DC vertex."""
    min_pt: np.ndarray   # (3,) lower-left corner in mm
    max_pt: np.ndarray   # (3,) upper-right corner in mm
    vertex: np.ndarray   # (3,) DC vertex position (QEF solution)
    vertex_id: int       # index into final vertex list


def build_octree_leaves(
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    k_base: float,
    half_t: float,
    rbf_field,
    rot_matrix,
    coarse_res: float,
    max_depth: int,
    sheet: int,
) -> list[OctreeCell]:
    """
    Build adaptive octree leaf cells that straddle the isosurface F=0
    of the given wall surface sheet (+1 or -1).

    Strategy:
      1. Seed with uniform coarse cells of size coarse_res.
      2. For each active (sign-changing) coarse cell, refine recursively
         up to max_depth levels (cell size halves each level).
      3. Return all leaf cells that are still active after refinement.
    """
    sheet_label = '+' if sheet > 0 else '-'
    print(f"  Building octree for G={sheet_label}half_t surface "
          f"(coarse={coarse_res:.3f} mm, depth={max_depth}) …")

    # Pre-evaluate SDF at coarse grid nodes
    xs = np.arange(domain_min[0], domain_max[0] + coarse_res * 0.5, coarse_res)
    ys = np.arange(domain_min[1], domain_max[1] + coarse_res * 0.5, coarse_res)
    zs = np.arange(domain_min[2], domain_max[2] + coarse_res * 0.5, coarse_res)
    xs = np.clip(xs, domain_min[0], domain_max[0])
    ys = np.clip(ys, domain_min[1], domain_max[1])
    zs = np.clip(zs, domain_min[2], domain_max[2])

    XG, YG, ZG = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.column_stack([XG.ravel(), YG.ravel(), ZG.ravel()])
    print(f"  Evaluating SDF at {len(pts):,} coarse nodes …")
    F_nodes = sdf_sheet(pts, k_base, half_t, rbf_field, rot_matrix, sheet).reshape(
        len(xs), len(ys), len(zs))

    # Traverse coarse cells, recursively refine active ones
    leaves: list[OctreeCell] = []
    vertex_counter = [0]
    total_coarse = (len(xs)-1) * (len(ys)-1) * (len(zs)-1)
    processed = [0]

    def _refine(cell_min: np.ndarray, cell_max: np.ndarray,
                corner_vals: np.ndarray, depth: int) -> None:
        """Recursively refine; corner_vals = SDF at the 8 corners."""
        active = (corner_vals.min() <= 0.0) and (corner_vals.max() >= 0.0)
        if not active:
            return

        if depth >= max_depth:
            # Leaf: compute DC vertex via QEF then project to F=0
            intersections, normals = [], []
            size = cell_max - cell_min
            # Step 3: h scaled to local cell size (5% of smallest dimension)
            h = 0.05 * float(size.min())
            for ci, cj in _CELL_EDGES:
                f0 = corner_vals[ci]; f1 = corner_vals[cj]
                if (f0 <= 0.0) == (f1 <= 0.0):
                    continue
                t = f0 / (f0 - f1)
                p0 = cell_min + _CORNER_OFFSETS[ci] * size
                p1 = cell_min + _CORNER_OFFSETS[cj] * size
                pt = p0 + t * (p1 - p0)
                # Step 3: use cell-relative h for edge-intersection normals
                n = _normalized_gradient(pt, k_base, half_t, rbf_field, rot_matrix, h, sheet)
                intersections.append(pt)
                normals.append(n)

            if not intersections:
                return

            ints_arr = np.array(intersections)
            norm_arr = np.array(normals)
            # Step 2: regularized QEF
            vtx = solve_qef(ints_arr, norm_arr, cell_min, cell_max)
            # Step 1: Newton-project the QEF vertex onto the exact zero level set
            vtx = _newton_project(vtx, k_base, half_t, rbf_field, rot_matrix,
                                  h, sheet, cell_min, cell_max)

            vid = vertex_counter[0]
            vertex_counter[0] += 1
            leaves.append(OctreeCell(cell_min.copy(), cell_max.copy(), vtx, vid))
            return

        # Subdivide into 8 children
        mid = 0.5 * (cell_min + cell_max)
        size = cell_max - cell_min
        child_size = 0.5 * size

        # Evaluate SDF at the 19 new sub-nodes (face, edge, body centres)
        # For simplicity, evaluate all 8 corners of each child from scratch
        for ci in range(2):
            for cj in range(2):
                for ck in range(2):
                    ch_min = cell_min + np.array([ci, cj, ck]) * child_size
                    ch_max = ch_min + child_size
                    corners = ch_min[np.newaxis, :] + _CORNER_OFFSETS * child_size
                    f_vals = sdf_sheet(corners, k_base, half_t, rbf_field, rot_matrix, sheet)
                    _refine(ch_min, ch_max, f_vals, depth + 1)

    # Seed from coarse grid
    nx, ny, nz = len(xs)-1, len(ys)-1, len(zs)-1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cv = F_nodes[i:i+2, j:j+2, k:k+2].ravel()[[0,1,2,3,4,5,6,7]]
                # Reorder to match _CORNER_OFFSETS: (i,j,k) corner layout
                cv = np.array([
                    F_nodes[i,   j,   k],
                    F_nodes[i+1, j,   k],
                    F_nodes[i,   j+1, k],
                    F_nodes[i+1, j+1, k],
                    F_nodes[i,   j,   k+1],
                    F_nodes[i+1, j,   k+1],
                    F_nodes[i,   j+1, k+1],
                    F_nodes[i+1, j+1, k+1],
                ])
                if (cv.min() > 0.0) or (cv.max() < 0.0):
                    continue
                c_min = np.array([xs[i], ys[j], zs[k]])
                c_max = np.array([xs[i+1], ys[j+1], zs[k+1]])
                _refine(c_min, c_max, cv, 0)
                processed[0] += 1
                if processed[0] % 500 == 0:
                    print(f"  Octree: processed {processed[0]:,}/{total_coarse:,} coarse cells, "
                          f"leaves so far: {len(leaves):,}", end='\r')

    print(f"  Octree done: {len(leaves):,} leaf cells                              ")
    return leaves


# ── Quad connectivity via spatial lookup ─────────────────────────────────────

def build_quads(leaves: list[OctreeCell]) -> tuple[np.ndarray, list[list[int]]]:
    """
    For each axis-aligned face shared by two adjacent leaf cells,
    find the four cells that share that face edge and emit a quad.

    For a uniform octree the rule is: two cells share an x-face when
    cell A's x_max == cell B's x_min and their y,z extents overlap.

    We use a tolerance-based spatial hash on face centres.

    Returns:
        vertices  – (V, 3) float64 array of DC vertex positions
        quads     – list of [v0, v1, v2, v3] index lists
    """
    if not leaves:
        return np.empty((0, 3)), []

    vertices = np.array([cell.vertex for cell in leaves])

    # Build lookup: centre of each face → list of (leaf_index, face_axis)
    # For each leaf and each of its 6 faces, key = (axis, face_coord, mid_other_a, mid_other_b)
    # rounded to a tolerance based on leaf size.

    # Determine typical leaf size for tolerance
    sizes = np.array([cell.max_pt - cell.min_pt for cell in leaves])
    tol = sizes.min() * 0.01   # 1% of smallest cell

    def _round(v: float) -> int:
        return int(round(v / tol))

    # face_map: key → list of leaf indices (cells sharing that face)
    # Key encodes axis + face position: for axis=0 (x-face), key=(0, x_val, y_mid, z_mid)
    face_map: dict[tuple, list[int]] = {}

    for idx, cell in enumerate(leaves):
        cx = 0.5 * (cell.min_pt + cell.max_pt)
        for axis in range(3):
            for side in (cell.min_pt[axis], cell.max_pt[axis]):
                other = [i for i in range(3) if i != axis]
                key = (axis,
                       _round(side),
                       _round(cx[other[0]]),
                       _round(cx[other[1]]))
                face_map.setdefault(key, []).append(idx)

    # For each face with exactly 2 cells: find the 4 cells around the shared edge.
    # For uniform same-size cells, 2 cells sharing a face → directly make a quad
    # from those 2 vertices + the 2 neighbours along the face's other two axes.
    #
    # More robustly: collect all faces shared by exactly 2 cells, then
    # for each pair (A, B) sharing an x-face, find all pairs (C, D) sharing the
    # same x-face at same x_val and adjacent y/z mid so that A,B,C,D form a ring.
    #
    # Simpler and correct for the uniform-at-leaf-level case:
    # For each axis, group pairs by (axis, face_val, other_mid_a, other_mid_b).
    # Four cells that tile a face in a 2×2 pattern form one quad.

    quads: list[list[int]] = []
    seen_quad_keys: set[frozenset] = set()

    # For each axis, find the 2×2 blocks of adjacent cells
    # Group leaves by the cell size at their level (cells of same size can form quads)
    # We group cells by their half-extent along each transverse axis

    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        # Map each leaf to its face-normal position and transverse grid coords
        # key2: (axis, face_val_rounded, other0_min_rounded, other1_min_rounded)
        face2: dict[tuple, list[int]] = {}
        for idx, cell in enumerate(leaves):
            # Use max_pt face (the "right" face along axis)
            fv = _round(cell.max_pt[axis])
            o0_min = _round(cell.min_pt[other[0]])
            o1_min = _round(cell.min_pt[other[1]])
            key2 = (axis, fv, o0_min, o1_min)
            face2.setdefault(key2, []).append(idx)

        # For each right face, find the matching left face of adjacent cell
        for (ax, fv, o0_min, o1_min), right_ids in face2.items():
            # The cell(s) whose left face matches this right face
            left_key = (ax, fv, o0_min, o1_min)
            # We need to find 4 cells that share the face:
            # Right cells at (o0_min, o1_min) and (o0_min+s, o1_min),
            # (o0_min, o1_min+s), (o0_min+s, o1_min+s)
            # where s = cell size in other directions

            for r_idx in right_ids:
                cell_r = leaves[r_idx]
                s0 = _round(cell_r.max_pt[other[0]] - cell_r.min_pt[other[0]])
                s1 = _round(cell_r.max_pt[other[1]] - cell_r.min_pt[other[1]])

                # The 4 right-face cells that tile a quad
                quad_right_keys = [
                    (ax, fv, o0_min,      o1_min),
                    (ax, fv, o0_min + s0, o1_min),
                    (ax, fv, o0_min,      o1_min + s1),
                    (ax, fv, o0_min + s0, o1_min + s1),
                ]
                quad_right_ids = []
                for qk in quad_right_keys:
                    ids = face2.get(qk, [])
                    if len(ids) == 1:
                        quad_right_ids.append(ids[0])
                    else:
                        break
                else:
                    if len(quad_right_ids) == 4:
                        key_set = frozenset(quad_right_ids)
                        if key_set not in seen_quad_keys:
                            seen_quad_keys.add(key_set)
                            # Consistent ordering: CCW when viewed from +axis
                            # Order: (o0_min,o1_min), (o0+s,o1_min), (o0+s,o1+s), (o0,o1+s)
                            ids_ordered = [
                                face2[(ax, fv, o0_min,      o1_min)][0],
                                face2[(ax, fv, o0_min + s0, o1_min)][0],
                                face2[(ax, fv, o0_min + s0, o1_min + s1)][0],
                                face2[(ax, fv, o0_min,      o1_min + s1)][0],
                            ]
                            quads.append(ids_ordered)

    print(f"  Quads constructed: {len(quads):,}")
    return vertices, quads


# ── Tangential Laplacian smoothing ───────────────────────────────────────────

def tangential_smooth(
    vertices: np.ndarray,
    quads: list[list[int]],
    k_base: float,
    half_t: float,
    rbf_field,
    rot_matrix,
    sheet: int,
    n_iters: int = 10,
    alpha: float = 0.3,
    h_scale: float = 0.05,
    proj_tol: float = 1e-7,
) -> np.ndarray:
    """
    Tangential Laplacian smoothing (Step 4).

    Each iteration:
      1. Compute the Laplacian displacement d = avg(neighbours) - v.
      2. Project d onto the surface tangent plane: d_t = d - (d·n)·n.
      3. Move v += alpha * d_t.
      4. Newton-project back onto F=0.

    Normals and the reprojection step ensure vertices stay on the surface
    and sharpness is preserved — no smoothing happens in the normal direction.

    h_scale: gradient step = h_scale * estimated local spacing (mean edge length).
    """
    if n_iters == 0:
        return vertices

    verts = vertices.copy()
    n_verts = len(verts)

    # Build neighbour list from quad connectivity
    neighbours: list[set[int]] = [set() for _ in range(n_verts)]
    for q in quads:
        for a in range(4):
            b = (a + 1) % 4
            neighbours[q[a]].add(q[b])
            neighbours[q[b]].add(q[a])
    nb_lists = [list(s) for s in neighbours]

    # Estimate a global h from mean quad edge length
    edge_sum = 0.0
    edge_count = 0
    for q in quads[:min(500, len(quads))]:
        for a in range(4):
            edge_sum += float(np.linalg.norm(verts[q[(a+1)%4]] - verts[q[a]]))
            edge_count += 1
    h_global = h_scale * (edge_sum / max(edge_count, 1))
    h_global = max(h_global, 1e-6)

    print(f"  Tangential smoothing: {n_iters} iters, α={alpha}, h={h_global:.4g} mm …")
    for it in range(n_iters):
        new_verts = verts.copy()
        for vi in range(n_verts):
            nbs = nb_lists[vi]
            if not nbs:
                continue
            avg = np.mean(verts[nbs], axis=0)
            d = avg - verts[vi]
            # Surface normal at current vertex
            n = _normalized_gradient(verts[vi], k_base, half_t,
                                     rbf_field, rot_matrix, h_global, sheet)
            # Tangential component only
            d_t = d - float(np.dot(d, n)) * n
            new_verts[vi] = verts[vi] + alpha * d_t

        # Newton re-project every vertex back onto F=0
        # Use a dummy cell bound of ±∞ so projection isn't clamped
        _inf = np.full(3,  1e18)
        _ninf = np.full(3, -1e18)
        for vi in range(n_verts):
            new_verts[vi] = _newton_project(
                new_verts[vi], k_base, half_t, rbf_field, rot_matrix,
                h_global, sheet, _ninf, _inf, max_iters=5, tol=proj_tol)

        verts = new_verts
        if (it + 1) % 5 == 0 or it == n_iters - 1:
            print(f"    iter {it+1}/{n_iters} done")

    return verts


# ── OBJ writer ────────────────────────────────────────────────────────────────

def write_obj(path: Path, vertices: np.ndarray, quads: list[list[int]]) -> None:
    """Write quad mesh to Wavefront OBJ (1-indexed, f lines with 4 vertices)."""
    n_verts = len(vertices)
    n_quads = len(quads)
    with open(path, 'w') as fh:
        fh.write(f"# Gyroid quad mesh  –  {n_verts} vertices  {n_quads} quads\n")
        fh.write("# Generated by gyroid_to_obj.py\n\n")
        for v in vertices:
            fh.write(f"v {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n")
        fh.write("\n")
        for q in quads:
            # OBJ uses 1-based indices
            fh.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Wrote {n_verts:,} vertices, {n_quads:,} quads  ({size_mb:.1f} MB)  →  {path}")


# ── YAML config reader ────────────────────────────────────────────────────────

def read_yaml_params(yaml_path: Path) -> dict:
    """Read gyroid_case_config.yaml for OBJ export parameters."""
    params = {}
    try:
        import yaml
        with open(yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        opt = cfg.get('optimization', {})
        params['unit']   = float(opt.get('unit', 1.5))
        params['wall']   = float(opt.get('wall', 0.30))
        params['kbound'] = float(opt.get('kbound', 2.0))
        geo = cfg.get('geometry', {})
        origin = geo.get('origin_mm', [0.0, 0.0, 0.0])
        size   = geo.get('size_mm',   [4.0, 2.5, 10.0])
        params['xmin'] = float(origin[0])
        params['ymin'] = float(origin[1])
        params['zmin'] = float(origin[2])
        params['xmax'] = float(origin[0]) + float(size[0])
        params['ymax'] = float(origin[1]) + float(size[1])
        params['zmax'] = float(origin[2]) + float(size[2])

        flow_axis = str(geo.get('flow_axis', 'z')).lower()
        axis_map  = {'x': 0, 'y': 1, 'z': 2}
        flow_idx  = axis_map.get(flow_axis, 2)
        transverse = [i for i in range(3) if i != flow_idx]

        inlet_cfg  = cfg.get('inlet',  {})
        outlet_cfg = cfg.get('outlet', {})
        inlet_face  = str(inlet_cfg.get('face',  'min')).lower()
        outlet_face = str(outlet_cfg.get('face', 'max')).lower()
        inlet_orig_2d  = inlet_cfg.get('window_origin_mm',  [0.0, 0.0])
        outlet_orig_2d = outlet_cfg.get('window_origin_mm', [0.0, 0.0])

        inlet_3d  = list(origin)
        outlet_3d = list(origin)
        inlet_3d[flow_idx]  = float(origin[flow_idx]) if inlet_face  == 'min' else float(origin[flow_idx]) + float(size[flow_idx])
        outlet_3d[flow_idx] = float(origin[flow_idx]) if outlet_face == 'min' else float(origin[flow_idx]) + float(size[flow_idx])
        for local_i, ax_i in enumerate(transverse):
            inlet_3d[ax_i]  = float(origin[ax_i]) + float(inlet_orig_2d[local_i])
            outlet_3d[ax_i] = float(origin[ax_i]) + float(outlet_orig_2d[local_i])

        direction = np.array(outlet_3d, dtype=float) - np.array(inlet_3d, dtype=float)
        norm = float(np.linalg.norm(direction))
        params['gyroid_rot_vec'] = (direction / norm).tolist() if norm > 1e-10 else None

    except ImportError:
        text = yaml_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            for key in ('unit', 'wall'):
                if line.startswith(key + ':'):
                    try:
                        params[key] = float(line.split(':')[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass
    return params


# ── Boundary loop closing ─────────────────────────────────────────────────────

#  Domain face descriptors: (name, fixed_axis, (other_axes), outward_sign)
_CAP_FACES = [
    ('xmin', 0, (1, 2), -1),
    ('xmax', 0, (1, 2), +1),
    ('ymin', 1, (0, 2), -1),
    ('ymax', 1, (0, 2), +1),
    ('zmin', 2, (0, 1), -1),
    ('zmax', 2, (0, 1), +1),
]
_FACE_INFO = {fd[0]: fd[1:] for fd in _CAP_FACES}  # fname → (faxis, (a0,a1), fsign)


def find_boundary_loops(quads: list[list[int]]) -> list[list[int]]:
    """
    Return all connected chains of boundary edges (edges shared by exactly 1 quad).
    Each chain contains at least 3 vertex indices.  A chain whose last vertex is
    adjacent to its first in the boundary graph is a closed loop; otherwise it is
    an open chain (indicating a mesh defect).
    """
    edge_count: dict[tuple[int,int], int] = {}
    for q in quads:
        for a in range(4):
            qa, qb = q[a], q[(a + 1) % 4]
            e = (min(qa, qb), max(qa, qb))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = {e for e, c in edge_count.items() if c == 1}

    adj: dict[int, list[int]] = {}
    for e in boundary_edges:
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])

    visited: set[int] = set()
    loops:   list[list[int]] = []
    for start in adj:
        if start in visited:
            continue
        chain: list[int] = [start]
        visited.add(start)
        cur = start
        while True:
            nexts = [nb for nb in adj.get(cur, []) if nb not in visited]
            if not nexts:
                break
            nxt = nexts[0]
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(chain) >= 3:
            loops.append(chain)
    return loops


def close_open_loops(
    vertices: np.ndarray,
    quads: list[list[int]],
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    k_base: float,
    half_t: float,
    rbf_field,
    rot_matrix,
    cap_res: float,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Find every open boundary loop on the mesh and close it.

    Face loops (loop centroid within 2*cap_res of a domain face):
      • Snap all loop vertices to lie exactly on the face plane.
      • Fill the solid cross-section on that face (|G| < half_t) with a regular
        grid of flat quads, using the snapped loop vertices as anchor nodes so
        the cap shares vertices with the mesh boundary.

    Interior loops (not near any face):
      • Close with a centroid + edge-midpoint fan, producing N all-quad patches.

    All 6 faces are processed; --open-faces is not considered here.
    """
    all_verts: list[np.ndarray] = list(vertices)
    all_quads: list[list[int]]  = list(quads)

    loops = find_boundary_loops(quads)
    print(f"  Found {len(loops):,} open boundary loops")

    face_tol = cap_res * 2.0

    # ── Classify loops ────────────────────────────────────────────────────────
    # Group by nearest face; unclassified → interior
    face_groups: dict[str, list[list[int]]] = {}
    interior_loops: list[list[int]] = []

    for loop in loops:
        lv = np.array([all_verts[i] for i in loop])
        best_name, best_d = None, float('inf')
        for fname, faxis, _, fsign in _CAP_FACES:
            fval = float(domain_min[faxis] if fsign < 0 else domain_max[faxis])
            d = float(np.abs(lv[:, faxis] - fval).mean())
            if d < face_tol and d < best_d:
                best_d, best_name = d, fname
        if best_name is not None:
            face_groups.setdefault(best_name, []).append(loop)
        else:
            interior_loops.append(loop)

    n_face_loops = sum(len(v) for v in face_groups.values())
    print(f"  Classification: {n_face_loops:,} face loops "
          f"({len(face_groups)} faces), {len(interior_loops):,} interior loops")

    # ── Close face loops (per face, all loops together) ───────────────────────
    for fname, face_loops in face_groups.items():
        faxis, (a0, a1), normal_sign = _FACE_INFO[fname]
        fval = float(domain_min[faxis] if normal_sign < 0 else domain_max[faxis])

        # Step 1: snap loop vertices to face plane
        for loop in face_loops:
            for idx in loop:
                v = all_verts[idx].copy()
                v[faxis] = fval
                all_verts[idx] = v

        # Step 2: build 2-D grid on this face, evaluate |G| - half_t
        lim0 = (float(domain_min[a0]), float(domain_max[a0]))
        lim1 = (float(domain_min[a1]), float(domain_max[a1]))
        ns0 = max(3, int(round((lim0[1] - lim0[0]) / cap_res)) + 1)
        ns1 = max(3, int(round((lim1[1] - lim1[0]) / cap_res)) + 1)
        s0  = np.linspace(lim0[0], lim0[1], ns0)
        s1  = np.linspace(lim1[0], lim1[1], ns1)

        S0, S1 = np.meshgrid(s0, s1, indexing='ij')
        pts = np.zeros((ns0 * ns1, 3))
        pts[:, faxis] = fval
        pts[:, a0]    = S0.ravel()
        pts[:, a1]    = S1.ravel()

        G_vals = gyroid_G(pts, k_base, rbf_field, rot_matrix)
        F_grid = (np.abs(G_vals) - half_t).reshape(ns0, ns1)
        pts_grid = pts.reshape(ns0, ns1, 3)

        # Step 3: emit flat cap quads using independent grid vertices.
        # We do NOT snap grid nodes to loop vertices: doing so would cause
        # non-manifold edges where a shared cap–sheet edge appears 3× (once in
        # the sheet interior, twice from adjacent cap cells).  Instead, the
        # snapped loop and the cap grid are coplanar; the boolean merge handles
        # the combination.
        cap_base   = len(all_verts)
        node_local: dict[tuple[int,int], int] = {}

        def _vid(i: int, j: int) -> int:
            if (i, j) not in node_local:
                node_local[(i, j)] = cap_base + len(node_local)
                all_verts.append(pts_grid[i, j].copy())
            return node_local[(i, j)]

        n_cap = 0
        for i in range(ns0 - 1):
            for j in range(ns1 - 1):
                f_avg = 0.25*(F_grid[i,j]+F_grid[i+1,j]+F_grid[i+1,j+1]+F_grid[i,j+1])
                if f_avg >= 0.0:
                    continue
                v00=_vid(i,j); v10=_vid(i+1,j); v11=_vid(i+1,j+1); v01=_vid(i,j+1)
                if normal_sign > 0:
                    all_quads.append([v00, v10, v11, v01])
                else:
                    all_quads.append([v00, v01, v11, v10])
                n_cap += 1

        n_seam = sum(len(l) for l in face_loops)
        print(f"  Face {fname:4s}: {len(face_loops):,} loops, "
              f"{n_seam:,} snapped verts, {n_cap:,} cap quads")

    # ── Close interior loops with centroid + midpoint all-quad fan ────────────
    for loop in interior_loops:
        N = len(loop)
        c_pos = np.mean([all_verts[i] for i in loop], axis=0)
        c_idx = len(all_verts);  all_verts.append(c_pos)

        mid_ids: list[int] = []
        for i in range(N):
            m = 0.5 * (all_verts[loop[i]] + all_verts[loop[(i + 1) % N]])
            mid_ids.append(len(all_verts));  all_verts.append(m)

        for i in range(N):
            all_quads.append([loop[i], mid_ids[i], c_idx, mid_ids[(i - 1) % N]])

    if interior_loops:
        print(f"  Interior: {len(interior_loops):,} loops closed with centroid fan")

    return np.array(all_verts), all_quads



# ── Non-manifold edge removal ─────────────────────────────────────────────────

def remove_nonmanifold_quads(
    vertices: np.ndarray,
    quads: list[list[int]],
    k_base: float,
    half_t: float,
    rbf_field,
    rot_matrix,
    sheet: int,
    max_passes: int = 20,
) -> list[list[int]]:
    """
    Iteratively remove quads that cause non-manifold edges (edges shared by 3+
    faces).  For each such edge, the quad whose face normal is least aligned
    with the SDF gradient at its centroid is discarded.  Repeats until no
    non-manifold edges remain or max_passes is reached.

    Face normal: cross product of the quad diagonals (v2-v0) × (v3-v1).
    SDF normal : _normalized_gradient at the quad centroid.
    Alignment  : dot(face_normal, sdf_normal)  — worst (smallest) is removed.
    """
    if not quads:
        return quads

    # h for gradient: 5% of mean edge length across a sample of quads
    sample = quads[:min(200, len(quads))]
    edge_lengths = [
        float(np.linalg.norm(vertices[q[(a + 1) % 4]] - vertices[q[a]]))
        for q in sample for a in range(4)
    ]
    h = max(0.05 * float(np.mean(edge_lengths)), 1e-5)

    quads = list(quads)       # work on a mutable copy
    total_removed = 0

    for pass_idx in range(max_passes):
        # Build edge (sorted vertex pair) → list of quad indices
        edge_map: dict[tuple[int, int], list[int]] = {}
        for qi, q in enumerate(quads):
            for a in range(4):
                e = (min(q[a], q[(a + 1) % 4]), max(q[a], q[(a + 1) % 4]))
                edge_map.setdefault(e, []).append(qi)

        bad_edges = {e: qis for e, qis in edge_map.items() if len(qis) > 2}
        if not bad_edges:
            break

        # For each bad edge pick the quad with the worst normal alignment
        to_remove: set[int] = set()
        for qis in bad_edges.values():
            worst_qi, worst_score = -1, 2.0   # best possible dot = 1.0
            for qi in qis:
                q = quads[qi]
                v = vertices[q]
                centroid = v.mean(axis=0)
                d1 = v[2] - v[0]
                d2 = v[3] - v[1]
                fn = np.cross(d1, d2)
                fn_n = float(np.linalg.norm(fn))
                fn = fn / fn_n if fn_n > 1e-12 else fn
                sdf_n = _normalized_gradient(
                    centroid, k_base, half_t, rbf_field, rot_matrix, h, sheet)
                alignment = float(np.dot(fn, sdf_n))
                # keep track of worst (lowest dot product)
                if alignment < worst_score:
                    worst_score, worst_qi = alignment, qi
            to_remove.add(worst_qi)

        quads = [q for qi, q in enumerate(quads) if qi not in to_remove]
        total_removed += len(to_remove)
        print(f"    non-manifold pass {pass_idx + 1}: "
              f"removed {len(to_remove):,} quads "
              f"({len(bad_edges):,} bad edges) → {len(quads):,} remain")
    else:
        # Ran out of passes — report how many edges are still bad
        edge_map = {}
        for qi, q in enumerate(quads):
            for a in range(4):
                e = (min(q[a], q[(a + 1) % 4]), max(q[a], q[(a + 1) % 4]))
                edge_map.setdefault(e, []).append(qi)
        remaining = sum(1 for qis in edge_map.values() if len(qis) > 2)
        if remaining:
            print(f"    WARNING: {remaining:,} non-manifold edges remain "
                  f"after {max_passes} passes")

    if total_removed:
        print(f"  Non-manifold cleanup: {total_removed:,} quads removed total")
    else:
        print(f"  Non-manifold check: mesh is already manifold")

    return quads


# ── Per-sheet worker (module-level so it is picklable by multiprocessing) ────

def _process_sheet(
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    k_base: float,
    half_t: float,
    rbf_field,
    rot_matrix,
    coarse_res: float,
    max_depth: int,
    sheet: int,
    smooth_iters: int,
    smooth_alpha: float,
) -> tuple[np.ndarray, list[list[int]], str]:
    """
    Full pipeline for one wall sheet: octree → DC → quads → smoothing.
    Stdout is captured into a string and returned so the main process can
    print each sheet's log sequentially (avoids interleaved output).
    Returns (vertices, quads, log_string).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"\n{'─'*60}")
        leaves = build_octree_leaves(
            domain_min, domain_max, k_base, half_t, rbf_field, rot_matrix,
            coarse_res=coarse_res, max_depth=max_depth, sheet=sheet,
        )

        if not leaves:
            print(f"  WARNING: sheet {sheet:+d} produced no active cells – skipping.")
            return np.empty((0, 3)), [], buf.getvalue()

        print(f"\n  Building quad connectivity …")
        verts, quads = build_quads(leaves)

        if not quads:
            print(f"  WARNING: sheet {sheet:+d}: no quads generated.")

        if quads:
            print(f"\n  Removing non-manifold quads (sheet {sheet:+d}) …")
            quads = remove_nonmanifold_quads(
                verts, quads, k_base, half_t, rbf_field, rot_matrix, sheet)

        if smooth_iters > 0 and quads:
            print(f"\n  Tangential smoothing (sheet {sheet:+d}) …")
            verts = tangential_smooth(
                verts, quads, k_base, half_t, rbf_field, rot_matrix,
                sheet=sheet, n_iters=smooth_iters, alpha=smooth_alpha,
            )

    return verts, quads, buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent

    defaults = dict(unit=1.5, wall=0.30, kbound=2.0,
                    xmin=0.0, xmax=4.0,
                    ymin=0.0, ymax=2.5,
                    zmin=0.0, zmax=10.0,
                    gyroid_rot_vec=None)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        defaults.update(read_yaml_params(Path(pre_args.config)))
    elif (script_dir / 'gyroid_case_config.yaml').exists():
        defaults.update(read_yaml_params(script_dir / 'gyroid_case_config.yaml'))

    parser = argparse.ArgumentParser(
        description='Export gyroid wall surface as quad-mesh OBJ via Dual Contouring.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ctrl',  default=str(script_dir / 'app' / 'gyroid_ctrl_pts_checkpoint.txt'))
    parser.add_argument('--out',   default=str(script_dir / 'gyroid_surface.obj'))
    parser.add_argument('--config', default=None)
    parser.add_argument('--unit',  type=float, default=defaults['unit'])
    parser.add_argument('--wall',  type=float, default=defaults['wall'])
    parser.add_argument('--kbound',type=float, default=defaults['kbound'])
    parser.add_argument('--res',   type=float, default=0.3,
                        help='Coarsest octree cell size in mm')
    parser.add_argument('--depth', type=int,   default=2,
                        help='Maximum octree refinement depth (cells halve at each level)')
    parser.add_argument('--bake',  type=float, default=0.3,
                        help='RBF bake-grid spacing in mm')
    parser.add_argument('--smooth-iters', type=int,   default=10, dest='smooth_iters',
                        help='Tangential smoothing iterations (0 = disabled)')
    parser.add_argument('--smooth-alpha', type=float, default=0.3, dest='smooth_alpha',
                        help='Tangential smoothing step size α ∈ (0,1)')
    parser.add_argument('--cap-res', type=float, default=None, dest='cap_res',
                        help='Cap grid resolution in mm (default: same as --res)')

    parser.add_argument('--xmin',  type=float, default=defaults['xmin'])
    parser.add_argument('--xmax',  type=float, default=defaults['xmax'])
    parser.add_argument('--ymin',  type=float, default=defaults['ymin'])
    parser.add_argument('--ymax',  type=float, default=defaults['ymax'])
    parser.add_argument('--zmin',  type=float, default=defaults['zmin'])
    parser.add_argument('--zmax',  type=float, default=defaults['zmax'])
    args = parser.parse_args()

    ctrl_path = Path(args.ctrl)
    out_path  = Path(args.out)
    k_base    = 2.0 * math.pi / args.unit
    _sqrt3    = math.sqrt(3)
    half_t    = 0.5 * args.wall * k_base * _sqrt3

    print(f"\n{'═'*60}")
    print(f"  Gyroid → OBJ quad mesh  (Dual Contouring + Octree)")
    print(f"{'═'*60}")
    print(f"  Checkpoint : {ctrl_path}")
    print(f"  Unit size  : {args.unit} mm   →  k_base = {k_base:.4f} rad/mm")
    print(f"  min wall   : {args.wall} mm   →  G_half_threshold = {half_t:.4f}")
    print(f"  Coarse res : {args.res} mm,  max depth = {args.depth}")
    print(f"  Domain     : x[{args.xmin},{args.xmax}]  "
          f"y[{args.ymin},{args.ymax}]  z[{args.zmin},{args.zmax}]  mm")

    # ── Load control points ────────────────────────────────────────────────────
    rbf_field = None
    if ctrl_path.exists():
        data = np.loadtxt(ctrl_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ctrl_pts = data[:, :3]; dk_ctrl = data[:, 3:6]
        if np.any(np.abs(dk_ctrl) > 1e-12):
            print(f"  Control pts: {len(ctrl_pts)}  "
                  f"(dk [{dk_ctrl.min():.4g}, {dk_ctrl.max():.4g}] rad/mm)")
            bbox_min = ctrl_pts.min(axis=0) - 0.5
            bbox_max = ctrl_pts.max(axis=0) + 0.5
            rbf_field = _BakedRBF(ctrl_pts, dk_ctrl, bbox_min, bbox_max,
                                  bake_spacing=args.bake)
            print(f"  RBF field ready.")
        else:
            print(f"  Control pts: {len(ctrl_pts)}  (all dk = 0 → uniform gyroid)")
    else:
        print(f"  WARNING: {ctrl_path} not found – using uniform gyroid")

    # ── Gyroid rotation ────────────────────────────────────────────────────────
    rot_matrix = None
    rot_vec = defaults.get('gyroid_rot_vec')
    if rot_vec is not None:
        rot_matrix = _gyroid_rotation_matrix(np.array(rot_vec, dtype=float))
        print(f"  Gyroid rot : ({rot_vec[0]:.4f}, {rot_vec[1]:.4f}, {rot_vec[2]:.4f})")

    domain_min = np.array([args.xmin, args.ymin, args.zmin])
    domain_max = np.array([args.xmax, args.ymax, args.zmax])

    all_vertices: list[np.ndarray] = []
    all_quads:    list[list[int]]  = []
    vertex_offset = 0

    # ── Run both sheets in parallel, one per core ─────────────────────────────
    # sheet=+1 → G = +half_t surface;  sheet=-1 → G = -half_t surface.
    # ProcessPoolExecutor spawns a real OS process per sheet, bypassing the GIL
    # and using two cores simultaneously.  stdout is captured inside each worker
    # and printed sequentially here so the log stays readable.
    sheet_args = dict(
        domain_min=domain_min, domain_max=domain_max,
        k_base=k_base, half_t=half_t,
        rbf_field=rbf_field, rot_matrix=rot_matrix,
        coarse_res=args.res, max_depth=args.depth,
        smooth_iters=args.smooth_iters, smooth_alpha=args.smooth_alpha,
    )
    print(f"\n  Spawning 2 worker processes (one per sheet) …")
    with ProcessPoolExecutor(max_workers=2) as pool:
        future_pos = pool.submit(_process_sheet, **sheet_args, sheet=+1)
        future_neg = pool.submit(_process_sheet, **sheet_args, sheet=-1)
        # Collect in submission order so log lines stay consistent
        results = [future_pos.result(), future_neg.result()]

    for verts_s, quads_s, log in results:
        sys.stdout.write(log)
        if len(verts_s) == 0:
            continue
        all_vertices.append(verts_s)
        all_quads.extend([[vi + vertex_offset for vi in q] for q in quads_s])
        vertex_offset += len(verts_s)

    if not all_vertices:
        sys.exit("  ERROR: no active cells found on either surface – check parameters.")

    vertices = np.concatenate(all_vertices, axis=0)
    print(f"\n{'─'*60}")
    print(f"  Combined: {len(vertices):,} vertices, {len(all_quads):,} quads across both sheets")

    # ── Close all open boundary loops ─────────────────────────────────────────
    # Face loops: vertices snapped to face plane, solid cross-section (|G|<half_t)
    # filled with flat quads sharing the snapped boundary vertices.
    # Interior loops: closed with a centroid + midpoint all-quad fan.
    cap_res = args.cap_res if args.cap_res is not None else args.res
    print(f"\n  Closing open boundary loops (cap_res={cap_res:.3f} mm) …")
    vertices, all_quads = close_open_loops(
        vertices, all_quads,
        domain_min, domain_max,
        k_base, half_t, rbf_field, rot_matrix,
        cap_res=cap_res,
    )
    print(f"  After close: {len(vertices):,} vertices, {len(all_quads):,} quads")

    # ── Write OBJ ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Writing OBJ …")
    write_obj(out_path, vertices, all_quads)
    print(f"\n  Done.")


if __name__ == '__main__':
    main()
