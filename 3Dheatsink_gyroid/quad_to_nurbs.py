#!/usr/bin/env python3
"""
quad_to_nurbs.py — Quad mesh OBJ → NURBS surface STEP file.

Pipeline
--------
1. Load quad OBJ.
2. Apply Catmull-Clark subdivision (default 2 levels) to obtain a smooth
   approximation of the limit surface.
3. For each original quad: extract the (2^n+1)×(2^n+1) sub-grid of vertices
   from the subdivided mesh → fit a cubic B-spline patch via OCC.
4. Sew all patches into a shell / solid shell.
5. Write STEP file.

Why one patch per original quad?
---------------------------------
The gyroid is a high-genus surface (many handles) that cannot be globally
parameterised as a single B-spline.  Each original quad defines a rectangular
region whose boundary is guaranteed to be compatible with its neighbours after
Catmull-Clark subdivision.  OCC's sewing tolerances close the seams.

Continuity at patch boundaries is C0 (patches independently fitted), which
is sufficient for most CAD/CAE applications.  For C1/C2 across patches the
standard industry tool is T-Splines (commercial).

Usage
-----
    python3 quad_to_nurbs.py input.obj output.step
    python3 quad_to_nurbs.py input.obj output.step \\
        --subd 2 --deg-min 3 --deg-max 8 --tol 0.001 --sew-tol 0.01

Requirements
------------
    OCC.wrapper  (ships with parapy-occ ≥ 7.3)
    numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── OCC imports ────────────────────────────────────────────────────────────────
try:
    from OCC.Core.gp                import gp_Pnt
    from OCC.Core.TColgp            import TColgp_Array2OfPnt
    from OCC.Core.GeomAPI           import GeomAPI_PointsToBSplineSurface
    from OCC.Core.GeomAbs           import GeomAbs_C2
    from OCC.Core.BRepBuilderAPI    import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCC.Core.BRep              import BRep_Builder
    from OCC.Core.TopoDS            import TopoDS_Compound
    from OCC.Core.STEPControl       import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect          import IFSelect_RetDone
except ImportError as e:
    sys.exit(f"ERROR: OCC not found ({e}).\n"
             "Install parapy-occ or pythonocc-core and retry.")


# ── OBJ loader ─────────────────────────────────────────────────────────────────

def load_quad_obj(path: Path):
    """Return vertices (N,3) float64 and quads (M,4) int64.  Triangles skipped."""
    verts, quads, tris_skipped = [], [], 0
    for line in path.read_text().splitlines():
        if line.startswith('v ') and not line.startswith(('vn ', 'vt ')):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            if len(idx) == 4:
                quads.append(idx)
            else:
                tris_skipped += 1
    if tris_skipped:
        print(f"  WARNING: {tris_skipped} triangular faces skipped "
              "(only quads supported)")
    return np.array(verts, dtype=np.float64), np.array(quads, dtype=np.int64)


# ── Catmull-Clark subdivision ──────────────────────────────────────────────────

def catmull_clark(verts: np.ndarray, quads: np.ndarray,
                  n_iter: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply n_iter levels of Catmull-Clark subdivision.

    After each step:
      new vertices  = [updated originals | edge points | face points]
      new faces     = 4 × old faces  (each quad → 4 child quads in fixed order)

    The fixed child order for parent quad [a,b,c,d] is:
      child 0: [a,   eab, fp,  eda]   ← bottom-left
      child 1: [eab, b,   ebc, fp ]   ← bottom-right
      child 2: [fp,  ebc, c,   ecd]   ← top-right
      child 3: [eda, fp,  ecd, d  ]   ← top-left

    This layout is exploited in extract_patch_grid() below.
    """
    for _ in range(n_iter):
        verts, quads = _cc_step(verts, quads)
    return verts, quads


def _cc_step(V: np.ndarray, F: np.ndarray):
    """One Catmull-Clark subdivision step.  Returns (new_V, new_F) with 4× the faces."""
    nV, nF = len(V), len(F)

    # ── face points ──────────────────────────────────────────────────────────
    FP = V[F].mean(axis=1)  # (nF, 3)

    # ── build edge → adjacent faces ──────────────────────────────────────────
    edge_to_faces: dict[tuple, list] = {}
    for fi, f in enumerate(F):
        for j in range(4):
            a, b = int(f[j]), int(f[(j + 1) % 4])
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append(fi)

    edges    = sorted(edge_to_faces.keys())
    edge_idx = {e: i for i, e in enumerate(edges)}
    nE       = len(edges)

    # ── edge points ──────────────────────────────────────────────────────────
    EP = np.empty((nE, 3), dtype=np.float64)
    for i, (a, b) in enumerate(edges):
        mid = 0.5 * (V[a] + V[b])
        fis = edge_to_faces[(a, b)]
        if len(fis) == 2:
            EP[i] = 0.5 * (mid + 0.5 * (FP[fis[0]] + FP[fis[1]]))
        else:
            EP[i] = mid  # boundary edge: just midpoint

    # ── updated original vertices ─────────────────────────────────────────────
    vf_list = [[] for _ in range(nV)]
    ve_list = [[] for _ in range(nV)]
    for fi, f in enumerate(F):
        for v in f:
            vf_list[int(v)].append(fi)
    for i, (a, b) in enumerate(edges):
        ve_list[a].append(i)
        ve_list[b].append(i)

    VP = np.empty((nV, 3), dtype=np.float64)
    for v in range(nV):
        vf = vf_list[v]
        ve = ve_list[v]
        n  = len(vf)
        if n == 0:
            VP[v] = V[v]
            continue
        bnd = [i for i in ve if len(edge_to_faces[edges[i]]) == 1]
        if bnd:
            # boundary vertex: weighted average with boundary edge midpoints
            m0 = 0.5 * (V[edges[bnd[0]][0]] + V[edges[bnd[0]][1]])
            m1 = 0.5 * (V[edges[bnd[-1]][0]] + V[edges[bnd[-1]][1]])
            VP[v] = 0.75 * V[v] + 0.125 * (m0 + m1)
        else:
            # interior: standard Catmull-Clark formula
            Q = FP[vf].mean(axis=0)
            R = EP[ve].mean(axis=0)
            VP[v] = (Q + 2.0 * R + (n - 3) * V[v]) / n

    # ── assemble new vertices and faces ──────────────────────────────────────
    all_V = np.concatenate([VP, EP, FP], axis=0)

    new_F = np.empty((nF * 4, 4), dtype=np.int64)
    for fi, f in enumerate(F):
        a, b, c, d = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        eab = nV + edge_idx[(min(a, b), max(a, b))]
        ebc = nV + edge_idx[(min(b, c), max(b, c))]
        ecd = nV + edge_idx[(min(c, d), max(c, d))]
        eda = nV + edge_idx[(min(d, a), max(d, a))]
        fp  = nV + nE + fi
        base = fi * 4
        new_F[base + 0] = [a,   eab, fp,  eda]
        new_F[base + 1] = [eab, b,   ebc, fp ]
        new_F[base + 2] = [fp,  ebc, c,   ecd]
        new_F[base + 3] = [eda, fp,  ecd, d  ]

    return all_V, new_F


# ── Patch grid extraction ──────────────────────────────────────────────────────

# After n_sub Catmull-Clark steps each original quad i corresponds to
# 4^n_sub child quads at consecutive indices [4^n_sub*i … 4^n_sub*(i+1)-1].
# They are arranged in a 2^n_sub × 2^n_sub spatial grid (child order as above).
# _CHILD_LAYOUT[n_sub] maps (row, col) → child index offset within the block.

def _build_child_layout(n_sub: int) -> np.ndarray:
    """Return (2^n, 2^n) array of child quad offsets for n levels of subdivision."""
    layout = np.array([[0]], dtype=np.int64)  # 1×1 at level 0
    for _ in range(n_sub):
        s = layout.shape[0]
        new_layout = np.empty((2 * s, 2 * s), dtype=np.int64)
        # Each cell c in layout expands to 4 children: 0=BL, 1=BR, 2=TR, 3=TL
        # placed at (2r,2c), (2r,2c+1), (2r+1,2c+1), (2r+1,2c)
        for r in range(s):
            for c in range(s):
                base = layout[r, c] * 4
                new_layout[2 * r,     2 * c    ] = base + 0  # BL
                new_layout[2 * r,     2 * c + 1] = base + 1  # BR
                new_layout[2 * r + 1, 2 * c + 1] = base + 2  # TR
                new_layout[2 * r + 1, 2 * c    ] = base + 3  # TL
        layout = new_layout
    return layout


def extract_patch_grid(V_sub: np.ndarray, F_sub: np.ndarray,
                       parent_idx: int, n_sub: int,
                       child_layout: np.ndarray) -> np.ndarray:
    """
    Return the (2^n_sub+1) × (2^n_sub+1) × 3 vertex grid for parent quad
    `parent_idx` after `n_sub` CC subdivision steps.

    Each sub-quad [v_BL, v_BR, v_TR, v_TL] (vertex order from _cc_step):
      index 0 = BL, 1 = BR, 2 = TR, 3 = TL
    The grid vertex at (row, col) is the BL corner of sub-quad (row, col),
    with the right/top/corner edges taken from the BR/TL/TR of boundary sub-quads.
    """
    S = 2 ** n_sub          # number of sub-quads per side
    G = S + 1               # grid vertices per side
    stride = 4 ** n_sub     # sub-quads per parent
    base   = parent_idx * stride

    grid = np.empty((G, G, 3), dtype=np.float64)

    for r in range(S):
        for c in range(S):
            f = F_sub[base + child_layout[r, c]]
            grid[r, c] = V_sub[f[0]]           # BL
        # right column: BR of the rightmost sub-quad in this row
        f = F_sub[base + child_layout[r, S - 1]]
        grid[r, S] = V_sub[f[1]]               # BR

    # top row: TL of sub-quads in the top row
    for c in range(S):
        f = F_sub[base + child_layout[S - 1, c]]
        grid[S, c] = V_sub[f[3]]               # TL

    # top-right corner: TR of top-right sub-quad
    f = F_sub[base + child_layout[S - 1, S - 1]]
    grid[S, S] = V_sub[f[2]]                   # TR

    return grid


# ── NURBS patch creation ───────────────────────────────────────────────────────

def grid_to_bspline_face(grid: np.ndarray,
                         deg_min: int = 3, deg_max: int = 8,
                         tol: float = 1e-3):
    """
    Fit a B-spline surface through an (M×N) grid of 3D points and return an
    OCC TopoDS_Face, or None if fitting / face-building fails.
    """
    M, N = grid.shape[:2]
    pts = TColgp_Array2OfPnt(1, M, 1, N)
    for i in range(M):
        for j in range(N):
            p = grid[i, j]
            pts.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    try:
        fitter = GeomAPI_PointsToBSplineSurface(pts, deg_min, deg_max,
                                                GeomAbs_C2, tol)
        if not fitter.IsDone():
            return None
        surf = fitter.Surface()
    except Exception:
        return None

    try:
        face_mk = BRepBuilderAPI_MakeFace(surf, tol)
        if not face_mk.IsDone():
            return None
        return face_mk.Face()
    except Exception:
        return None


# ── STEP writer ────────────────────────────────────────────────────────────────

def write_step(faces: list, out_path: Path, sew_tol: float = 0.01) -> bool:
    """Sew faces into a shell and write a STEP file."""
    print(f"  Sewing {len(faces)} patches (tol={sew_tol} mm) …")
    t0 = time.time()
    sew = BRepBuilderAPI_Sewing(sew_tol)
    for face in faces:
        sew.Add(face)
    sew.Perform()
    shape = sew.SewedShape()
    print(f"  Sewing done  [{time.time()-t0:.1f}s]")

    # Wrap in compound for STEP export
    builder  = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, shape)

    print(f"  Writing STEP: {out_path.name} …")
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(str(out_path))
    return status == IFSelect_RetDone


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Quad mesh OBJ → NURBS STEP via Catmull-Clark + OCC B-spline fitting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input',   type=Path, help='Input quad OBJ')
    parser.add_argument('output',  type=Path, help='Output STEP file')
    parser.add_argument('--subd',     type=int,   default=1,
                        help='Catmull-Clark subdivision levels (0 = no subdivision)')
    parser.add_argument('--deg-min',  type=int,   default=3, dest='deg_min',
                        help='Minimum B-spline degree')
    parser.add_argument('--deg-max',  type=int,   default=8, dest='deg_max',
                        help='Maximum B-spline degree')
    parser.add_argument('--tol',      type=float, default=1e-3,
                        help='B-spline fitting tolerance (mm)')
    parser.add_argument('--sew-tol',  type=float, default=0.05, dest='sew_tol',
                        help='OCC sewing tolerance for closing seams (mm)')
    parser.add_argument('--max-faces', type=int,  default=0, dest='max_faces',
                        help='Process only the first N quads (0 = all); useful for testing')
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  Quad mesh → NURBS STEP")
    print(f"{'═'*60}")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  SubD    : {args.subd} CC levels  "
          f"→ {2**args.subd + 1}×{2**args.subd + 1} point grid per patch")
    print(f"  B-spline: degree [{args.deg_min}–{args.deg_max}]  tol={args.tol} mm")
    print(f"  Sew tol : {args.sew_tol} mm\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("  Loading OBJ …")
    t0 = time.time()
    V0, F0 = load_quad_obj(args.input)
    n_orig = len(F0)
    print(f"  Loaded: {len(V0):,} vertices, {n_orig:,} quads  [{time.time()-t0:.1f}s]")

    if n_orig == 0:
        sys.exit("ERROR: no quad faces found in input OBJ.")

    n_proc = n_orig if args.max_faces <= 0 else min(args.max_faces, n_orig)
    if n_proc < n_orig:
        print(f"  (limiting to first {n_proc:,} quads — truncating before SubD)")
        # Truncate BEFORE subdivision so CC only runs on the working set
        used_verts = np.unique(F0[:n_proc])
        v_remap = np.full(len(V0), -1, dtype=np.int64)
        v_remap[used_verts] = np.arange(len(used_verts))
        V0 = V0[used_verts]
        F0 = v_remap[F0[:n_proc]]

    # ── 2. Catmull-Clark subdivision ──────────────────────────────────────────
    if args.subd > 0:
        print(f"  Catmull-Clark subdivision: {args.subd} level(s) …")
        t0 = time.time()
        V_sub, F_sub = catmull_clark(V0, F0, n_iter=args.subd)
        print(f"  After SubD: {len(V_sub):,} vertices, {len(F_sub):,} quads  "
              f"[{time.time()-t0:.1f}s]")
    else:
        V_sub, F_sub = V0, F0

    child_layout = _build_child_layout(args.subd)

    # ── 3. Fit B-spline patches ───────────────────────────────────────────────
    print(f"  Fitting B-spline patches for {n_proc:,} quads …")
    t0 = time.time()
    faces      = []
    n_failed   = 0
    report_every = max(1, n_proc // 20)

    for i in range(n_proc):
        if i > 0 and i % report_every == 0:
            pct = 100 * i / n_proc
            elapsed = time.time() - t0
            eta = elapsed / i * (n_proc - i)
            print(f"    {i:>7,} / {n_proc:,}  ({pct:.0f}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  "
                  f"failed {n_failed}")

        if args.subd > 0:
            grid = extract_patch_grid(V_sub, F_sub, i, args.subd, child_layout)
        else:
            # No subdivision: use the 4 quad corners as a 2×2 grid
            f = F0[i]
            grid = V0[f].reshape(2, 2, 3)

        face = grid_to_bspline_face(grid, args.deg_min, args.deg_max, args.tol)
        if face is not None:
            faces.append(face)
        else:
            n_failed += 1

    elapsed = time.time() - t0
    print(f"  Patches: {len(faces):,} OK  {n_failed:,} failed  [{elapsed:.1f}s]")

    if not faces:
        sys.exit("ERROR: all patches failed — check input mesh quality.")

    # ── 4. Sew + write STEP ───────────────────────────────────────────────────
    ok = write_step(faces, args.output, sew_tol=args.sew_tol)
    if ok:
        mb = args.output.stat().st_size / 1e6
        print(f"\n  Done.  {args.output.name}  ({mb:.1f} MB)")
    else:
        sys.exit("ERROR: STEP write failed.")


if __name__ == '__main__':
    main()
