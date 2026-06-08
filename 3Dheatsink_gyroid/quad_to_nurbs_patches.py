#!/usr/bin/env python3
"""
quad_to_nurbs_patches.py — Quad mesh OBJ → multi-patch NURBS surface STEP file
via motorcycle-graph segmentation.

Pipeline
--------
1. Load a pure-quad, field-aligned OBJ (mostly valence-4 vertices).
2. Build halfedge connectivity and find irregular interior vertices
   (valence ≠ 4) — these are the singularities of the cross-field, and the
   sources of the "motorcycle graph" separatrices.
3. From every irregular vertex, trace a separatrix in each outgoing quad
   direction: walk straight across the grid (enter a quad through one edge,
   exit the topologically opposite edge) until hitting another irregular
   vertex or the mesh boundary.  The union of all traced edges is the
   motorcycle graph; it partitions the mesh into logically-rectangular blocks.
4. Flood-fill quads into blocks, never crossing a motorcycle-graph edge.
5. Order each block's quads into a 2D (rows × cols) array by straight-walking,
   and read off the (rows+1) × (cols+1) corner-vertex grid — one grid per
   block.  Blocks that are not actually rectangular (an L/T-junction where a
   separatrix terminates mid-patch) fall back to one 2×2 grid per quad, so no
   geometry is ever lost.
6. Fit one cubic B-spline surface per grid (reusing the same OCC fitting code
   as quad_to_nurbs.py), sew all patches into a shell, and write a STEP file.

Why this is better than one-patch-per-quad
-------------------------------------------
A field-aligned quad mesh is *almost* a regular grid: away from singularities
it tiles into large rectangular regions whose interior quads all share the
same two grid directions.  Fitting one B-spline per such region instead of
per quad collapses tens/hundreds of thousands of tiny patches into hundreds —
with surface detail/curvature adaptivity still inherited from the local quad
density of the input mesh (denser quads → more interior grid lines → a
higher-resolution fit for that region).

Note on correctness of the simplification: a "true" motorcycle graph applies
crash/priority rules between separatrices that meet mid-edge.  This
implementation simply unions every traced separatrix unconditionally, which
can only ever produce a *finer* partition (more, smaller blocks) than the true
graph — never an invalid one.  So patch count may be larger than optimal on
very high-genus meshes, but the result is always geometrically sound.

Continuity at patch boundaries is C0 (each patch is independently fitted),
exactly as in quad_to_nurbs.py.  For C1/C2 continuity across patches the
standard industry tool is T-Splines (commercial).

Usage
-----
    python3 quad_to_nurbs_patches.py input.obj output.step
    python3 quad_to_nurbs_patches.py input.obj output.step \\
        --deg-min 3 --deg-max 8 --tol 0.001 --sew-tol 0.05 --max-faces 2000

Requirements
------------
    OCC.wrapper  (ships with parapy-occ ≥ 7.3)
    numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

# ── OCC imports ────────────────────────────────────────────────────────────────
try:
    from OCC.wrapper.gp                import gp_Pnt
    from OCC.wrapper.TColgp            import TColgp_Array2OfPnt
    from OCC.wrapper.GeomAPI           import GeomAPI_PointsToBSplineSurface
    from OCC.wrapper.GeomAbs           import GeomAbs_C2
    from OCC.wrapper.BRepBuilderAPI    import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCC.wrapper.BRep              import BRep_Builder
    from OCC.wrapper.TopoDS            import TopoDS_Compound
    from OCC.wrapper.STEPControl       import STEPControl_Writer, STEPControl_AsIs
    from OCC.wrapper.IFSelect          import IFSelect_RetDone
except ImportError as e:
    sys.exit(f"ERROR: OCC not found ({e}).\n"
             "Install parapy-occ or pythonocc-core and retry.")


# ── OBJ loader (verbatim from quad_to_nurbs.py) ────────────────────────────────

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


# ── NURBS patch creation (verbatim from quad_to_nurbs.py) ──────────────────────

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


# ── STEP writer (verbatim from quad_to_nurbs.py) ───────────────────────────────

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


# ── Quad halfedge topology ──────────────────────────────────────────────────────

def build_quad_topology(V: np.ndarray, F: np.ndarray):
    """
    Build halfedge tables for a pure-quad mesh.

    A directed halfedge is the pair (a, b) for consecutive face corners
    f[j] -> f[(j+1)%4].  Returns:
      he   : dict (a, b) -> (face_index, local_corner_j)
      twin : dict (face_index, j) -> (face, j') | None  — the opposite-direction
             halfedge (b, a); None on a boundary (no reverse halfedge exists).
      val  : Counter of vertex valence (each undirected edge counted once per
             endpoint, i.e. the number of edges incident to a vertex).

    Robust to non-manifold edges: if (a, b) already has a halfedge recorded,
    the duplicate is skipped (first one wins) rather than raising.
    """
    he: dict = {}
    val: Counter = Counter()
    seen_undirected: set = set()

    for fi, f in enumerate(F):
        for j in range(4):
            a, b = int(f[j]), int(f[(j + 1) % 4])
            if (a, b) in he:
                continue  # non-manifold duplicate — keep the first
            he[(a, b)] = (fi, j)
            key = (min(a, b), max(a, b))
            if key not in seen_undirected:
                seen_undirected.add(key)
                val[a] += 1
                val[b] += 1

    twin: dict = {}
    for (a, b), (fi, j) in he.items():
        twin[(fi, j)] = he.get((b, a))

    return he, twin, val


def irregular_vertices(val: Counter, F: np.ndarray, he: dict, twin: dict):
    """
    Return the set of interior vertices with valence != 4.

    A vertex is interior iff every halfedge leaving it has a twin (i.e. no
    incident edge is a mesh boundary edge).  Boundary vertices are excluded:
    they're handled as ordinary patch corners/edges, not separatrix sources.
    """
    boundary_vertices: set = set()
    for (fi, j), tw in twin.items():
        if tw is None:
            a = int(F[fi][j])
            b = int(F[fi][(j + 1) % 4])
            boundary_vertices.add(a)
            boundary_vertices.add(b)

    irregular = {v for v, n in val.items()
                 if n != 4 and v not in boundary_vertices}
    return irregular


# ── Separatrix tracing (motorcycle graph) ──────────────────────────────────────

def step_straight(he: dict, twin: dict, fi: int, j: int):
    """
    Enter face `fi` across local edge `j`; continue straight by exiting
    through the topologically opposite edge (j+2)%4 and crossing its twin
    into the neighbouring face.

    Returns (next_face, next_local_edge), or None at a mesh boundary.
    The twin (nf, nj) IS the next state: nj is the local edge of the
    neighbour through which we just entered, so the next straight exit is
    again the edge opposite nj — exactly what a recursive call needs.
    """
    opposite = (j + 2) % 4
    return twin.get((fi, opposite))


def trace_separatrix(he: dict, twin: dict, F: np.ndarray,
                      start_face: int, start_edge: int, irregular: set):
    """
    Walk straight from (start_face, start_edge), accumulating the ordered
    list of (face, local_edge) crossed.

    The starting edge is always the launch edge out of an irregular source
    vertex (so it always has an irregular endpoint by construction) and is
    unconditionally included.  Walking then continues, crossing one edge per
    step, until either:
      (a) step_straight runs off the mesh boundary (returns None), or
      (b) the just-crossed edge has an endpoint in `irregular` — i.e. we've
          reached another singularity and the separatrix terminates there.

    A hard iteration cap guards against pathological/closed-loop input.
    """
    fi, j = start_face, start_edge
    path = [(fi, j)]

    for _ in range(100_000):
        nxt = step_straight(he, twin, fi, j)
        if nxt is None:
            break
        fi, j = nxt
        path.append((fi, j))
        a = int(F[fi][j])
        b = int(F[fi][(j + 1) % 4])
        if a in irregular or b in irregular:
            break

    return path


def collect_separatrix_edges(V: np.ndarray, F: np.ndarray, he: dict, twin: dict,
                             val: Counter):
    """
    For every irregular vertex, launch a separatrix down each outgoing quad
    direction and union all crossed undirected edges into the motorcycle-graph
    edge set `sep`.

    Returns (sep, irregular):
      sep       : set of undirected edges as tuple(sorted((a, b)))
      irregular : the set of irregular interior vertices (returned so callers
                  don't have to recompute it)
    """
    irregular = irregular_vertices(val, F, he, twin)

    out_from: dict = {}
    for (a, b), (fi, j) in he.items():
        out_from.setdefault(a, []).append((fi, j))

    sep: set = set()
    for v in irregular:
        for (fi, j) in out_from.get(v, []):
            path = trace_separatrix(he, twin, F, fi, j, irregular)
            for pfi, pj in path:
                a = int(F[pfi][pj])
                b = int(F[pfi][(pj + 1) % 4])
                sep.add((min(a, b), max(a, b)))

    return sep, irregular


# ── Block ordering ──────────────────────────────────────────────────────────────

def _walk_straight_strip(F: np.ndarray, he: dict, twin: dict, is_border,
                         members_set: set, start_fi: int, start_o: int,
                         direction: str, cap: int):
    """
    Walk a straight strip of faces starting at (start_fi, start_o), where
    `start_o` is the local corner index that is the block-relative
    bottom-left (BL) corner of `start_fi`.

    direction='u' walks toward increasing column, 'v' toward increasing row.
    The crossing/landing bookkeeping mirrors the corner-quad derivation:
        +u: cross edge (o+1)%4 via step_straight(fi, (o+3)%4);
            landing face's BL local index = (entry_local_edge + 1) % 4
        +v: cross edge (o+2)%4 via step_straight(fi,  o       );
            landing face's BL local index =  entry_local_edge

    Returns the ordered list [(face, o_idx), ...] (length >= 1, terminating
    at a separatrix/boundary/outside-block edge), or None if the walk would
    revisit a face (closed loop) or exceed `cap` steps — both signs that the
    block cannot be a simple rectangle.
    """
    strip = [(start_fi, start_o)]
    seen  = {start_fi}
    fi, o = start_fi, start_o

    for _ in range(cap):
        if direction == 'u':
            cross_edge, ref_edge = (o + 1) % 4, (o + 3) % 4
        else:
            cross_edge, ref_edge = (o + 2) % 4, o

        if is_border(fi, cross_edge):
            return strip
        nxt = step_straight(he, twin, fi, ref_edge)
        if nxt is None:
            return strip

        nf, nj = nxt
        if nf not in members_set or nf in seen:
            return None  # left the block, or closed loop — not rectangular

        fi = nf
        o  = (nj + 1) % 4 if direction == 'u' else nj
        seen.add(fi)
        strip.append((fi, o))

    return None  # exceeded the cap — pathological / closed topology


def order_block_into_grid(V: np.ndarray, F: np.ndarray, he: dict, twin: dict,
                          sep: set, members: list):
    """
    Order a block's quads into a rectangular (rows × cols) face grid and
    harvest the (rows+1) × (cols+1) corner-vertex grid — or return None if
    the block isn't actually rectangular (an L/T-junction where a separatrix
    terminates mid-block).  Callers fall back to per-quad 2×2 grids on None,
    so no geometry is ever lost.

    Orientation bookkeeping
    -----------------------
    For any face whose local corner `o` has been identified as the
    block-relative bottom-left (BL), the remaining corners follow the walk
    axes by the standard cyclic rule — independent of the face's raw OBJ
    winding:
        BL = corner[o]           BR = corner[(o+1) % 4]
        TL = corner[(o+3) % 4]   TR = corner[(o+2) % 4]
    Propagating `o` consistently through every straight-walk step (see
    _walk_straight_strip) is what guarantees adjacent patches share identical
    seam vertices — critical for OCC sewing to close the shell.
    """
    members_set = set(members)

    def is_border(fi, j):
        a = int(F[fi][j])
        b = int(F[fi][(j + 1) % 4])
        if (min(a, b), max(a, b)) in sep:
            return True
        tw = twin.get((fi, j))
        if tw is None:
            return True
        return tw[0] not in members_set

    # ── 1. Corner quad: two adjacent border edges (j, (j+1)%4) meet at local
    #       corner (j+1)%4 — that vertex becomes the block's grid origin (0,0).
    corner = None
    for fi in members:
        for j in range(4):
            if is_border(fi, j) and is_border(fi, (j + 1) % 4):
                corner = (fi, (j + 1) % 4)
                break
        if corner is not None:
            break
    if corner is None:
        return None  # no boundary corner found (e.g. a closed-loop block)

    fi0, o_idx0 = corner
    cap = len(members) + 1

    # ── 2/3. Walk the first row (+u) and first column (+v) from the corner ──
    row0 = _walk_straight_strip(F, he, twin, is_border, members_set,
                                fi0, o_idx0, 'u', cap)
    col0 = _walk_straight_strip(F, he, twin, is_border, members_set,
                                fi0, o_idx0, 'v', cap)
    if row0 is None or col0 is None:
        return None
    cols, rows = len(row0), len(col0)
    if rows * cols != len(members):
        return None

    # ── 4. Fill the interior: walk +u across each row from its column-0 face
    face_grid = [row0]
    for r in range(1, rows):
        start_fi, start_o = col0[r]
        row = _walk_straight_strip(F, he, twin, is_border, members_set,
                                   start_fi, start_o, 'u', cols)
        if row is None or len(row) != cols:
            return None
        face_grid.append(row)

    # ── 5. Validate exact, non-overlapping coverage of the block ────────────
    seen_faces = {fi for row in face_grid for fi, _ in row}
    if len(seen_faces) != rows * cols or seen_faces != members_set:
        return None

    # ── 6. Harvest the (rows+1) x (cols+1) corner-vertex grid ───────────────
    # Mirrors extract_patch_grid() in quad_to_nurbs.py: BL of every cell, plus
    # BR along the right edge, TL along the top edge, TR at the far corner.
    grid = np.empty((rows + 1, cols + 1, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            fi, o = face_grid[r][c]
            grid[r, c] = V[int(F[fi][o])]                      # BL
        fi, o = face_grid[r][cols - 1]
        grid[r, cols] = V[int(F[fi][(o + 1) % 4])]             # BR
    for c in range(cols):
        fi, o = face_grid[rows - 1][c]
        grid[rows, c] = V[int(F[fi][(o + 3) % 4])]             # TL
    fi, o = face_grid[rows - 1][cols - 1]
    grid[rows, cols] = V[int(F[fi][(o + 2) % 4])]              # TR

    return grid


# ── Orchestration ───────────────────────────────────────────────────────────────

def collect_patches(V: np.ndarray, F: np.ndarray):
    """
    Orchestrate the full motorcycle-graph segmentation pipeline:
      1. Build halfedge topology and trace the motorcycle graph (separatrix
         edges from every irregular vertex).
      2. Flood-fill quads into blocks, never crossing a motorcycle-graph edge
         — each block is bounded by separatrices and/or the mesh boundary.
      3. Order each block into a rectangular grid and harvest its corner-
         vertex grid; blocks that aren't actually rectangular (an L/T-shape
         where a separatrix terminates mid-block) fall back to one 2×2 grid
         per quad, so no geometry is ever lost.

    Returns (grids, n_fallback_quads, n_blocks_total, n_blocks_rect):
      grids            : list of (rows+1, cols+1, 3) float64 vertex grids,
                         one per patch to be fitted
      n_fallback_quads : number of quads emitted as standalone 2×2 fallbacks
      n_blocks_total   : number of flood-filled blocks
      n_blocks_rect    : number of those blocks that ordered rectangularly
    """
    he, twin, val = build_quad_topology(V, F)
    sep, irregular = collect_separatrix_edges(V, F, he, twin, val)

    # ── flood-fill quads into blocks, never crossing a motorcycle-graph edge ──
    nF = len(F)
    block_of = np.full(nF, -1, dtype=np.int64)
    blocks: list[list[int]] = []

    for seed in range(nF):
        if block_of[seed] != -1:
            continue
        block_id = len(blocks)
        members: list[int] = [seed]
        block_of[seed] = block_id
        queue = deque([seed])
        while queue:
            fi = queue.popleft()
            for j in range(4):
                a = int(F[fi][j])
                b = int(F[fi][(j + 1) % 4])
                if (min(a, b), max(a, b)) in sep:
                    continue
                tw = twin.get((fi, j))
                if tw is None:
                    continue
                nf = tw[0]
                if block_of[nf] == -1:
                    block_of[nf] = block_id
                    members.append(nf)
                    queue.append(nf)
        blocks.append(members)

    # ── order each block; fall back to per-quad 2×2 grids on failure ────────
    grids: list[np.ndarray] = []
    n_fallback_quads = 0
    n_blocks_rect    = 0

    for members in blocks:
        grid = order_block_into_grid(V, F, he, twin, sep, members)
        if grid is not None:
            grids.append(grid)
            n_blocks_rect += 1
        else:
            for fi in members:
                grids.append(V[F[fi]].reshape(2, 2, 3))
            n_fallback_quads += len(members)

    return grids, n_fallback_quads, len(blocks), n_blocks_rect


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Quad mesh OBJ → multi-patch NURBS STEP via motorcycle-graph '
                    'segmentation + OCC B-spline fitting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input',   type=Path, help='Input quad OBJ')
    parser.add_argument('output',  type=Path, help='Output STEP file')
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
    print(f"  Quad mesh → multi-patch NURBS STEP (motorcycle-graph segmentation)")
    print(f"{'═'*60}")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  B-spline: degree [{args.deg_min}–{args.deg_max}]  tol={args.tol} mm")
    print(f"  Sew tol : {args.sew_tol} mm\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("  Loading OBJ …")
    t0 = time.time()
    V, F = load_quad_obj(args.input)
    n_orig = len(F)
    print(f"  Loaded: {len(V):,} vertices, {n_orig:,} quads  [{time.time()-t0:.1f}s]")

    if n_orig == 0:
        sys.exit("ERROR: no quad faces found in input OBJ.")

    if args.max_faces > 0 and args.max_faces < n_orig:
        n_proc = args.max_faces
        print(f"  (limiting to first {n_proc:,} quads)")
        used_verts = np.unique(F[:n_proc])
        v_remap = np.full(len(V), -1, dtype=np.int64)
        v_remap[used_verts] = np.arange(len(used_verts))
        V = V[used_verts]
        F = v_remap[F[:n_proc]]

    # ── 2. Segment into logically-rectangular blocks ─────────────────────────
    print("  Segmenting mesh via motorcycle-graph tracing …")
    t0 = time.time()
    grids, n_fallback_quads, n_blocks_total, n_blocks_rect = collect_patches(V, F)
    print(f"  Segmented into {len(grids):,} patches (from {len(F):,} quads)  "
          f"[{time.time()-t0:.1f}s]")
    print(f"  Blocks: {n_blocks_total:,} total, {n_blocks_rect:,} rectangular, "
          f"{n_blocks_total - n_blocks_rect:,} irregular "
          f"({n_fallback_quads:,} quads emitted as 2×2 fallback patches)")

    hist = Counter((g.shape[0] - 1, g.shape[1] - 1) for g in grids)
    print("  Patch grid-size histogram (rows×cols : count), top 15:")
    for (rows, cols), cnt in hist.most_common(15):
        print(f"    {rows:>4} × {cols:<4} : {cnt:,}")

    # ── 3. Fit B-spline patches ───────────────────────────────────────────────
    print(f"  Fitting B-spline patches for {len(grids):,} blocks …")
    t0 = time.time()
    faces    = []
    n_failed = 0
    n_total  = len(grids)
    report_every = max(1, n_total // 20)

    for i, grid in enumerate(grids):
        if i > 0 and i % report_every == 0:
            pct = 100 * i / n_total
            elapsed = time.time() - t0
            eta = elapsed / i * (n_total - i)
            print(f"    {i:>7,} / {n_total:,}  ({pct:.0f}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  "
                  f"failed {n_failed}")

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
