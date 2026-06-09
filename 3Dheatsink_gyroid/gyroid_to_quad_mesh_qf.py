"""
gyroid_to_quad_mesh_qf.py — Export the gyroid surface as a curvature-aligned quad mesh.

Pipeline
--------
1. Load RBF control-point checkpoint and bake the frequency perturbation field.
2. Write a binary params file (gyroid_mesh_params.bin) consumed by the C++
   CGAL mesher (gyroid_implicit_mesh.cpp).
3. Compile and run the CGAL mesher to produce a curvature-adaptive triangle OBJ.
   With --split, the mesher also writes separate _plus, _minus, and _sides OBJs
   (the two gyroid sheets and the flat domain-boundary closing caps).
4. Optionally auto-decimate the triangle mesh with Open3D QEM if the input/target
   ratio exceeds --max-ratio (prevents OOM in QuadriFlow).
5. Optionally apply Taubin low-pass smoothing (λ=0.5, μ=−0.53) to reduce
   high-frequency noise before remeshing.
6. Run either QuadriFlow or Instant Meshes to produce a curvature-aligned quad OBJ.
7. Repair the output (weld duplicates, remove degenerate/non-manifold faces).
8. When using the split-surface pipeline, sew the per-sheet OBJs back into a
   single combined OBJ after per-sheet remeshing and repair.

The params binary format is documented in gyroid_implicit_mesh.cpp and is the
same format produced by this script and consumed by the C++ mesher.

Usage
-----
    python gyroid_to_quad_mesh_qf.py [--config gyroid_case_config.yaml]
                                     [--ctrl <checkpoint.txt>]
                                     [--out <output.obj>]
                                     [--backend quadriflow|instant-meshes]
                                     [--target-faces N] [--no-split-surfaces]
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
from gyroid_to_stl import _BakedRBF, _gyroid_rotation_matrix, read_yaml_params

# ── Paths ──────────────────────────────────────────────────────────────────────
CPP_SRC_MESH    = _DIR / "gyroid_implicit_mesh.cpp"
CPP_BIN_MESH    = _DIR / "gyroid_implicit_mesh"
PARAMS_FILE     = _DIR / "gyroid_mesh_params.bin"
QUADRIFLOW      = Path("/workspace/quadriflow/build/quadriflow")
INSTANT_MESHES  = Path("/workspace/instant-meshes/Instant Meshes")


# ── Binary params file (identical format to gyroid_to_quad_mesh.py) ───────────

def write_params_binary(path, xmin, xmax, ymin, ymax, zmin, zmax,
                        k_base, half_t, rot_matrix, rbf_field):
    """Write the binary params file consumed by gyroid_implicit_mesh.cpp.

    Format: magic 'GYROID01' | domain+physics (8 float64) | rotation flag+matrix
            (1 byte + 9 float64) | RBF flag (1 byte) | [nx,ny,nz, origin, step,
            grid data] when RBF is present.
    """
    with open(path, 'wb') as f:
        f.write(b'GYROID01')
        f.write(struct.pack('<8d', xmin, xmax, ymin, ymax, zmin, zmax,
                            k_base, half_t))
        if rot_matrix is not None:
            f.write(struct.pack('<B', 1))
            f.write(np.asarray(rot_matrix, dtype='<f8').flatten().tobytes())
        else:
            f.write(struct.pack('<B', 0))
            f.write(np.eye(3, dtype='<f8').flatten().tobytes())
        if rbf_field is not None:
            f.write(struct.pack('<B', 1))
            nx, ny, nz = rbf_field._shape
            f.write(struct.pack('<3i', nx, ny, nz))
            gx_min = float(rbf_field._axes[0][0])
            gy_min = float(rbf_field._axes[1][0])
            gz_min = float(rbf_field._axes[2][0])
            f.write(struct.pack('<6d', gx_min, gy_min, gz_min,
                                float(rbf_field._step[0]),
                                float(rbf_field._step[1]),
                                float(rbf_field._step[2])))
            rbf_field._grid.astype('<f8').tofile(f)
        else:
            f.write(struct.pack('<B', 0))
    print(f"  Params file: {path.name}  ({path.stat().st_size/1024:.0f} kB)")


# ── Compile helper ─────────────────────────────────────────────────────────────

def _compile(src: Path, out: Path) -> None:
    """Compile src with g++ -O3 if the binary is older than the source."""
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"[build] {out.name} is up to date")
        return
    cmd = ["g++", "-std=c++17", "-O3", "-I/usr/include/eigen3",
           str(src), "-o", str(out), "-lgmp", "-lmpfr"]
    print(f"[build] {src.name}  →  {out.name}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE ERROR:\n", r.stderr)
        sys.exit(1)
    print(f"[build] OK ({time.time()-t0:.1f}s)")


# ── OBJ repair ────────────────────────────────────────────────────────────────

def _repair_quad_obj(path: Path, weld_tol: float = 1e-7) -> dict:
    """
    Repair a quad OBJ in-place. Preserves quad topology throughout.
    Returns a dict of repair statistics.

    Steps
    -----
    1. Weld vertices whose positions agree to within weld_tol (L∞).
    2. Remove degenerate faces (repeated vertex index within one face).
    3. Remove duplicate faces (same vertex set regardless of winding).
    4. Remove non-manifold edges (edge shared by ≥3 faces — drop extras).
    5. Remove isolated vertices (unreferenced after all face removals).
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    # ── parse ──────────────────────────────────────────────────────────────────
    header   = []   # comment / mtl / o / g / s / usemtl lines
    v_lines  = []   # raw "v x y z" lines
    vn_lines = []   # raw "vn …" lines (kept verbatim, indices will be dropped)
    vt_lines = []   # raw "vt …" lines (kept verbatim)
    f_raw    = []   # [(vidx_list, original_line), ...]  1-based, after v// stripping

    for line in lines:
        if line.startswith('v '):
            v_lines.append(line)
        elif line.startswith('vn ') or line.startswith('vt '):
            pass   # drop — we will not re-emit normals/texcoords (they become stale)
        elif line.startswith('f '):
            vidxs = [int(tok.split('/')[0]) for tok in line.split()[1:]]
            f_raw.append(vidxs)
        else:
            header.append(line)

    n_v_orig = len(v_lines)
    n_f_orig = len(f_raw)

    # ── 1. weld duplicate vertices ────────────────────────────────────────────
    coords = []
    for line in v_lines:
        parts = line.split()
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    # grid-hash: bucket by rounded position
    inv_tol = 1.0 / weld_tol
    bucket: dict[tuple, int] = {}   # rounded key → canonical 1-based index
    remap  = {}                      # old 1-based → new 1-based
    new_coords = []

    for i, (x, y, z) in enumerate(coords):
        key = (round(x * inv_tol), round(y * inv_tol), round(z * inv_tol))
        if key not in bucket:
            bucket[key] = len(new_coords) + 1
            new_coords.append((x, y, z))
        remap[i + 1] = bucket[key]

    n_welded = n_v_orig - len(new_coords)

    # remap all face vertex indices
    f_remapped = [[remap[v] for v in face] for face in f_raw]

    # ── 2. remove degenerate faces ────────────────────────────────────────────
    f_valid = [f for f in f_remapped if len(set(f)) == len(f)]
    n_degen = len(f_remapped) - len(f_valid)

    # ── 3. remove duplicate faces ─────────────────────────────────────────────
    seen: set[frozenset] = set()
    f_uniq = []
    for f in f_valid:
        key = frozenset(f)
        if key not in seen:
            seen.add(key)
            f_uniq.append(f)
    n_dup = len(f_valid) - len(f_uniq)

    # ── 4. remove non-manifold edges (edge shared by ≥3 faces) ───────────────
    edge_faces: dict[tuple, list] = {}
    for fi, f in enumerate(f_uniq):
        n = len(f)
        for j in range(n):
            e = tuple(sorted((f[j], f[(j + 1) % n])))
            edge_faces.setdefault(e, []).append(fi)

    nm_face_idx: set[int] = set()
    for e, fis in edge_faces.items():
        if len(fis) > 2:
            for fi in fis[2:]:   # keep first two (manifold pair), drop rest
                nm_face_idx.add(fi)

    f_manifold = [f for fi, f in enumerate(f_uniq) if fi not in nm_face_idx]
    n_nm = len(nm_face_idx)

    # ── 5. remove isolated vertices ───────────────────────────────────────────
    used = set(v for f in f_manifold for v in f)
    sorted_used = sorted(used)
    reindex = {old: new + 1 for new, old in enumerate(sorted_used)}
    n_isolated = len(new_coords) - len(sorted_used)

    final_faces = [[reindex[v] for v in f] for f in f_manifold]
    final_coords = [new_coords[v - 1] for v in sorted_used]

    # ── write ──────────────────────────────────────────────────────────────────
    out = []
    out.extend(header)
    for x, y, z in final_coords:
        out.append(f'v {x:.8g} {y:.8g} {z:.8g}\n')
    for f in final_faces:
        out.append('f ' + ' '.join(map(str, f)) + '\n')
    path.write_text(''.join(out))

    return dict(
        v_orig=n_v_orig,       v_final=len(final_coords),
        f_orig=n_f_orig,       f_final=len(final_faces),
        welded=n_welded,       degenerate=n_degen,
        duplicate=n_dup,       non_manifold=n_nm,
        isolated=n_isolated,
    )


# ── Sew meshes ────────────────────────────────────────────────────────────────

def _sew_meshes(obj_paths: list, weld_tol: float, out_path: Path) -> dict:
    """
    Load OBJs for [plus, minus, sides], merge into one OBJ with face groups,
    welding boundary vertices within weld_tol, then run _repair_quad_obj.
    Returns stats dict with 'welded' count.
    """
    group_names = ['plus_sheet', 'minus_sheet', 'sides']

    all_coords: list[tuple[float, float, float]] = []
    all_groups: list[tuple[str, list[list[int]]]] = []

    inv_tol = 1.0 / weld_tol
    bucket: dict[tuple, int] = {}   # rounded key → 1-based index in all_coords

    total_welded = 0

    for label, path in zip(group_names, obj_paths):
        raw_verts: list[tuple[float, float, float]] = []
        raw_faces: list[list[int]] = []
        for line in Path(path).read_text().splitlines():
            if line.startswith('v '):
                parts = line.split()
                raw_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                vidxs = [int(tok.split('/')[0]) for tok in line.split()[1:]]
                raw_faces.append(vidxs)

        # Build local → global vertex mapping with welding
        local_to_global: dict[int, int] = {}
        n_new_before = len(all_coords)
        for i, (x, y, z) in enumerate(raw_verts):
            key = (round(x * inv_tol), round(y * inv_tol), round(z * inv_tol))
            if key not in bucket:
                all_coords.append((x, y, z))
                bucket[key] = len(all_coords)   # 1-based
            local_to_global[i + 1] = bucket[key]   # 1-based old → 1-based global
        total_welded += (len(raw_verts) - (len(all_coords) - n_new_before))

        remapped = [[local_to_global[v] for v in f] for f in raw_faces]
        all_groups.append((label, remapped))

    # Write combined OBJ
    out = []
    for x, y, z in all_coords:
        out.append(f'v {x:.8g} {y:.8g} {z:.8g}\n')
    for label, faces in all_groups:
        out.append(f'g {label}\n')
        for f in faces:
            out.append('f ' + ' '.join(map(str, f)) + '\n')
    out_path.write_text(''.join(out))

    stats = _repair_quad_obj(out_path, weld_tol=weld_tol)
    stats['seam_welded'] = total_welded
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Defaults from YAML ────────────────────────────────────────────────────
    defaults = dict(unit=1.5, wall=0.30, kbound=2.0,
                    xmin=0.0, xmax=5.0, ymin=0.0, ymax=2.5,
                    zmin=0.0, zmax=10.0, gyroid_rot_vec=None, bake_spacing=0.3)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = (Path(pre_args.config) if pre_args.config
                   else _DIR / 'gyroid_case_config.yaml')
    if config_path.exists():
        defaults.update(read_yaml_params(config_path))

    parser = argparse.ArgumentParser(
        description='Gyroid surface → curvature-aligned quad mesh via QuadriFlow.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',   default=str(config_path))
    parser.add_argument('--ctrl',     default=str(_DIR / 'app'
                                      / 'gyroid_ctrl_pts_checkpoint.txt'))
    parser.add_argument('--out',      default=str(_DIR / 'gyroid_implicit_qf.obj'))
    parser.add_argument('--unit',     type=float, default=defaults['unit'])
    parser.add_argument('--wall',     type=float, default=defaults['wall'])
    parser.add_argument('--bake',     type=float, default=defaults['bake_spacing'])
    parser.add_argument('--xmin',     type=float, default=defaults['xmin'])
    parser.add_argument('--xmax',     type=float, default=defaults['xmax'])
    parser.add_argument('--ymin',     type=float, default=defaults['ymin'])
    parser.add_argument('--ymax',     type=float, default=defaults['ymax'])
    parser.add_argument('--zmin',     type=float, default=defaults['zmin'])
    parser.add_argument('--zmax',     type=float, default=defaults['zmax'])

    # CGAL meshing criteria
    parser.add_argument('--angular',  type=float, default=30.0,
                        help='CGAL angular bound (deg)')
    parser.add_argument('--radius',   type=float, default=1.00,
                        help='CGAL circumradius bound (mm) — controls triangle density')
    parser.add_argument('--distance', type=float, default=3.50,
                        help='CGAL surface-deviation bound (mm) — curvature-adaptive')

    # QuadriFlow parameters
    parser.add_argument('--target-faces', type=int, default=50_000,
                        dest='target_faces',
                        help='Number of quads the remesher should produce')
    parser.add_argument('--backend', default='instant-meshes',
                        choices=['quadriflow', 'instant-meshes'],
                        help='Quad remeshing backend (default: quadriflow)')
    # QuadriFlow-specific
    parser.add_argument('--max-ratio',    type=float, default=6.0,
                        dest='max_ratio',
                        help='(QuadriFlow) max input/target ratio before auto-decimation')
    parser.add_argument('--no-adaptive',  action='store_true',
                        help='(QuadriFlow) disable curvature-adaptive density')
    parser.add_argument('--no-sharp',     action='store_true',
                        help='(QuadriFlow) disable sharp-feature preservation')
    parser.add_argument('--seed',         type=int, default=42,
                        help='(QuadriFlow) RNG seed for reproducibility')
    # Shared / Instant Meshes
    parser.add_argument('--no-boundary',  action='store_true',
                        help='Disable boundary preservation (both backends)')
    parser.add_argument('--crease',       type=float, default=25.0,
                        help='(instant-meshes) dihedral-angle crease threshold (deg)')
    parser.add_argument('--smooth',       type=int,   default=2,
                        help='(instant-meshes) smoothing + reprojection steps')
    # Taubin pre-smoothing
    parser.add_argument('--taubin-iter',  type=int,   default=10,
                        dest='taubin_iter',
                        help='Taubin low-pass smoothing iterations applied to the triangle '
                             'mesh before remeshing (0 = disabled, default 10)')

    # Split-surface pipeline
    parser.add_argument('--no-split-surfaces', action='store_true',
                        dest='no_split_surfaces',
                        help='Disable split-surface pipeline (default: split-surface is on)')
    parser.add_argument('--no-taubin-sides', action='store_true',
                        dest='no_taubin_sides',
                        help='(split-surfaces) skip Taubin smoothing on the sides surface')
    parser.add_argument('--quads-sides', action='store_true',
                        dest='quads_sides',
                        help='(split-surfaces) also run quad remesher on the sides surface '
                             '(default: leave sides as triangles)')
    parser.add_argument('--sides', action='store_true',
                        dest='sides',
                        help='(split-surfaces) include the closing side surfaces; '
                             'default is open mesh (plus + minus sheets only)')
    parser.add_argument('--weld-tol',  type=float, default=0.50,
                        dest='weld_tol',
                        help='(split-surfaces) vertex weld tolerance for sewing seams (mm)')

    args = parser.parse_args()

    ctrl_path  = Path(args.ctrl)
    out_path   = Path(args.out)
    tri_obj    = out_path.with_name(out_path.stem + '_tri.obj')

    k_base = 2.0 * math.pi / args.unit
    half_t = 0.5 * args.wall * k_base * math.sqrt(3)

    print(f"\n{'═'*60}")
    print(f"  Gyroid → curvature-aligned quad mesh  [{args.backend}]")
    print(f"{'═'*60}")
    print(f"  Unit     : {args.unit} mm  →  k_base = {k_base:.4f} rad/mm")
    print(f"  Wall     : {args.wall} mm  →  half_t = {half_t:.4f}")
    print(f"  Domain   : x[{args.xmin},{args.xmax}]  "
          f"y[{args.ymin},{args.ymax}]  z[{args.zmin},{args.zmax}] mm")
    print(f"  CGAL     : angular={args.angular}°  "
          f"radius={args.radius} mm  dist={args.distance} mm")
    if args.backend == 'quadriflow':
        print(f"  QFlow    : target_faces={args.target_faces}  "
              f"adaptive={not args.no_adaptive}  "
              f"boundary={not args.no_boundary}  "
              f"sharp={not args.no_sharp}  seed={args.seed}")
    else:
        print(f"  InstMesh : target_faces={args.target_faces}  "
              f"boundary={not args.no_boundary}  "
              f"crease={args.crease}°  smooth={args.smooth}")
    taubin_str = f"{args.taubin_iter} iter" if args.taubin_iter > 0 else "disabled"
    print(f"  Taubin   : {taubin_str}")

    # ── Load ctrl pts + build baked RBF ──────────────────────────────────────
    rbf_field  = None
    rot_matrix = None

    if ctrl_path.exists():
        data = np.loadtxt(ctrl_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ctrl_pts = data[:, :3]; dk_ctrl = data[:, 3:6]
        if np.any(np.abs(dk_ctrl) > 1e-12):
            print(f"\n  Control pts: {len(ctrl_pts)}  "
                  f"(dk ∈ [{dk_ctrl.min():.3g}, {dk_ctrl.max():.3g}] rad/mm)")
            print(f"  Baking RBF (spacing={args.bake} mm) …")
            t0 = time.time()
            bbox_min = ctrl_pts.min(0) - 0.5; bbox_max = ctrl_pts.max(0) + 0.5
            rbf_field = _BakedRBF(ctrl_pts, dk_ctrl, bbox_min, bbox_max,
                                  bake_spacing=args.bake)
            nx, ny, nz = rbf_field._shape
            print(f"  RBF grid   : {nx}×{ny}×{nz}  ({time.time()-t0:.1f}s)")
        else:
            print(f"  Control pts: {len(ctrl_pts)}  (dk = 0 → uniform gyroid)")
    else:
        print(f"  WARNING: ctrl file not found → uniform gyroid")

    rot_vec = defaults.get('gyroid_rot_vec')
    if rot_vec is not None:
        rot_matrix = _gyroid_rotation_matrix(np.array(rot_vec, dtype=float))
        print(f"  Gyroid rotation: ({rot_vec[0]:.4f}, {rot_vec[1]:.4f}, {rot_vec[2]:.4f})")

    # ── Write params file ─────────────────────────────────────────────────────
    print("\n  Writing params file …")
    write_params_binary(PARAMS_FILE,
                        args.xmin, args.xmax, args.ymin, args.ymax,
                        args.zmin, args.zmax, k_base, half_t,
                        rot_matrix, rbf_field)

    # ── Step 1: CGAL → triangle OBJ ──────────────────────────────────────────
    print()
    _compile(CPP_SRC_MESH, CPP_BIN_MESH)

    print(f"\n  [1/2] CGAL implicit mesher → triangle OBJ …")
    cmd_mesh = [
        str(CPP_BIN_MESH), str(PARAMS_FILE), str(tri_obj),
        "--angular",  str(args.angular),
        "--radius",   str(args.radius),
        "--distance", str(args.distance),
        # Output is .obj → gyroid_implicit_mesh auto-selects repair+write_OBJ
    ]
    split_surfaces = not args.no_split_surfaces
    if split_surfaces:
        cmd_mesh.append("--split")
    print(f"  {' '.join(cmd_mesh)}\n")
    t0 = time.time()
    r = subprocess.run(cmd_mesh, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR: CGAL mesher exited with code {r.returncode}")
    elapsed = time.time() - t0
    tri_mb = tri_obj.stat().st_size / 1e6
    print(f"\n  Triangle OBJ: {tri_obj.name}  ({tri_mb:.1f} MB)  [{elapsed:.1f}s]")

    # ── Split-surface pipeline ────────────────────────────────────────────────
    if split_surfaces:
        stem = tri_obj.stem
        # C++ strips _tri suffix when deriving split paths; match that here
        base_stem = stem[:-4] if stem.endswith('_tri') else stem
        split_tri = {
            'plus':  tri_obj.with_name(base_stem + '_plus.obj'),
            'minus': tri_obj.with_name(base_stem + '_minus.obj'),
            'sides': tri_obj.with_name(base_stem + '_sides.obj'),
        }
        for label, p in split_tri.items():
            if not p.exists():
                sys.exit(f"ERROR: expected split OBJ not found: {p}")
            n = sum(1 for line in open(p) if line.startswith('f '))
            print(f"  {label}: {n:,} triangles  ({p.stat().st_size/1e6:.2f} MB)")

        final_outputs: list[Path] = []
        surface_labels = ['plus', 'minus', 'sides'] if args.sides else ['plus', 'minus']

        for label in surface_labels:
            src = split_tri[label]
            print(f"\n  ── Processing '{label}' surface ──")

            n_src = sum(1 for line in open(src) if line.startswith('f '))
            ratio = n_src / args.target_faces
            print(f"  Input triangles: {n_src:,}  →  ratio {ratio:.1f}×  "
                  f"(limit {args.max_ratio:.1f}×)")

            qf_input = src

            if ratio > args.max_ratio:
                import open3d as o3d
                target_tri = int(args.max_ratio * args.target_faces)
                print(f"  Decimating {n_src:,} → {target_tri:,} triangles (Open3D QEM) …")
                t0 = time.time()
                mesh_o3d = o3d.io.read_triangle_mesh(str(src))
                mesh_dec = mesh_o3d.simplify_quadric_decimation(
                    target_number_of_triangles=target_tri)
                dec_obj = src.with_name(src.stem + '_dec.obj')
                o3d.io.write_triangle_mesh(str(dec_obj), mesh_dec,
                                           write_ascii=True,
                                           write_vertex_normals=False,
                                           write_vertex_colors=False)
                n_dec = len(mesh_dec.triangles)
                print(f"  Decimated: {n_dec:,} triangles  [{time.time()-t0:.1f}s]")
                qf_input = dec_obj
            else:
                print(f"  Ratio within limit — passing directly to next step")

            do_taubin = args.taubin_iter > 0 and not (
                label == 'sides' and args.no_taubin_sides)
            if do_taubin:
                import open3d as o3d
                print(f"  Taubin smoothing: {args.taubin_iter} iterations (λ=0.5, μ=-0.53) …")
                t0 = time.time()
                mesh_s = o3d.io.read_triangle_mesh(str(qf_input))
                mesh_s = mesh_s.filter_smooth_taubin(
                    number_of_iterations=args.taubin_iter,
                    lambda_filter=0.5,
                    mu=-0.53,
                )
                mesh_s.compute_vertex_normals()
                smooth_obj = qf_input.with_name(qf_input.stem + '_smooth.obj')
                o3d.io.write_triangle_mesh(str(smooth_obj), mesh_s,
                                           write_ascii=True,
                                           write_vertex_normals=False,
                                           write_vertex_colors=False)
                print(f"  Smoothed: {smooth_obj.name}  [{time.time()-t0:.1f}s]")
                qf_input = smooth_obj

            do_quads = label != 'sides' or args.quads_sides
            surface_out = out_path.with_name(
                out_path.stem + f'_{label}.obj')

            if do_quads:
                t0 = time.time()
                if args.backend == 'quadriflow':
                    if not QUADRIFLOW.exists():
                        sys.exit(f"ERROR: QuadriFlow binary not found at {QUADRIFLOW}")
                    print(f"  QuadriFlow → quad OBJ …")
                    cmd = [
                        str(QUADRIFLOW),
                        "-i", str(qf_input),
                        "-o", str(surface_out),
                        "-f", str(args.target_faces),
                        "-seed", str(args.seed),
                    ]
                    if not args.no_adaptive:  cmd.append("-adaptive")
                    if not args.no_boundary:  cmd.append("-boundary")
                    if not args.no_sharp:     cmd.append("-sharp")
                    print(f"  {' '.join(cmd)}\n")
                    r = subprocess.run(cmd, text=True)
                    if r.returncode == -9:
                        sys.exit(
                            f"ERROR: QuadriFlow was killed (SIGKILL — likely OOM) "
                            f"on '{label}' surface.")
                    if r.returncode != 0:
                        sys.exit(f"ERROR: QuadriFlow exited with code {r.returncode}")
                else:
                    if not INSTANT_MESHES.exists():
                        sys.exit(f"ERROR: Instant Meshes binary not found at {INSTANT_MESHES}")
                    print(f"  Instant Meshes → quad OBJ …")
                    cmd = [
                        str(INSTANT_MESHES),
                        "-r", "4", "-p", "4",
                        "-f", str(args.target_faces),
                        "-S", str(args.smooth),
                        "-c", str(args.crease),
                        "-o", str(surface_out),
                        str(qf_input),
                    ]
                    if not args.no_boundary:  cmd.insert(1, "--boundaries")
                    print(f"  {' '.join(cmd)}\n")
                    import os
                    env = {"DISPLAY": ":99"}
                    env.update(os.environ)
                    r = subprocess.run(cmd, text=True, env=env)
                    if r.returncode != 0:
                        sys.exit(f"ERROR: Instant Meshes exited with code {r.returncode}")
                elapsed = time.time() - t0
                print(f"  Remesh done [{elapsed:.1f}s]")
            else:
                import shutil
                shutil.copy2(str(qf_input), str(surface_out))
                print(f"  Sides kept as triangles → {surface_out.name}")

            if surface_out.exists():
                stats = _repair_quad_obj(surface_out)
                print(f"  Repair: welded {stats['welded']}  degen {stats['degenerate']}  "
                      f"dup {stats['duplicate']}  nm {stats['non_manifold']}  "
                      f"iso {stats['isolated']}")
                print(f"  Result: {stats['v_final']:,} verts  {stats['f_final']:,} faces")
            final_outputs.append(surface_out)

        # ── Sew surfaces together ─────────────────────────────────────────────
        parts_desc = "plus + minus + sides" if args.sides else "plus + minus"
        print(f"\n  Sewing {parts_desc} → {out_path.name} …")
        sew_stats = _sew_meshes(final_outputs, args.weld_tol, out_path)
        print(f"  Sew: {sew_stats['seam_welded']} vertices welded at seams")
        print(f"  Repair: welded {sew_stats['welded']}  degen {sew_stats['degenerate']}  "
              f"dup {sew_stats['duplicate']}  nm {sew_stats['non_manifold']}  "
              f"iso {sew_stats['isolated']}")
        print(f"  Final: {sew_stats['v_final']:,} verts  {sew_stats['f_final']:,} faces")
        obj_mb = out_path.stat().st_size / 1e6
        print(f"\n  Combined OBJ: {out_path.name}  ({obj_mb:.1f} MB)")
        print(f"\n  Individual surfaces:")
        for p in final_outputs:
            print(f"    {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
        print(f"\n  Done.  Output: {out_path}")
        return

    # ── Step 1½: auto-decimate if input/target ratio is too high ─────────────
    # QuadriFlow's global scale solver is O(N) in memory but has large constants;
    # empirically it OOMs (SIGKILL / exit -9) when input_triangles > ~8–10× target.
    # We use PyVista QEM decimation to bring the ratio down to max_ratio before
    # passing to QuadriFlow.  This preserves more geometric detail than just using
    # a coarser CGAL radius, because the fine CGAL mesh captures fine curvature
    # information that guides the curvature-adaptive decimation.

    # Count faces by scanning the OBJ header (fast, no full load needed)
    n_tri = sum(1 for line in open(tri_obj) if line.startswith('f '))
    ratio = n_tri / args.target_faces
    print(f"\n  Input triangles: {n_tri:,}  →  ratio {ratio:.1f}×  (limit {args.max_ratio:.1f}×)")

    qf_input = tri_obj   # will be overwritten if decimation is needed
    n_qf_input = n_tri   # face count of what QuadriFlow actually receives

    if ratio > args.max_ratio:
        # PyVista QEM decimation on OPEN meshes creates non-manifold boundary edges
        # that crash QuadriFlow.  Open3D's QEM preserves boundary loops correctly.
        import open3d as o3d

        target_tri = int(args.max_ratio * args.target_faces)
        print(f"  Ratio exceeds limit — decimating {n_tri:,} → {target_tri:,} triangles "
              f"(Open3D QEM) …")
        t0 = time.time()
        mesh_o3d = o3d.io.read_triangle_mesh(str(tri_obj))
        mesh_dec = mesh_o3d.simplify_quadric_decimation(
            target_number_of_triangles=target_tri)
        dec_obj = tri_obj.with_name(tri_obj.stem + '_dec.obj')
        o3d.io.write_triangle_mesh(str(dec_obj), mesh_dec,
                                   write_ascii=True, write_vertex_normals=False,
                                   write_vertex_colors=False)
        n_dec = len(mesh_dec.triangles)
        print(f"  Decimated: {n_dec:,} triangles  [{time.time()-t0:.1f}s]")
        qf_input = dec_obj
        n_qf_input = n_dec
    else:
        print(f"  Ratio within limit — passing CGAL mesh directly to remesher")

    # ── Step 1¾: Taubin low-pass smoothing ───────────────────────────────────
    # Smooths the orientation field seed geometry so Instant Meshes / QuadriFlow
    # sees a cleaner surface → fewer clustered singularities in the quad layout.
    # λ=0.5, μ=-0.53 is the standard Taubin pair (near-zero shrinkage).
    if args.taubin_iter > 0:
        import open3d as o3d
        print(f"\n  Taubin smoothing: {args.taubin_iter} iterations (λ=0.5, μ=-0.53) …")
        t0 = time.time()
        mesh_s = o3d.io.read_triangle_mesh(str(qf_input))
        mesh_s = mesh_s.filter_smooth_taubin(
            number_of_iterations=args.taubin_iter,
            lambda_filter=0.5,
            mu=-0.53,
        )
        mesh_s.compute_vertex_normals()
        smooth_obj = qf_input.with_name(qf_input.stem + '_smooth.obj')
        o3d.io.write_triangle_mesh(str(smooth_obj), mesh_s,
                                   write_ascii=True, write_vertex_normals=False,
                                   write_vertex_colors=False)
        print(f"  Smoothed mesh: {smooth_obj.name}  [{time.time()-t0:.1f}s]")
        qf_input = smooth_obj

    # ── Step 2: remesh → quad OBJ ────────────────────────────────────────────
    t0 = time.time()

    if args.backend == 'quadriflow':
        if not QUADRIFLOW.exists():
            sys.exit(f"ERROR: QuadriFlow binary not found at {QUADRIFLOW}")
        print(f"\n  [2/2] QuadriFlow → curvature-aligned quad OBJ …")
        cmd = [
            str(QUADRIFLOW),
            "-i", str(qf_input),
            "-o", str(out_path),
            "-f", str(args.target_faces),
            "-seed", str(args.seed),
        ]
        if not args.no_adaptive:  cmd.append("-adaptive")
        if not args.no_boundary:  cmd.append("-boundary")
        if not args.no_sharp:     cmd.append("-sharp")
        print(f"  {' '.join(cmd)}\n")
        r = subprocess.run(cmd, text=True)
        if r.returncode == -9:
            sys.exit(
                "ERROR: QuadriFlow was killed (SIGKILL — likely OOM).\n"
                f"  Input had {n_qf_input:,} triangles for {args.target_faces:,} target quads.\n"
                f"  Try --backend instant-meshes, or --max-ratio {args.max_ratio*0.6:.1f}, "
                f"or --target-faces {int(n_qf_input/4):,}."
            )
        if r.returncode != 0:
            sys.exit(f"ERROR: QuadriFlow exited with code {r.returncode}")

    else:  # instant-meshes
        if not INSTANT_MESHES.exists():
            sys.exit(f"ERROR: Instant Meshes binary not found at {INSTANT_MESHES}")
        print(f"\n  [2/2] Instant Meshes → curvature-aligned quad OBJ …")
        # -r 4 -p 4 : quad orientation + position symmetry (pure quads)
        cmd = [
            str(INSTANT_MESHES),
            "-r", "4", "-p", "4",
            "-f", str(args.target_faces),
            "-S", str(args.smooth),
            "-c", str(args.crease),
            "-o", str(out_path),
            str(qf_input),
        ]
        if not args.no_boundary:  cmd.insert(1, "--boundaries")
        print(f"  {' '.join(cmd)}\n")
        env = {"DISPLAY": ":99"}   # headless: point at a dummy display
        import os
        env.update(os.environ)
        r = subprocess.run(cmd, text=True, env=env)
        if r.returncode != 0:
            sys.exit(f"ERROR: Instant Meshes exited with code {r.returncode}")

    elapsed = time.time() - t0

    if out_path.exists():
        print(f"\n  Repairing mesh …")
        stats = _repair_quad_obj(out_path)
        print(f"  Repair:  welded {stats['welded']} verts  |  "
              f"removed {stats['degenerate']} degenerate  |  "
              f"{stats['duplicate']} duplicate  |  "
              f"{stats['non_manifold']} non-manifold faces  |  "
              f"{stats['isolated']} isolated verts")
        print(f"  Result:  {stats['v_final']:,} vertices  {stats['f_final']:,} faces")
        obj_mb = out_path.stat().st_size / 1e6
        print(f"\n  Quad OBJ: {out_path.name}  ({obj_mb:.1f} MB)  [{elapsed:.1f}s]")
    print(f"\n  Done.  Output: {out_path}")
    print(f"Print direction: {defaults['gyroid_rot_vec']}")



if __name__ == '__main__':
    main()
