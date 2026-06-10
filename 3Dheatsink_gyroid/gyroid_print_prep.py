#!/usr/bin/env python3
"""
gyroid_print_prep.py — pySLM print preparation for an STL from gyroid_to_stl.py.

Pipeline
--------
1. Load the lattice STL (e.g. gyroid_surface_lattice.stl) into a pyslm.Part.
2. Optionally re-orient the mesh so the configured build direction
   (optimization.build_direction in gyroid_case_config.yaml, or --build-direction)
   points along +Z, since pySLM's overhang/slicing/support utilities all assume
   +Z is "up" (the build direction). The re-oriented, platform-dropped mesh is
   also exported as its own STL via --build-mesh-out (default:
   <stem>_build.stl); use --skip-build-mesh to skip this.
3. Use pyslm.support.getOverhangMesh() to extract the down-facing ("downskin")
   surface, i.e. faces whose normal lies within --overhang-angle of straight
   down, and export it as its own STL.
4. Split the downskin mesh into connected islands and flatten each to a 2D
   footprint (pyslm.support.SupportStructure.flattenSupportRegion). Islands
   whose largest in-plane span exceeds --max-bridge-length cannot reliably
   self-bridge during printing and get a block support generated from the
   baseplate up to the underside of the overhang (optionally booleaned
   against the part so supports only occupy void space). Smaller islands are
   left unsupported on the assumption that they self-bridge.
5. Estimate the total build time by slicing the part at --layer-thickness,
   hatching each layer with pyslm.hatching.Hatcher, summing
   pyslm.analysis.getLayerTime() across layers and adding a fixed
   --recoater-time per layer. --layer-stride samples every Nth layer and
   extrapolates the scan time, since fine layers over a tall part can mean
   tens of thousands of slices. --num-workers > 1 slices and hatches the
   sampled layers in parallel worker processes.

Notes / assumptions
--------------------
* "Maximum bridge length" (--max-bridge-length) is the self-supporting span
  (mm) for the chosen process/material. There is no single correct value;
  ~1.0 mm is a conservative default for laser powder bed fusion of metals.
  A downskin island's "span" is the short side of the bounding rectangle
  (over all orientations) with the largest aspect ratio - i.e. the width of
  the island measured across its most elongated direction. If that width
  exceeds --max-bridge-length, support is generated.
* Process parameters for the build-time estimate (hatch distance, laser
  speed, recoat time, etc.) are generic L-PBF defaults and should be
  replaced with values for the actual machine/material.

Usage
-----
    python gyroid_print_prep.py --stl gyroid_surface_lattice.stl
    python gyroid_print_prep.py --max-bridge-length 1.5 --skip-print-time
    python gyroid_print_prep.py --layer-thickness 0.03 --layer-stride 20
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon

from pyslm import Part
from pyslm.support import getOverhangMesh, SupportStructure
from pyslm.hatching import Hatcher
from pyslm.geometry import Model, BuildStyle
from pyslm.analysis import getLayerTime

sys.path.insert(0, str(Path(__file__).parent))
from gyroid_to_stl import write_binary_stl  # noqa: E402
from gyroid_case_wrapper import (  # noqa: E402
    BoxGeometry, InletSettings, OutletSettings, _compute_flow_unit_vector,
)


# ── Build direction / config helpers ────────────────────────────────────────

def _parse_build_direction(value) -> np.ndarray:
    """Parse a build direction from a CLI string or a config value (list/tuple)."""
    if isinstance(value, str):
        s = value.strip().lower()
        mapping = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}
        if s in mapping:
            v = mapping[s]
        else:
            parts = [float(p) for p in s.replace(',', ' ').split()]
            if len(parts) != 3:
                raise ValueError(f"Invalid --build-direction {value!r}; expected 'x'/'y'/'z' or 'bx,by,bz'")
            v = parts
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        v = [float(c) for c in value]
    else:
        raise ValueError(f"Invalid build direction {value!r}")

    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("build_direction must be a non-zero vector")
    return v / norm


def _flow_unit_vector_from_config(cfg: dict) -> tuple[float, float, float]:
    """Compute the inlet->outlet flow unit vector from a config dict.

    Mirrors gyroid_case_wrapper._compute_flow_unit_vector / resolve_settings,
    using only the fields needed for that calculation (geometry.origin_mm,
    geometry.size_mm, geometry.flow_axis, inlet/outlet face + window_origin_mm).
    """
    geo_cfg = cfg.get('geometry', {})
    origin_mm = tuple(float(v) for v in geo_cfg['origin_mm'])
    size_mm = tuple(float(v) for v in geo_cfg['size_mm'])
    flow_axis = str(geo_cfg['flow_axis']).lower()

    geometry = BoxGeometry(origin_mm=origin_mm, size_mm=size_mm, cells=(1, 1, 1), flow_axis=flow_axis)

    inlet_cfg = cfg.get('inlet', {})
    outlet_cfg = cfg.get('outlet', {})
    inlet_face = str(inlet_cfg.get('face', 'min')).lower()
    outlet_face_default = 'max' if inlet_face == 'min' else 'min'
    outlet_face = str(outlet_cfg.get('face', outlet_face_default)).lower()

    inlet = InletSettings(
        face=inlet_face,
        window_origin_mm=tuple(float(v) for v in inlet_cfg.get('window_origin_mm', [0.0, 0.0])),
        window_size_mm=(0.0, 0.0),
        velocity_magnitude=0.0,
        temperature=0.0,
    )
    outlet = OutletSettings(
        face=outlet_face,
        window_origin_mm=tuple(float(v) for v in outlet_cfg.get('window_origin_mm', [0.0, 0.0])),
        window_size_mm=(0.0, 0.0),
        pressure=0.0,
    )

    return _compute_flow_unit_vector(geometry, inlet, outlet)


def read_am_params(yaml_path: Path) -> dict:
    """Read overhang-angle / build-direction defaults from gyroid_case_config.yaml."""
    params = {}
    try:
        import yaml
        with open(yaml_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        opt = cfg.get('optimization', {})
        if 'am_theta' in opt:
            params['overhang_angle'] = float(opt['am_theta'])

        build_direction = opt.get('build_direction')

        if bool(opt.get('am_align_build_to_flow', False)):
            try:
                build_direction = _flow_unit_vector_from_config(cfg)
                print(f"  am_align_build_to_flow=true: build direction set to "
                      f"inlet->outlet flow vector {build_direction}")
            except Exception as exc:
                print(f"  WARNING: am_align_build_to_flow=true but the flow direction "
                      f"could not be computed ({exc}); using optimization.build_direction")

        if build_direction is not None:
            params['build_direction'] = build_direction
    except (ImportError, OSError, ValueError, TypeError):
        pass
    return params


# ── Geometry helpers ─────────────────────────────────────────────────────────

def align_to_build_direction(mesh: trimesh.Trimesh, build_dir: np.ndarray) -> bool:
    """Rotate `mesh` in-place so `build_dir` (in part-local axes) points along +Z.

    Returns True if a rotation was applied.
    """
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(build_dir, z, atol=1e-9):
        return False
    M = trimesh.geometry.align_vectors(build_dir, z)
    mesh.apply_transform(M)
    return True


def _polygon_span(poly, n_angles: int = 180) -> float:
    """Short side (mm) of the bounding rectangle with the largest aspect ratio.

    Sweeps `poly`'s convex hull through `n_angles` orientations over [0, 90)
    degrees and, for each, measures the axis-aligned bounding box. The
    orientation that maximises length/width (i.e. makes the rectangle as
    elongated as possible) is taken to best capture the island's "narrow"
    direction - e.g. an elongated or L-shaped downskin reads as narrow even
    if its overall axis-aligned bounding box is roughly square. The short
    side of that rectangle is returned as the bridging span.
    """
    hull = poly.convex_hull
    coords = np.asarray(hull.exterior.coords[:-1]) if hull.geom_type == 'Polygon' else None
    if coords is None or len(coords) < 2:
        minx, miny, maxx, maxy = poly.bounds
        return min(maxx - minx, maxy - miny)

    angles = np.linspace(0.0, np.pi / 2, n_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    x = coords[:, 0]
    y = coords[:, 1]

    rx = np.outer(cos_a, x) + np.outer(sin_a, y)
    ry = np.outer(-sin_a, x) + np.outer(cos_a, y)

    dx = rx.max(axis=1) - rx.min(axis=1)
    dy = ry.max(axis=1) - ry.min(axis=1)

    short = np.minimum(dx, dy)
    long = np.maximum(dx, dy)

    aspect = np.divide(long, short, out=np.full_like(long, np.inf), where=short > 1e-12)

    best = int(np.argmax(aspect))
    return float(short[best])


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.1f}s  ({seconds:,.1f} s)"


# ── Stage 1: overhang / downskin extraction ─────────────────────────────────

def export_overhang_mesh(part: Part, overhang_angle: float, out_path: Path | None) -> trimesh.Trimesh:
    overhang = getOverhangMesh(part, overhang_angle, splitMesh=False)

    total_area = float(part.geometry.area)
    overhang_area = float(overhang.area)
    frac = overhang_area / total_area if total_area > 0 else 0.0
    print(f"  Overhang angle threshold : {overhang_angle:.1f} deg from horizontal")
    print(f"  Downskin faces           : {len(overhang.faces):,} / {len(part.geometry.faces):,}")
    print(f"  Downskin area            : {overhang_area:.2f} mm^2  ({frac * 100:.1f}% of total surface area)")

    if out_path is not None:
        write_binary_stl(out_path, overhang.vertices, overhang.faces)

    return overhang


# ── Stage 2: bridge-length-filtered support generation ──────────────────────

def generate_bridge_supports(part: Part, overhang: trimesh.Trimesh, max_bridge_length: float,
                              support_inset: float, support_gap: float, min_support_area: float,
                              remove_part_overlap: bool, max_supports: int) -> trimesh.Trimesh | None:
    print("\n  Splitting downskin into connected islands ...")
    regions = overhang.split(only_watertight=False)
    print(f"  {len(regions)} island(s) found")

    baseplate_z = float(part.boundingBox[2])

    prisms = []
    n_bridging = 0
    n_too_small = 0
    n_supported = 0
    n_failed = 0

    for region in regions:
        if len(region.faces) == 0:
            continue

        try:
            poly = SupportStructure.flattenSupportRegion(region)
        except Exception:
            n_failed += 1
            continue

        if poly.is_empty or poly.area < min_support_area:
            n_too_small += 1
            continue

        span = _polygon_span(poly)
        if span <= max_bridge_length:
            n_bridging += 1
            continue

        footprint = poly.buffer(-support_inset) if support_inset > 0 else poly
        if footprint.is_empty:
            footprint = poly

        top_z = float(region.vertices[:, 2].min()) - support_gap
        height = top_z - baseplate_z
        if height <= 1e-6:
            continue

        sub_polys = footprint.geoms if isinstance(footprint, MultiPolygon) else [footprint]
        for sub in sub_polys:
            if sub.is_empty or sub.area < 1e-9:
                continue
            v2, f2 = trimesh.creation.triangulate_polygon(sub)
            prism = trimesh.creation.extrude_triangulation(v2, f2, height=height)
            prism.apply_translation([0.0, 0.0, baseplate_z])
            prisms.append(prism)

        n_supported += 1
        if max_supports and n_supported >= max_supports:
            print(f"  WARNING: reached --max-supports={max_supports}, stopping early")
            break

    print(f"  Self-bridging (span <= {max_bridge_length:g} mm)   : {n_bridging}")
    print(f"  Below --min-support-area              : {n_too_small}")
    print(f"  Could not flatten (skipped)           : {n_failed}")
    print(f"  Islands requiring support             : {n_supported}")

    if not prisms:
        return None

    support_mesh = trimesh.util.concatenate(prisms)
    print(f"  Raw support volume(s): {len(prisms)} prism(s), {len(support_mesh.faces):,} faces")

    if remove_part_overlap:
        print("  Removing overlap with part geometry (manifold boolean diff) ...")
        try:
            support_mesh = trimesh.boolean.difference([support_mesh, part.geometry],
                                                        engine='manifold', check_volume=False)
            print(f"  Supports after diff: {len(support_mesh.faces):,} faces")
        except Exception as exc:
            print(f"  WARNING: boolean difference failed ({exc}); exporting raw support envelopes")

    return support_mesh


# ── Stage 3: print time estimate ─────────────────────────────────────────────

# Populated by _init_hatch_worker() in each worker process spawned by
# estimate_build_time(); module-level so the Pool.map target (_hatch_layer)
# can reuse the rebuilt Part/Hatcher inputs across many tasks without
# re-sending them on every call.
_worker_part: Part | None = None
_worker_models: list | None = None
_worker_hatch_params: dict | None = None


def _init_hatch_worker(vertices: np.ndarray, faces: np.ndarray,
                        hatch_params: dict, model_params: dict) -> None:
    """Pool initializer: rebuild the (already build-oriented) Part once per worker process."""
    global _worker_part, _worker_models, _worker_hatch_params

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    part = Part('worker')
    part.setGeometryByMesh(mesh)
    _worker_part = part
    _worker_hatch_params = hatch_params

    bstyle = BuildStyle()
    bstyle.bid = model_params['bid']
    bstyle.laserSpeed = model_params['laser_speed']
    model = Model(mid=model_params['mid'])
    model.buildStyles.append(bstyle)
    _worker_models = [model]


def _hatch_layer(task: tuple) -> tuple:
    """Slice and hatch a single layer in a worker process. Returns (scan_time_s, has_geometry)."""
    z, hatch_angle = task
    hp = _worker_hatch_params

    hatcher = Hatcher()
    hatcher.hatchAngle = hatch_angle
    hatcher.hatchDistance = hp['hatch_distance']
    hatcher.layerAngleIncrement = hp['layer_angle_increment']
    hatcher.numInnerContours = hp['num_inner_contours']
    hatcher.numOuterContours = hp['num_outer_contours']
    hatcher.spotCompensation = hp['spot_compensation']

    boundary = _worker_part.getVectorSlice(z)
    if not boundary:
        return 0.0, False

    layer = hatcher.hatch(boundary)
    if layer is None:
        return 0.0, False

    model = _worker_models[0]
    for lg in layer.geometry:
        lg.mid = model.mid
        lg.bid = model.buildStyles[0].bid

    return float(getLayerTime(layer, _worker_models)), True


def estimate_build_time(part: Part, layer_thickness: float, hatch_distance: float, hatch_angle: float,
                         layer_angle_increment: float, num_inner_contours: int, num_outer_contours: int,
                         spot_compensation: float, laser_speed: float, recoater_time: float,
                         layer_stride: int, num_workers: int = 1) -> dict:
    bbox = part.boundingBox
    zmin, zmax = float(bbox[2]), float(bbox[5])
    n_total = max(0, int(math.floor((zmax - zmin) / layer_thickness + 1e-9)))
    z_levels = zmin + layer_thickness * (1 + np.arange(n_total))
    sampled = z_levels[::layer_stride]

    print(f"  Part height        : {zmax - zmin:.3f} mm")
    print(f"  Layer thickness    : {layer_thickness:g} mm  ->  {n_total:,} layer(s)")
    print(f"  Sampling stride    : {layer_stride}  ->  {len(sampled):,} layer(s) sliced & hatched")

    # The hatch angle is rotated by layerAngleIncrement per (real) layer; precompute the
    # absolute angle for each sampled layer so layers can be processed independently / out of order.
    hatch_angles = (hatch_angle + np.arange(len(sampled)) * layer_angle_increment * layer_stride) % 180.0
    tasks = list(zip(sampled.tolist(), hatch_angles.tolist()))

    hatch_params = dict(hatch_distance=hatch_distance, layer_angle_increment=layer_angle_increment,
                         num_inner_contours=num_inner_contours, num_outer_contours=num_outer_contours,
                         spot_compensation=spot_compensation)
    model_params = dict(mid=1, bid=1, laser_speed=laser_speed)

    scan_time_sampled = 0.0
    n_with_geometry = 0
    t0 = time.time()

    if num_workers > 1 and len(tasks) > 1:
        print(f"  Using {num_workers} worker process(es) for slicing & hatching ...")
        ctx = get_context('spawn')
        with ctx.Pool(processes=num_workers, initializer=_init_hatch_worker,
                       initargs=(part.geometry.vertices, part.geometry.faces,
                                 hatch_params, model_params)) as pool:
            for i, (scan_time, has_geom) in enumerate(pool.imap_unordered(_hatch_layer, tasks)):
                scan_time_sampled += scan_time
                n_with_geometry += int(has_geom)
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    print(f"    ... {i + 1:,}/{len(tasks):,} sampled layers ({elapsed:.1f}s elapsed)", end='\r')
    else:
        bstyle = BuildStyle()
        bstyle.bid = model_params['bid']
        bstyle.laserSpeed = model_params['laser_speed']
        model = Model(mid=model_params['mid'])
        model.buildStyles.append(bstyle)
        models = [model]

        hatcher = Hatcher()
        hatcher.hatchDistance = hatch_distance
        hatcher.layerAngleIncrement = layer_angle_increment
        hatcher.numInnerContours = num_inner_contours
        hatcher.numOuterContours = num_outer_contours
        hatcher.spotCompensation = spot_compensation

        for i, (z, ha) in enumerate(tasks):
            hatcher.hatchAngle = ha
            boundary = part.getVectorSlice(z)
            if boundary:
                layer = hatcher.hatch(boundary)
                if layer is not None:
                    for lg in layer.geometry:
                        lg.mid = model.mid
                        lg.bid = bstyle.bid
                    scan_time_sampled += getLayerTime(layer, models)
                    n_with_geometry += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                print(f"    ... {i + 1:,}/{len(tasks):,} sampled layers ({elapsed:.1f}s elapsed)", end='\r')

    if len(tasks) > 0:
        print()

    scale = (n_total / len(sampled)) if len(sampled) > 0 else 0.0
    scan_time_total = scan_time_sampled * scale
    recoat_time_total = n_total * recoater_time
    build_time_total = scan_time_total + recoat_time_total

    return dict(
        n_total_layers=n_total,
        n_sampled_layers=len(sampled),
        n_sampled_with_geometry=n_with_geometry,
        scan_time_s=scan_time_total,
        recoat_time_s=recoat_time_total,
        build_time_s=build_time_total,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        am_defaults = read_am_params(Path(pre_args.config))
    elif (script_dir / 'gyroid_case_config.yaml').exists():
        am_defaults = read_am_params(script_dir / 'gyroid_case_config.yaml')
    else:
        am_defaults = {}

    parser = argparse.ArgumentParser(
        description='pySLM overhang/support/print-time preparation for a gyroid lattice STL.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--stl', default=str(script_dir / 'gyroid_surface_lattice.stl'),
                        help='Input STL (e.g. output of gyroid_to_stl.py)')
    parser.add_argument('--config', default=None,
                        help='Path to gyroid_case_config.yaml (auto-detected if omitted)')
    parser.add_argument('--overhang-angle', type=float, default=am_defaults.get('overhang_angle', 45.0),
                        help='Faces within this angle (deg) of straight down are downskin/overhang')
    parser.add_argument('--build-direction', default=am_defaults.get('build_direction', 'z'),
                        help="Build-up direction in part axes: 'x'/'y'/'z' or 'bx,by,bz'")
    parser.add_argument('--build-mesh-out', default=None,
                        help='Output STL for the mesh re-oriented into the build direction')
    parser.add_argument('--skip-build-mesh', action='store_true',
                        help='Skip exporting the build-oriented mesh')
    parser.add_argument('--max-bridge-length', type=float, default=1.5,
                        help='Max self-supporting in-plane span (mm) before a downskin island needs support')
    parser.add_argument('--support-inset', type=float, default=0.1,
                        help='Shrink each support footprint inward by this much (mm)')
    parser.add_argument('--support-gap', type=float, default=0.1,
                        help='Gap (mm) left between the top of a support and the downskin surface')
    parser.add_argument('--min-support-area', type=float, default=1.5,
                        help='Ignore downskin islands with a flattened area below this (mm^2)')
    parser.add_argument('--no-remove-part-overlap', dest='remove_part_overlap', action='store_false',
                        help='Skip the boolean diff that carves supports out of solid part material')
    parser.add_argument('--max-supports', type=int, default=300,
                        help='Safety cap on the number of support islands generated (0 = unlimited)')
    parser.add_argument('--overhang-out', default=None, help='Output STL for the downskin mesh')
    parser.add_argument('--supports-out', default=None, help='Output STL for the generated supports')
    parser.add_argument('--skip-overhang', action='store_true', help='Skip overhang extraction/export')
    parser.add_argument('--skip-supports', action='store_true', help='Skip support generation')
    parser.add_argument('--skip-print-time', action='store_true', help='Skip the print time estimate')

    parser.add_argument('--layer-thickness', type=float, default=0.03, help='Slicing layer thickness (mm)')
    parser.add_argument('--hatch-distance', type=float, default=0.10, help='Hatch line spacing (mm)')
    parser.add_argument('--hatch-angle', type=float, default=0.0, help='Base hatch angle (deg)')
    parser.add_argument('--layer-angle-increment', type=float, default=66.7,
                        help='Hatch angle rotation per layer (deg)')
    parser.add_argument('--num-inner-contours', type=int, default=1, help='Number of inner border contours')
    parser.add_argument('--num-outer-contours', type=int, default=1, help='Number of outer border contours')
    parser.add_argument('--spot-compensation', type=float, default=0.06, help='Laser spot/beam offset (mm)')
    parser.add_argument('--laser-speed', type=float, default=800.0, help='Laser scan speed (mm/s)')
    parser.add_argument('--recoater-time', type=float, default=10.0, help='Fixed recoat/dwell time per layer (s)')
    parser.add_argument('--layer-stride', type=int, default=1,
                        help='Slice & hatch every Nth layer and extrapolate (use >1 for tall parts)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Worker processes for parallel slicing & hatching of sampled layers (1 = sequential)')

    args = parser.parse_args()

    stl_path = Path(args.stl)
    build_dir = _parse_build_direction(args.build_direction)

    print(f"Loading mesh: {stl_path}")
    mesh = trimesh.load_mesh(stl_path, process=False)
    print(f"  {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices")

    if align_to_build_direction(mesh, build_dir):
        print(f"  Re-oriented build direction {build_dir.tolist()} -> +Z")

    part = Part(stl_path.stem)
    part.setGeometry(mesh, fixGeometry=True, mergeVertices=True)
    part.dropToPlatform(0.0)

    bbox = part.boundingBox
    print(f"  Bounding box (build frame): "
          f"x[{bbox[0]:.3f}, {bbox[3]:.3f}]  y[{bbox[1]:.3f}, {bbox[4]:.3f}]  z[{bbox[2]:.3f}, {bbox[5]:.3f}] mm")

    if not args.skip_build_mesh:
        build_mesh_out = Path(args.build_mesh_out) if args.build_mesh_out else \
            stl_path.with_name(stl_path.stem + '_build.stl')
        print(f"  Writing build-oriented mesh -> {build_mesh_out}")
        write_binary_stl(build_mesh_out, part.geometry.vertices, part.geometry.faces)

    overhang = None
    if not args.skip_overhang or not args.skip_supports:
        print("\n[1/3] Overhang / downskin extraction")
        if args.overhang_out:
            overhang_out = Path(args.overhang_out)
        elif args.skip_overhang:
            overhang_out = None
        else:
            overhang_out = stl_path.with_name(stl_path.stem + '_overhang.stl')
        overhang = export_overhang_mesh(part, args.overhang_angle, overhang_out)

    if not args.skip_supports:
        print("\n[2/3] Bridge-length-filtered support generation")
        support_mesh = generate_bridge_supports(
            part, overhang, args.max_bridge_length, args.support_inset, args.support_gap,
            args.min_support_area, args.remove_part_overlap, args.max_supports)
        if support_mesh is not None:
            supports_out = Path(args.supports_out) if args.supports_out else \
                stl_path.with_name(stl_path.stem + '_supports.stl')
            write_binary_stl(supports_out, support_mesh.vertices, support_mesh.faces)
        else:
            print("  No support structures required.")

    if not args.skip_print_time:
        print("\n[3/3] Print time estimate")
        if args.layer_stride > 1:
            print(f"  NOTE: sampling every {args.layer_stride} layer(s) and extrapolating "
                  f"the scan time; recoat time is exact.")
        stats = estimate_build_time(
            part, args.layer_thickness, args.hatch_distance, args.hatch_angle,
            args.layer_angle_increment, args.num_inner_contours, args.num_outer_contours,
            args.spot_compensation, args.laser_speed, args.recoater_time, args.layer_stride,
            args.num_workers)

        print(f"\n  Total layers          : {stats['n_total_layers']:,}")
        print(f"  Sampled layers        : {stats['n_sampled_layers']:,} "
              f"({stats['n_sampled_with_geometry']:,} with geometry)")
        print(f"  Estimated scan time   : {_format_duration(stats['scan_time_s'])}")
        print(f"  Recoat time           : {_format_duration(stats['recoat_time_s'])}")
        print(f"  Estimated build time  : {_format_duration(stats['build_time_s'])}")


if __name__ == '__main__':
    main()
