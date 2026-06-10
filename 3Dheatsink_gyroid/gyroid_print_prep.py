#!/usr/bin/env python3
"""
gyroid_print_prep.py — pySLM print preparation for an STL from gyroid_to_stl.py.

Pipeline
--------
1. Load the lattice STL (e.g. gyroid_surface_lattice.stl) into a pyslm.Part.
2. Optionally re-orient the mesh so the configured build direction
   (optimization.build_direction in gyroid_case_config.yaml, or --build-direction)
   points along +Z, since pySLM's overhang/slicing/support utilities all assume
   +Z is "up" (the build direction).
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
   tens of thousands of slices.

Notes / assumptions
--------------------
* "Maximum bridge length" (--max-bridge-length) is the self-supporting span
  (mm) for the chosen process/material. There is no single correct value;
  ~1.0 mm is a conservative default for laser powder bed fusion of metals.
  A downskin island's "span" is conservatively taken as the longer side of
  its minimum-area bounding rectangle - i.e. if EITHER in-plane dimension of
  the island exceeds --max-bridge-length, support is generated.
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
        if 'build_direction' in opt:
            params['build_direction'] = opt['build_direction']
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


def _polygon_span(poly) -> float:
    """Largest in-plane dimension (mm) of `poly`'s minimum-area bounding rectangle.

    Used as a conservative proxy for the unsupported bridging distance across a
    downskin island: if either side of the rectangle exceeds the printable
    bridge length, the island is treated as requiring support.
    """
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords) if rect.geom_type == 'Polygon' else None
    if not coords or len(coords) < 4:
        minx, miny, maxx, maxy = poly.bounds
        return max(maxx - minx, maxy - miny)
    edges = [math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) for i in range(2)]
    return max(edges)


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

def estimate_build_time(part: Part, layer_thickness: float, hatch_distance: float, hatch_angle: float,
                         layer_angle_increment: float, num_inner_contours: int, num_outer_contours: int,
                         spot_compensation: float, laser_speed: float, recoater_time: float,
                         layer_stride: int) -> dict:
    hatcher = Hatcher()
    hatcher.hatchAngle = hatch_angle
    hatcher.hatchDistance = hatch_distance
    hatcher.layerAngleIncrement = layer_angle_increment
    hatcher.numInnerContours = num_inner_contours
    hatcher.numOuterContours = num_outer_contours
    hatcher.spotCompensation = spot_compensation

    bstyle = BuildStyle()
    bstyle.bid = 1
    bstyle.laserSpeed = laser_speed
    model = Model(mid=1)
    model.buildStyles.append(bstyle)
    models = [model]

    bbox = part.boundingBox
    zmin, zmax = float(bbox[2]), float(bbox[5])
    n_total = max(0, int(math.floor((zmax - zmin) / layer_thickness + 1e-9)))
    z_levels = zmin + layer_thickness * (1 + np.arange(n_total))
    sampled = z_levels[::layer_stride]

    print(f"  Part height        : {zmax - zmin:.3f} mm")
    print(f"  Layer thickness    : {layer_thickness:g} mm  ->  {n_total:,} layer(s)")
    print(f"  Sampling stride    : {layer_stride}  ->  {len(sampled):,} layer(s) sliced & hatched")

    scan_time_sampled = 0.0
    n_with_geometry = 0
    t0 = time.time()
    for i, z in enumerate(sampled):
        boundary = part.getVectorSlice(float(z))
        if boundary:
            layer = hatcher.hatch(boundary)
            if layer is not None:
                for lg in layer.geometry:
                    lg.mid = model.mid
                    lg.bid = bstyle.bid
                scan_time_sampled += getLayerTime(layer, models)
                n_with_geometry += 1

        hatcher.hatchAngle = (hatcher.hatchAngle + hatcher.layerAngleIncrement * layer_stride) % 180.0

        if (i + 1) % 50 == 0 or (i + 1) == len(sampled):
            elapsed = time.time() - t0
            print(f"    ... {i + 1:,}/{len(sampled):,} sampled layers ({elapsed:.1f}s elapsed)", end='\r')
    if len(sampled) > 0:
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
    parser.add_argument('--max-bridge-length', type=float, default=1.0,
                        help='Max self-supporting in-plane span (mm) before a downskin island needs support')
    parser.add_argument('--support-inset', type=float, default=0.1,
                        help='Shrink each support footprint inward by this much (mm)')
    parser.add_argument('--support-gap', type=float, default=0.1,
                        help='Gap (mm) left between the top of a support and the downskin surface')
    parser.add_argument('--min-support-area', type=float, default=0.5,
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
            args.spot_compensation, args.laser_speed, args.recoater_time, args.layer_stride)

        print(f"\n  Total layers          : {stats['n_total_layers']:,}")
        print(f"  Sampled layers        : {stats['n_sampled_layers']:,} "
              f"({stats['n_sampled_with_geometry']:,} with geometry)")
        print(f"  Estimated scan time   : {_format_duration(stats['scan_time_s'])}")
        print(f"  Recoat time           : {_format_duration(stats['recoat_time_s'])}")
        print(f"  Estimated build time  : {_format_duration(stats['build_time_s'])}")


if __name__ == '__main__':
    main()
