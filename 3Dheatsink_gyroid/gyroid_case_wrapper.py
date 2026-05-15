"""High-level case wrapper for the gyroid RBF optimizer.

This script owns the OpenFOAM case preparation step:

1. Reset the case workspace.
2. Write a box mesh with configurable size and inlet/outlet axis.
3. Write material and decomposition dictionaries.
4. Run blockMesh, topoSet, and decomposePar.
5. Launch the existing gyroid optimizer with matching geometry bounds.

The wrapper keeps the optimizer itself unchanged in spirit: it still evaluates
the gyroid field and drives the solver, but it no longer has to know how the
case was assembled.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from clean_for_gyroid import clean_case
from gyroid_rbf_optimizer import GyroidRBFOptimizer, ParetoExplorer, SOLID_DENSITY_G_PER_MM3


MM_TO_MESH_UNIT = 10.0


@dataclass(frozen=True)
class BoxGeometry:
    origin_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    cells: tuple[int, int, int]
    flow_axis: str


@dataclass(frozen=True)
class MaterialProperties:
    nu: float
    alpha_max: float
    mma_init: float
    mma_dec: float
    mma_inc: float
    movlim: float
    voluse: float
    filter_radius: float
    solid_area: float
    fluid_area: float
    test_pd: float
    d_normalization: float
    d0: float
    d1: float
    geo_dim: float
    b1: float
    qu: float
    kf: float
    ks: float
    rhoc: float
    t_alpha: float
    solid_density_g_per_mm3: float
    darcy_number: float
    Texterior: float
    hconv: float


@dataclass(frozen=True)
class InletSettings:
    face: str
    window_origin_mm: tuple[float, float]
    window_size_mm: tuple[float, float]
    velocity_magnitude: float
    temperature: float


@dataclass(frozen=True)
class OutletSettings:
    face: str
    window_origin_mm: tuple[float, float]
    window_size_mm: tuple[float, float]
    pressure: float


@dataclass(frozen=True)
class TurbulenceProperties:
    simulation_type: str
    ras_model: str
    turbulence: str
    print_coeffs: str


@dataclass(frozen=True)
class ThermalSettings:
    initial_temperature: float


@dataclass(frozen=True)
class RunSettings:
    case: str
    solver: str
    parallel: int
    postprocess: str
    skip_clean: bool
    iters: int


@dataclass(frozen=True)
class OptimizationSettings:
    unit: float
    wall: float
    epsilon: float
    spacing: float
    bake_spacing: float
    kbound: float
    mode: str
    meantT_max: float | None
    dissPower_max: float | None
    mu_penalty: float
    mu_kmin: float
    am_theta: float
    am_P_bar: float
    mu_overhang: float
    no_overhang: bool
    pareto_enabled: bool
    pareto_weights: list | None
    pareto_n_weights: int
    pareto_iters_per_point: int
    pareto_warmstart: bool


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _require(mapping: dict, key: str, context: str) -> object:
    if key not in mapping:
        raise KeyError(f"Missing required key '{key}' in {context}")
    return mapping[key]


def _as_float_tuple(values: object, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{context} must be a sequence of {length} numbers")
    return tuple(float(v) for v in values)


def _as_int_tuple(values: object, length: int, context: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{context} must be a sequence of {length} integers")
    return tuple(int(v) for v in values)


def _as_bool(value: object) -> bool:
    return bool(value)


def _as_foam_switch(value: object) -> str:
    """Convert YAML value to OpenFOAM on/off switch string.
    
    YAML may parse 'on'/'off' as booleans, so we need to handle both
    boolean and string inputs, converting them to OpenFOAM-compatible values.
    """
    if isinstance(value, bool):
        return 'on' if value else 'off'
    s = str(value).lower().strip()
    if s in {'true', '1', 'yes', 'on'}:
        return 'on'
    if s in {'false', '0', 'no', 'off'}:
        return 'off'
    return s


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def resolve_settings(config: dict, cli_args: argparse.Namespace) -> tuple[RunSettings, BoxGeometry, MaterialProperties, OptimizationSettings, InletSettings, OutletSettings, TurbulenceProperties, ThermalSettings]:
    run_cfg = dict(config.get('run', {}))
    geometry_cfg = dict(config.get('geometry', {}))
    material_cfg = dict(config.get('material', {}))
    optimization_cfg = dict(config.get('optimization', {}))

    case = cli_args.case or run_cfg.get('case', 'app')
    solver = cli_args.solver or run_cfg.get('solver', 'MTO_TF')
    parallel = int(cli_args.parallel if cli_args.parallel is not None else run_cfg.get('parallel', 1))
    postprocess = cli_args.postprocess or run_cfg.get('postprocess', 'postProcess')
    skip_clean = bool(cli_args.skip_clean or run_cfg.get('skip_clean', False))
    iters = int(cli_args.iters if cli_args.iters is not None else run_cfg.get('iters', 50))

    origin_mm = _as_float_tuple(_require(geometry_cfg, 'origin_mm', 'geometry'), 3, 'geometry.origin_mm')
    size_mm = _as_float_tuple(_require(geometry_cfg, 'size_mm', 'geometry'), 3, 'geometry.size_mm')
    cells = _as_int_tuple(_require(geometry_cfg, 'cells', 'geometry'), 3, 'geometry.cells')
    flow_axis = str(_require(geometry_cfg, 'flow_axis', 'geometry')).lower()

    geometry = BoxGeometry(
        origin_mm=origin_mm,
        size_mm=size_mm,
        cells=cells,
        flow_axis=flow_axis,
    )

    inlet_cfg = dict(config.get('inlet', {}))
    inlet_face = str(inlet_cfg.get('face', 'min')).lower()
    if inlet_face not in {'min', 'max'}:
        raise ValueError("inlet.face must be either 'min' or 'max'")

    transverse_sizes = [geometry.size_mm[i] for i in range(3) if i != _axis_index(geometry.flow_axis)]
    inlet_origin_mm = inlet_cfg.get('window_origin_mm', [0.0, 0.0])
    inlet_size_mm = inlet_cfg.get('window_size_mm', transverse_sizes)
    window_origin_mm = _as_float_tuple(inlet_origin_mm, 2, 'inlet.window_origin_mm')
    window_size_mm = _as_float_tuple(inlet_size_mm, 2, 'inlet.window_size_mm')
    inlet = InletSettings(
        face=inlet_face,
        window_origin_mm=window_origin_mm,
        window_size_mm=window_size_mm,
        velocity_magnitude=float(inlet_cfg.get('velocity_magnitude', 0.1)),
        temperature=float(inlet_cfg.get('temperature', 0.0)),
    )

    outlet_cfg = dict(config.get('outlet', {}))
    outlet_face_default = 'max' if inlet_face == 'min' else 'min'
    outlet_face = str(outlet_cfg.get('face', outlet_face_default)).lower()
    if outlet_face not in {'min', 'max'}:
        raise ValueError("outlet.face must be either 'min' or 'max'")
    if outlet_face == inlet_face:
        raise ValueError('inlet.face and outlet.face must be opposite sides of the flow axis')

    outlet_origin_mm = outlet_cfg.get('window_origin_mm', [0.0, 0.0])
    outlet_size_mm = outlet_cfg.get('window_size_mm', transverse_sizes)
    outlet = OutletSettings(
        face=outlet_face,
        window_origin_mm=_as_float_tuple(outlet_origin_mm, 2, 'outlet.window_origin_mm'),
        window_size_mm=_as_float_tuple(outlet_size_mm, 2, 'outlet.window_size_mm'),
        pressure=float(outlet_cfg.get('pressure', 0.0)),
    )

    turbulence_cfg = dict(config.get('turbulence', {}))
    turbulence = TurbulenceProperties(
        simulation_type=str(turbulence_cfg.get('simulation_type', 'laminar')),
        ras_model=str(turbulence_cfg.get('ras_model', 'laminar')),
        turbulence=_as_foam_switch(turbulence_cfg.get('turbulence', 'off')),
        print_coeffs=_as_foam_switch(turbulence_cfg.get('print_coeffs', 'off')),
    )

    thermal_cfg = dict(config.get('thermal', {}))
    thermal = ThermalSettings(
        initial_temperature=float(thermal_cfg.get('initial_temperature', 0.0)),
    )

    # Compute alphaMax from Darcy number if provided, otherwise use direct value.
    # alphaMax = nu / (L^2 * Da) where L is the geometric mean of transverse dimensions.
    nu_value = float(_require(material_cfg, 'nu', 'material'))
    darcy_number_value = float(material_cfg.get('darcy_number', 0.0))
    if darcy_number_value > 0.0:
        transverse_sizes_array = np.array(transverse_sizes, dtype=float)
        L_characteristic = math.sqrt(transverse_sizes_array[0] * transverse_sizes_array[1])
        alpha_max_computed = nu_value / (L_characteristic**2 * darcy_number_value) * 10e6
    else:
        alpha_max_computed = float(_require(material_cfg, 'alpha_max', 'material'))

    props = MaterialProperties(
        nu=nu_value,
        alpha_max=float(alpha_max_computed),
        mma_init=float(_require(material_cfg, 'mma_init', 'material')),
        mma_dec=float(_require(material_cfg, 'mma_dec', 'material')),
        mma_inc=float(_require(material_cfg, 'mma_inc', 'material')),
        movlim=float(_require(material_cfg, 'movlim', 'material')),
        voluse=float(_require(material_cfg, 'voluse', 'material')),
        filter_radius=float(_require(material_cfg, 'filter_radius', 'material')),
        solid_area=float(_require(material_cfg, 'solid_area', 'material')),
        fluid_area=float(_require(material_cfg, 'fluid_area', 'material')),
        test_pd=float(_require(material_cfg, 'test_pd', 'material')),
        d_normalization=float(_require(material_cfg, 'd_normalization', 'material')),
        d0=float(_require(material_cfg, 'd0', 'material')),
        d1=float(optimization_cfg['dissPower_max']) if optimization_cfg.get('dissPower_max') is not None else float(material_cfg.get('d1', 4.0)),
        geo_dim=float(_require(material_cfg, 'geo_dim', 'material')),
        b1=float(_require(material_cfg, 'b1', 'material')),
        qu=float(_require(material_cfg, 'qu', 'material')),
        kf=float(_require(material_cfg, 'kf', 'material')),
        ks=float(_require(material_cfg, 'ks', 'material')),
        rhoc=float(_require(material_cfg, 'rhoc', 'material')),
        t_alpha=float(_require(material_cfg, 't_alpha', 'material')),
        solid_density_g_per_mm3=float(material_cfg.get('solid_density_g_per_mm3', SOLID_DENSITY_G_PER_MM3)),
        darcy_number=float(material_cfg.get('darcy_number', 0.0)),
        Texterior=float(_require(material_cfg, 'Texterior', 'material')),
        hconv=float(material_cfg.get('hconv', 10.0)),
    )

    optimisation = OptimizationSettings(
        unit=float(optimization_cfg.get('unit', 1.5)),
        wall=float(optimization_cfg.get('wall', 0.30)),
        epsilon=float(optimization_cfg.get('epsilon', 0.04)),
        spacing=float(optimization_cfg.get('spacing', 2.0)),
        bake_spacing=float(optimization_cfg.get('bake_spacing', 0.4)),
        kbound=float(optimization_cfg.get('kbound', 2.0)),
        mode=str(optimization_cfg.get('mode', 'heat')),
        meantT_max=optimization_cfg.get('meantT_max', None),
        dissPower_max=optimization_cfg.get('dissPower_max', None),
        mu_penalty=float(optimization_cfg.get('mu_penalty', 100.0)),
        mu_kmin=float(optimization_cfg.get('mu_kmin', 0.0)),
        am_theta=float(optimization_cfg.get('am_theta', 45.0)),
        am_P_bar=float(optimization_cfg.get('am_P_bar', 0.01)),
        mu_overhang=float(optimization_cfg.get('mu_overhang', 1.0)),
        no_overhang=bool(optimization_cfg.get('no_overhang', False)),
        pareto_enabled=bool(optimization_cfg.get('pareto_enabled', False)),
        pareto_weights=optimization_cfg.get('pareto_weights', None),
        pareto_n_weights=int(optimization_cfg.get('pareto_n_weights', 9)),
        pareto_iters_per_point=int(optimization_cfg.get('pareto_iters_per_point', 30)),
        pareto_warmstart=bool(optimization_cfg.get('pareto_warmstart', True)),
    )

    run = RunSettings(
        case=case,
        solver=solver,
        parallel=parallel,
        postprocess=postprocess,
        skip_clean=skip_clean,
        iters=iters,
    )

    return run, geometry, props, optimisation, inlet, outlet, turbulence, thermal


def _axis_index(axis: str) -> int:
    mapping = {'x': 0, 'y': 1, 'z': 2}
    if axis not in mapping:
        raise ValueError(f"Unsupported axis '{axis}'. Use x, y, or z.")
    return mapping[axis]


def _flow_axis_velocity_vector(flow_axis: str, inlet_face: str, magnitude: float = 0.1) -> tuple[float, float, float]:
    sign = 1.0 if inlet_face == 'min' else -1.0
    velocity = [0.0, 0.0, 0.0]
    velocity[_axis_index(flow_axis)] = sign * magnitude
    return tuple(velocity)


def _balanced_subdomains(n_subdomains: int) -> tuple[int, int, int]:
    if n_subdomains < 1:
        raise ValueError('number of subdomains must be >= 1')

    best = (n_subdomains, 1, 1)
    best_score = float('inf')
    for nx in range(1, n_subdomains + 1):
        if n_subdomains % nx:
            continue
        rem = n_subdomains // nx
        for ny in range(1, rem + 1):
            if rem % ny:
                continue
            nz = rem // ny
            triple = tuple(sorted((nx, ny, nz)))
            score = triple[2] - triple[0]
            if score < best_score:
                best_score = score
                best = (nx, ny, nz)
    return best


def _foam_header(location: str, object_name: str, class_name: str = 'dictionary') -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     |                                                 |
|   \\  /    A nd           |                                                 |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    location    "{location}";
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _axis_coords(start: float, length: float, window_start: float | None = None, window_size: float | None = None) -> list[float]:
    coords = [start, start + length]
    if window_start is not None and window_size is not None:
        window_min = start + window_start
        window_max = window_min + window_size
        if window_min < start - 1e-9 or window_max > start + length + 1e-9:
            raise ValueError('Inlet window must lie inside the face bounds')
        coords.extend([window_min, window_max])
    coords = sorted({round(coord, 12) for coord in coords})
    if len(coords) < 2:
        raise ValueError('Invalid coordinate partition')
    return [float(coord) for coord in coords]


def _split_cell_counts(total_cells: int, coords: list[float]) -> list[int]:
    segments = [coords[index + 1] - coords[index] for index in range(len(coords) - 1)]
    segment_count = len(segments)
    if segment_count == 1:
        return [total_cells]
    if total_cells < segment_count:
        raise ValueError(f'Cannot split {total_cells} cells across {segment_count} segments')

    base = np.ones(segment_count, dtype=int)
    remaining = total_cells - segment_count
    weights = np.array(segments, dtype=float)
    weights = weights / weights.sum()
    extras_float = weights * remaining
    extras = np.floor(extras_float).astype(int)
    base += extras
    leftover = remaining - int(extras.sum())
    if leftover > 0:
        order = np.argsort(-(extras_float - extras))
        for index in order[:leftover]:
            base[index] += 1
    if int(base.sum()) != total_cells:
        raise ValueError('Failed to allocate all block cells')
    return base.tolist()


def write_block_mesh_dict(system_dir: Path, geometry: BoxGeometry, inlet: InletSettings, outlet: OutletSettings) -> None:
    axis_names = ('x', 'y', 'z')
    flow_axis = geometry.flow_axis.lower()
    if flow_axis not in axis_names:
        raise ValueError("flow_axis must be one of 'x', 'y', or 'z'")

    flow_index = _axis_index(flow_axis)
    transverse_axes = [name for name in axis_names if name != flow_axis]
    if inlet.face not in {'min', 'max'}:
        raise ValueError("inlet.face must be 'min' or 'max'")

    axis_data: dict[str, dict[str, object]] = {}
    for axis_name in axis_names:
        idx = _axis_index(axis_name)
        start = geometry.origin_mm[idx]
        length = geometry.size_mm[idx]
        if axis_name == flow_axis:
            coords = [start, start + length]
            cell_counts = [geometry.cells[idx]]
        else:
            # For transverse axes, create subdivisions for both inlet and outlet windows
            local_idx = transverse_axes.index(axis_name)
            axis_start = start
            
            # Compute absolute coordinates for inlet window
            inlet_min = axis_start + inlet.window_origin_mm[local_idx]
            inlet_max = inlet_min + inlet.window_size_mm[local_idx]
            
            # Compute absolute coordinates for outlet window
            outlet_min = axis_start + outlet.window_origin_mm[local_idx]
            outlet_max = outlet_min + outlet.window_size_mm[local_idx]
            
            # Start with domain boundaries
            coords = [start, start + length]
            
            # Add window boundaries (both inlet and outlet)
            for coord in [inlet_min, inlet_max, outlet_min, outlet_max]:
                if start - 1e-9 < coord < start + length + 1e-9:
                    coords.append(coord)
            
            # Sort and deduplicate
            coords = sorted({round(coord, 12) for coord in coords})
            coords = [float(coord) for coord in coords]
            
            cell_counts = _split_cell_counts(geometry.cells[idx], coords)
        axis_data[axis_name] = {'coords': coords, 'cell_counts': cell_counts}

    x_coords = axis_data['x']['coords']
    y_coords = axis_data['y']['coords']
    z_coords = axis_data['z']['coords']
    x_counts = axis_data['x']['cell_counts']
    y_counts = axis_data['y']['cell_counts']
    z_counts = axis_data['z']['cell_counts']

    nx = len(x_coords)
    ny = len(y_coords)
    nz = len(z_coords)

    def vertex_id(ix: int, iy: int, iz: int) -> int:
        return ix + nx * (iy + ny * iz)

    def vertex_line(x_mm: float, y_mm: float, z_mm: float) -> str:
        return f"    ({x_mm / MM_TO_MESH_UNIT:.6g} {y_mm / MM_TO_MESH_UNIT:.6g} {z_mm / MM_TO_MESH_UNIT:.6g})"

    def face_line(indices: tuple[int, int, int, int]) -> str:
        return f"            ({indices[0]} {indices[1]} {indices[2]} {indices[3]})"

    patches: dict[str, list[str]] = {'inlet': [], 'outlet': [], 'wall': [], 'force': [], 'sym': []}

    inlet_window_ranges: dict[str, tuple[float, float]] = {}
    for local_index, axis_name in enumerate(transverse_axes):
        axis_start = geometry.origin_mm[_axis_index(axis_name)]
        window_min = axis_start + inlet.window_origin_mm[local_index]
        window_max = window_min + inlet.window_size_mm[local_index]
        inlet_window_ranges[axis_name] = (window_min, window_max)

    outlet_window_ranges: dict[str, tuple[float, float]] = {}
    for local_index, axis_name in enumerate(transverse_axes):
        axis_start = geometry.origin_mm[_axis_index(axis_name)]
        window_min = axis_start + outlet.window_origin_mm[local_index]
        window_max = window_min + outlet.window_size_mm[local_index]
        outlet_window_ranges[axis_name] = (window_min, window_max)

    def face_is_inside_inlet(ix: int, iy: int, iz: int) -> bool:
        segment_indices = {'x': ix, 'y': iy, 'z': iz}
        for axis_name in transverse_axes:
            coords = axis_data[axis_name]['coords']
            seg_min = coords[segment_indices[axis_name]]
            seg_max = coords[segment_indices[axis_name] + 1]
            window_min, window_max = inlet_window_ranges[axis_name]
            if seg_min < window_min - 1e-9 or seg_max > window_max + 1e-9:
                return False
        return True

    def face_is_inside_outlet(ix: int, iy: int, iz: int) -> bool:
        segment_indices = {'x': ix, 'y': iy, 'z': iz}
        for axis_name in transverse_axes:
            coords = axis_data[axis_name]['coords']
            seg_min = coords[segment_indices[axis_name]]
            seg_max = coords[segment_indices[axis_name] + 1]
            window_min, window_max = outlet_window_ranges[axis_name]
            if seg_min < window_min - 1e-9 or seg_max > window_max + 1e-9:
                return False
        return True

    content = _foam_header('system', 'blockMeshDict')
    content += "\nconvertToMeters 0.01;\n\nvertices\n(\n"
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                content += vertex_line(x, y, z) + "\n"
    content += ");\n\nblocks\n(\n"

    for ix in range(len(x_counts)):
        for iy in range(len(y_counts)):
            for iz in range(len(z_counts)):
                v000 = vertex_id(ix, iy, iz)
                v100 = vertex_id(ix + 1, iy, iz)
                v110 = vertex_id(ix + 1, iy + 1, iz)
                v010 = vertex_id(ix, iy + 1, iz)
                v001 = vertex_id(ix, iy, iz + 1)
                v101 = vertex_id(ix + 1, iy, iz + 1)
                v111 = vertex_id(ix + 1, iy + 1, iz + 1)
                v011 = vertex_id(ix, iy + 1, iz + 1)
                cell_counts = f"({x_counts[ix]} {y_counts[iy]} {z_counts[iz]})"
                content += f"    hex ({v000} {v100} {v110} {v010} {v001} {v101} {v111} {v011})\n"
                content += f"    {cell_counts}\n"
                content += "    simpleGrading (1 1 1)\n\n"

                segment_indices = {'x': ix, 'y': iy, 'z': iz}
                max_indices = {'x': len(x_counts) - 1, 'y': len(y_counts) - 1, 'z': len(z_counts) - 1}
                face_vertices = {
                    'x': {'min': (v000, v010, v011, v001), 'max': (v100, v101, v111, v110)},
                    'y': {'min': (v000, v100, v101, v001), 'max': (v010, v110, v111, v011)},
                    'z': {'min': (v000, v100, v110, v010), 'max': (v001, v101, v111, v011)},
                }

                # Process transverse boundaries (walls, symmetry, force)
                for axis_name in transverse_axes:
                    if segment_indices[axis_name] not in {0, max_indices[axis_name]}:
                        continue
                    side = 'min' if segment_indices[axis_name] == 0 else 'max'
                    face = face_vertices[axis_name][side]
                    
                    if axis_name == transverse_axes[0]:
                        patch_name = 'wall'
                    else:
                        patch_name = 'sym' if side == 'min' else 'force'
                    
                    patches[patch_name].append(face_line(face))

                # Process flow-axis boundaries (inlet/outlet on min/max faces)
                # Handle both min face (inlet side) and max face (outlet side)
                if segment_indices[flow_axis] == 0:
                    # Min face of flow axis
                    face = face_vertices[flow_axis]['min']
                    if inlet.face == 'min':
                        patch_name = 'inlet' if face_is_inside_inlet(ix, iy, iz) else 'wall'
                    else:
                        patch_name = 'wall'
                    patches[patch_name].append(face_line(face))
                
                if segment_indices[flow_axis] == max_indices[flow_axis]:
                    # Max face of flow axis
                    face = face_vertices[flow_axis]['max']
                    if outlet.face == 'max':
                        patch_name = 'outlet' if face_is_inside_outlet(ix, iy, iz) else 'wall'
                    else:
                        patch_name = 'wall'
                    patches[patch_name].append(face_line(face))

    content += ");\n\nedges\n(\n);\n\nboundary\n(\n"
    content += "    inlet\n    {\n        type patch;\n        faces\n        (\n"
    content += "\n".join(patches['inlet']) + "\n"
    content += "        );\n    }\n    outlet\n    {\n        type patch;\n        faces\n        (\n"
    content += "\n".join(patches['outlet']) + "\n"
    content += "        );\n    }\n    wall\n    {\n        type wall;\n        faces\n        (\n"
    content += "\n".join(patches['wall']) + "\n"
    content += "        );\n    }\n    force\n    {\n        type patch;\n        faces\n        (\n"
    content += "\n".join(patches['force']) + "\n"
    content += "        );\n    }\n    sym\n    {\n        type symmetry;\n        faces\n        (\n"
    content += "\n".join(patches['sym']) + "\n"
    content += "        );\n    }\n);\n\nmergePatchPairs\n(\n);\n\n// flow axis: " + flow_axis + f" (index {flow_index})\n"
    content += f"// inlet face: {inlet.face}\n"
    content += f"// outlet face: {outlet.face}\n"
    content += f"// inlet window local origin mm: {inlet.window_origin_mm} size mm: {inlet.window_size_mm}\n"
    content += f"// outlet window local origin mm: {outlet.window_origin_mm} size mm: {outlet.window_size_mm}\n"
    content += "// wall: transverse pair 1, sym/force: transverse pair 2\n\n// ************************************************************************* //\n"
    _write_text(system_dir / 'blockMeshDict', content)


def write_control_dict(system_dir: Path, solver: str) -> None:
    content = _foam_header('system', 'controlDict')
    content += f"""
application     {solver};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  4;
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable true;
libs ( \"libOpenFOAM.so\" ) ;

// ************************************************************************* //
"""
    _write_text(system_dir / 'controlDict', content)


def write_decompose_par_dict(system_dir: Path, n_subdomains: int) -> None:
    nx, ny, nz = _balanced_subdomains(n_subdomains)
    content = _foam_header('system', 'decomposeParDict')
    content += f"""
numberOfSubdomains {n_subdomains};

method          simple;

simpleCoeffs
{{
    n               ({nx} {ny} {nz});
    delta           0.001;
}}

hierarchicalCoeffs
{{
    n               (1 1 1);
    delta           0.001;
    order           xyz;
}}

manualCoeffs
{{
    dataFile        "";
}}

distributed     no;

roots           ( );


// ************************************************************************* //
"""
    _write_text(system_dir / 'decomposeParDict', content)


def write_topo_set_dict(system_dir: Path, geometry: BoxGeometry) -> None:
    """Write a topoSetDict that creates the solver's expected cell zones.

    The current solver only checks for the existence of zone_test,
    zone_solid, and zone_fluid before optionally applying fixed-cell masks.
    For this case, those zones are intentionally created as empty zones so the
    mesh contains the expected names without overriding the optimizer field.
    """
    domain_max_mm = max(
        geometry.origin_mm[index] + geometry.size_mm[index]
        for index in range(3)
    )
    box_min = (domain_max_mm + 10.0) / MM_TO_MESH_UNIT
    box_max = (domain_max_mm + 11.0) / MM_TO_MESH_UNIT

    content = _foam_header('system', 'topoSetDict')
    content += f"""
actions
(
    {{
        name    zone_testCellSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({box_min:.6g} {box_min:.6g} {box_min:.6g}) ({box_max:.6g} {box_max:.6g} {box_max:.6g});
    }}
    {{
        name    zone_test;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     zone_testCellSet;
    }}

    {{
        name    zone_solidCellSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({box_min:.6g} {box_min:.6g} {box_min:.6g}) ({box_max:.6g} {box_max:.6g} {box_max:.6g});
    }}
    {{
        name    zone_solid;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     zone_solidCellSet;
    }}

    {{
        name    zone_fluidCellSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({box_min:.6g} {box_min:.6g} {box_min:.6g}) ({box_max:.6g} {box_max:.6g} {box_max:.6g});
    }}
    {{
        name    zone_fluid;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     zone_fluidCellSet;
    }}
);

// ************************************************************************* //
"""
    _write_text(system_dir / 'topoSetDict', content)


def write_initial_velocity_field(case_dir: Path, geometry: BoxGeometry, inlet: InletSettings) -> None:
    inlet_velocity = _flow_axis_velocity_vector(geometry.flow_axis, inlet.face, inlet.velocity_magnitude)
    ux, uy, uz = inlet_velocity
    content = _foam_header('0', 'U', 'volVectorField')
    content += f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({ux:.12g} {uy:.12g} {uz:.12g});
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
    }}
    force
    {{
        type            noSlip;
    }}
    wall
    {{
        type            noSlip;
    }}

    sym
    {{
        type            symmetry;
    }}

}}

// ************************************************************************* //
"""
    _write_text(case_dir / '0' / 'U', content)


def write_initial_temperature_field(case_dir: Path, inlet: InletSettings, thermal: ThermalSettings, outlet: OutletSettings) -> None:
    inlet_temp = inlet.temperature
    content = _foam_header('0', 'T', 'volScalarField')
    content += f"""
dimensions      [0 0 0 1 0 0 0];

internalField   uniform {thermal.initial_temperature:.12g};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {inlet_temp:.12g};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    wall
    {{
        type            zeroGradient;
    }}
    force
    {{
        type            zeroGradient;
    }}
    sym
    {{
        type            symmetry;
    }}

}}

// ************************************************************************* //
"""
    _write_text(case_dir / '0' / 'T', content)


def write_initial_adjoint_temperature_field(case_dir: Path) -> None:
    """Write Tb initial conditions with the correct adjoint BCs.

    The adjoint heat equation uses -div(-phi, Tb) which transports Tb backward
    (outlet → inlet). The outlet is therefore the adjoint inflow boundary and
    must have Tb = 0 (no adjoint signal arriving from downstream). The inlet
    is the adjoint outflow boundary and uses zeroGradient (natural outflow).
    """
    content = _foam_header('0', 'Tb', 'volScalarField')
    content += """
dimensions      [0 0 0 1 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    wall
    {
        type            zeroGradient;
    }
    force
    {
        type            zeroGradient;
    }
    sym
    {
        type            symmetry;
    }

}

// ************************************************************************* //
"""
    _write_text(case_dir / '0' / 'Tb', content)


def write_transport_properties(constant_dir: Path, props: MaterialProperties, opt1 = 1, opt2 = 1) -> None:
    content = _foam_header('constant', 'transportProperties')
    content += f"""
transportModel  Newtonian;
// fluid
nu                           nu [0 2 -1 0 0 0 0] {props.nu:.12g};
alphaMax               alphaMax [0 0 -1 0 0 0 0] {props.alpha_max:.12g};
alphamax               alphamax [0 0 -1 0 0 0 0] {props.alpha_max:.12g};

// opt
raa0                   {props.mma_init:.12g};
mma_init               {props.mma_init:.12g};
mma_dec                {props.mma_dec:.12g};
mma_inc                {props.mma_inc:.12g};
movlim                 {props.movlim:.12g};

voluse                 {props.voluse:.12g};
filter_Radius          {props.filter_radius:.12g};
solid_area             {props.solid_area:.12g};
fluid_area             {props.fluid_area:.12g};
test_PD                {props.test_pd:.12g};

D_normalization        {props.d_normalization:.12g};
D0                     {props.d0:.12g};
D1                     {props.d1:.12g};
geo_dim                {props.geo_dim:.12g};

b1                     b1 [0 2 -2 -2 0 0 0] {props.b1:.12g};
qu                     {props.qu:.12g};
opt1                   {opt1};
opt2                   {opt2};


// ************************************************************************* //
"""
    _write_text(constant_dir / 'transportProperties', content)


def write_thermal_properties(constant_dir: Path, props: MaterialProperties) -> None:
    content = _foam_header('constant', 'thermalProperties')
    content += f"""
kf                           kf [1 1 -3 -1 0 0 0] {props.kf:.12g};
ks                           ks [1 1 -3 -1 0 0 0] {props.ks:.12g};
rhoc                       rhoc [1 -1 -2 -1 0 0 0] {props.rhoc:.12g};

Talpha                   Taplha [0 0 0 -1 0 0 0] {props.t_alpha:.12g};

Texterior                 Texterior [0 0 0 1 0 0 0] {props.Texterior:.12g};

hconv                      hconv [1 0 -3 -1 0 0 0] {props.hconv:.12g};


// ************************************************************************* //
"""
    _write_text(constant_dir / 'thermalProperties', content)


def write_turbulence_properties(constant_dir: Path, props: TurbulenceProperties) -> None:
    content = _foam_header('constant', 'turbulenceProperties')
    content += f"""
simulationType {props.simulation_type};
//simulationType RAS;

RAS
{{
    RASModel        {props.ras_model};

    turbulence      {props.turbulence};

    printCoeffs     {props.print_coeffs};
}}


// ************************************************************************* //
"""
    _write_text(constant_dir / 'turbulenceProperties', content)


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_case(case_dir: Path, geometry: BoxGeometry, inlet: InletSettings, outlet: OutletSettings, props: MaterialProperties, turbulence: TurbulenceProperties, thermal: ThermalSettings, solver: str,
                 n_subdomains: int) -> None:
    print(f"Preparing case at {case_dir}")
    clean_case(case_dir, dry_run=False)

    for extra in (case_dir / 'constant' / 'polyMesh', case_dir / 'latest_fluid_state'):
        if extra.exists():
            shutil.rmtree(extra)
    for proc_dir in case_dir.glob('processor*'):
        if proc_dir.is_dir():
            shutil.rmtree(proc_dir)

    write_block_mesh_dict(case_dir / 'system', geometry, inlet, outlet)
    write_topo_set_dict(case_dir / 'system', geometry)
    write_initial_velocity_field(case_dir, geometry, inlet)
    write_initial_temperature_field(case_dir, inlet, thermal, outlet)
    write_initial_adjoint_temperature_field(case_dir)
    write_control_dict(case_dir / 'system', solver)
    write_decompose_par_dict(case_dir / 'system', n_subdomains=n_subdomains)
    write_transport_properties(case_dir / 'constant', props)
    write_thermal_properties(case_dir / 'constant', props)
    write_turbulence_properties(case_dir / 'constant', turbulence)


def build_optimizer(case_dir: Path, geometry: BoxGeometry, run: RunSettings, optimisation: OptimizationSettings,
                    material: MaterialProperties) -> GyroidRBFOptimizer:
    ox, oy, oz = geometry.origin_mm
    sx, sy, sz = geometry.size_mm
    opt_min = np.array([ox, oy, oz], dtype=float)
    opt_max = np.array([ox + sx, oy + sy, oz + sz], dtype=float)
    func_callback = lambda optvar : write_transport_properties(case_dir / 'constant', material, opt1=optvar[0], opt2=optvar[1])

    return GyroidRBFOptimizer(
        case_dir=case_dir,
            k_base=2.0 * math.pi / optimisation.unit,
            wall_thickness=optimisation.wall,
            epsilon=optimisation.epsilon,
            control_spacing=optimisation.spacing,
            k_amp_bound=optimisation.kbound,
            bake_spacing=optimisation.bake_spacing,
            solver=run.solver,
            n_procs=run.parallel,
            of_binary=run.postprocess,
        opt_bounds_min=opt_min,
        opt_bounds_max=opt_max,
            am_theta_max=math.radians(optimisation.am_theta),
            am_P_bar=optimisation.am_P_bar,
            am_mu_overhang=optimisation.mu_overhang,
            use_overhang=not optimisation.no_overhang,
            mode='pareto' if optimisation.pareto_enabled else optimisation.mode,
            target_meanT=optimisation.meantT_max,
            target_disspower=optimisation.dissPower_max,
            mu_mass=optimisation.mu_penalty,
            solid_density_g_per_mm3=material.solid_density_g_per_mm3,
            func_callback=func_callback,
            opt1 = 1,
            opt2 = 1,
            Texterior=material.Texterior,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a box case and launch the gyroid optimizer from YAML config.')
    parser.add_argument('--config', default='gyroid_case_config.yaml', help='YAML configuration file relative to this script')
    parser.add_argument('--write-config-template', action='store_true', help='Write a commented starter config and exit')
    parser.add_argument('--case', default=None, help='Override the case directory from the config')
    parser.add_argument('--solver', default=None, help='Override the solver executable from the config')
    parser.add_argument('--parallel', type=int, default=None, help='Override the core count from the config')
    parser.add_argument('--postprocess', default=None, help='Override the postProcess binary from the config')
    parser.add_argument('--skip-clean', action='store_true', help='Skip the cleanup step before preparing the case')
    parser.add_argument('--iters', type=int, default=None, help='Override the iteration count from the config')
    parser.add_argument('--load-ctrl', default=None, metavar='FILE', help='Warm-start from a previous gyroid_ctrl_pts_*.txt file')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = (script_dir / args.config).resolve()



    config = load_config(config_path)
    run, geometry, props, optimisation, inlet, outlet, turbulence, thermal = resolve_settings(config, args)

    case_dir = (script_dir / run.case).resolve()
    if not case_dir.is_dir():
        raise SystemExit(f'ERROR: case directory not found: {case_dir}')

    if not run.skip_clean:
        prepare_case(case_dir, geometry, inlet, outlet, props, turbulence, thermal, run.solver, max(1, run.parallel))
    else:
        write_block_mesh_dict(case_dir / 'system', geometry, inlet, outlet)
        write_topo_set_dict(case_dir / 'system', geometry)
        write_initial_velocity_field(case_dir, geometry, inlet)
        write_initial_temperature_field(case_dir, inlet, thermal, outlet)
        write_control_dict(case_dir / 'system', run.solver)
        write_decompose_par_dict(case_dir / 'system', n_subdomains=max(1, run.parallel))
        write_transport_properties(case_dir / 'constant', props)
        write_thermal_properties(case_dir / 'constant', props)
        write_turbulence_properties(case_dir / 'constant', turbulence)

    run_command(['blockMesh', '-case', str(case_dir)], cwd=case_dir)
    run_command(['topoSet', '-case', str(case_dir)], cwd=case_dir)
    run_command([run.postprocess, '-func', 'writeCellCentres', '-time', '0', '-case', str(case_dir)], cwd=case_dir)

    if run.parallel > 1:
        run_command(['decomposePar', '-force', '-case', str(case_dir)], cwd=case_dir)

    load_ctrl = Path(args.load_ctrl).resolve() if args.load_ctrl else None
    optimiser = build_optimizer(case_dir, geometry, run, optimisation, props)

    if optimisation.pareto_enabled:
        explorer = ParetoExplorer(
            optimizer           = optimiser,
            weights             = optimisation.pareto_weights,
            n_weights           = optimisation.pareto_n_weights,
            n_iters_per_point   = optimisation.pareto_iters_per_point,
            warmstart_from_prev = optimisation.pareto_warmstart,
        )
        x0 = None
        if load_ctrl is not None:
            x0 = np.loadtxt(load_ctrl)[:, 3:6].ravel()
        explorer.run(x0=x0)
    else:
        optimiser.run(n_iters=run.iters, load_ctrl=load_ctrl)


if __name__ == '__main__':
    main()