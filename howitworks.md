## First-time `wmake` in `src_TF`

For the first build, go to `src_TF` in the terminal and run `wmake` once so the optimiser is compiled. To do this: type this in the terminal:

```bash
cd 3Dheatsink_gyroid/src_TF
wmake
```

If the build succeeds, the optimiser is ready to run with a YAML configuration file.

## Clean the directory from a previous run
In 3Dheatsink_gyroid, run:
```bash
python clean_for_gyroid.py
```

## Using the YAML file to run the optimiser

The YAML file is the optimiser's runtime configuration. It usually defines:

- the case or input path
- optimiser type and parameters
- iteration or stopping settings
- output directory

Run the optimiser with the YAML file passed as a config argument, for example:

```bash
cd 3Dheatsink_gyroid
python gyroid_case_wrapper.py --config gyroid_case_config.yaml
```

Read the yaml file to choose the correct inputs

## TODO:
- Truly parallelize
- Reduce conductivity such that temperature is less homogeneous through heat exchanger and the geometry actually influences oulet temperature
- Better optimizer that actually performs more iterations in comparison to evaluations.
- Create opt3 that controls Heaviside thickness
- Remove the thickness constraint, instead use the wall thickness parameter directly
- Remove heat source Q, add surface heating
- Maybe a new fsens that is for minmizing outlet temp instead of meanT
- Make other geometries possible
- Make robust
- Make comparison cases (i.e. just a beam in the middle, or un deformed Gyroid) for thermal performance
- Adjust penalty size
- Optimization point count and positioning
- pyMMAopt to Dockerfile
- Check if this is faster:
```python
rbf = RBFInterpolator(
    ctrl_pts_mm,
    np.eye(n_ctrl),
    kernel="thin_plate_spline",
    degree=1
)
W = rbf(cell_centers_mm)
```