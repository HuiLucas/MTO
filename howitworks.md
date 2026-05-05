## First-time `wmake` in `src_TF`

For the first build, go to `src_TF` in the terminal and run `wmake` once so the optimiser is compiled. To do this: type this in the terminal:

```bash
cd 3Dheatsink_gyroid/src_TF
wmake
```

If the build succeeds, the optimiser is ready to run with a YAML configuration file.

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