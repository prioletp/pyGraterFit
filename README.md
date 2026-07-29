# pyGraterFit

![pyGraterFit banner](docs/assets/pyGraterFit-banner.png)

Fitting tools for [`pyGrater`](https://github.com/prioletp/pyGrater) dust-disk
models.

`pyGraterFit` wraps the SED, image, and interferometric modelling machinery in
`pyGrater` with user-facing fitters for deterministic optimization, MCMC, and
nested sampling. It is designed for debris-disk modelling where a model may
contain one ring, several rings, one dust composition, or several compositions
per ring.

## What it provides

- Single-ring SED fitters.
- Multi-ring additive SED fitters.
- Multi-composition fitters with one total dust normalization per ring plus
  composition fractions.
- SciPy optimization, `emcee` MCMC, and `dynesty` nested sampling interfaces.
- Correlated-flux fitters that combine pyGrater SEDs with analytical ring
  visibilities.
- MCMC and nested-sampling plotting helpers.
- Restartable MCMC backends and checkpointed nested-sampling runs.

## Installation

For development or editable use:

```bash
git clone https://github.com/prioletp/pyGraterFit.git
cd pyGraterFit
pip install -e .
```

To install directly from GitHub:

```bash
pip install git+https://github.com/prioletp/pyGraterFit.git
```

If you previously installed the old package name in editable mode, remove it
first:

```bash
pip uninstall -y fitters_for_pyGrater fitters-for-pyGrater
pip install -e .
```

## Quick start

```python
from pyGraterFit import SingleRingSEDScipyFitter

fitter = SingleRingSEDScipyFitter(
    grain=grain,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params={
        "r0": (0.1, 10.0),
        "alphaout": (-10.0, -1.0),
        "a_min": (1e-7, 1e-4),
        "a_max": 1e-3,
        "kappa": (1.0, 5.0),
        "h0": lambda p: 0.05 * p["r0"],
        "A_norm": (1e28, 1e38),
    },
)

result = fitter.fit(maxiter=1000)
fitter.summary()
fitter.plot_best_fit().savefig("best_fit.png", dpi=150)
```

## Which fitter should I use?

| Model family | SciPy | MCMC | Nested sampling |
| --- | --- | --- | --- |
| Single ring, one composition, SED | `SingleRingSEDScipyFitter` | `SingleRingSEDMCMCFitter` | `SingleRingSEDNestedFitter` |
| Multi-ring additive SED | `MultiRingSEDScipyFitter` | `MultiRingSEDMCMCFitter` | `MultiRingSEDNestedFitter` |
| Multi-ring correlated flux, optional SED | `MultiRingSEDCorrelatedFluxFitter` | `MultiRingSEDCorrelatedFluxFitter` | `MultiRingSEDCorrelatedFluxNestedFitter` |
| Multi-ring SED plus squared visibilities from images | - | `MultiRingSEDVisibilityMCMCFitter` | - |

Prefer top-level imports:

```python
from pyGraterFit import MultiRingSEDNestedFitter
```

instead of importing from internal submodules.

## Parameter dictionaries

Model parameters are described with ordinary Python dictionaries:

```python
params = {
    "r0": (0.1, 10.0),          # fitted between 0.1 and 10 au
    "alphaout": (-10.0, -1.0),  # fitted
    "a_min": (1e-7, 1e-4),      # fitted
    "a_max": 1e-3,              # fixed
    "kappa": (1.0, 5.0),        # fitted
    "h0": lambda p: 0.05 * p["r0"],  # dependent parameter
    "A_norm": (1e28, 1e38),     # fitted dust normalization
}
```

The conventions are:

- a scalar is fixed;
- a two-value tuple is fitted;
- a callable is evaluated from the other parameters.

Positive scale parameters such as `r0`, `a_min`, `A_norm`, and `M_tot` are
sampled or optimized in log-space by default where the fitter supports
log-space coordinates.

## Single-ring SED fits

Use the single-ring fitters when the model has one physical ring and one grain
composition.

### SciPy

```python
from pyGraterFit import SingleRingSEDScipyFitter

fitter = SingleRingSEDScipyFitter(
    grain=grain,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params=params,
)

result = fitter.fit(maxiter=1000)
print(fitter.format_total_mass())
```

### MCMC

```python
from pyGraterFit import SingleRingSEDMCMCFitter

fitter = SingleRingSEDMCMCFitter(
    grain=grain,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params=params,
)

fitter.fit(maxiter=1000)
fitter.run_mcmc(
    nwalkers=40,
    nsteps=5000,
    burn_in=1000,
    thin=1,
    backend_path="single_ring_backend.h5",
)
fitter.mcmc_summary()
```

### Nested sampling

```python
from pyGraterFit import SingleRingSEDNestedFitter

fitter = SingleRingSEDNestedFitter(
    grain=grain,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params=params,
)

fitter.run(
    npoints=500,
    method="multi",
    sample="rslice",
    dynamic=True,
    dlogz=0.1,
    checkpoint_file="single_ring_nested.checkpoint",
)
fitter.summary()
```

## Multi-ring, one-composition SED fits

If all rings use the same grain composition, each ring can be treated as one
additive component:

```python
components = {
    "inner_ring": grain,
    "outer_ring": grain,
}

params_by_component = {
    "inner_ring": {
        "r0": (0.1, 3.0),
        "alphaout": (-10.0, -1.0),
        "a_min": (1e-7, 1e-4),
        "a_max": 1e-3,
        "kappa": (1.0, 5.0),
        "h0": lambda p: 0.05 * p["r0"],
        "A_norm": (1e28, 1e38),
    },
    "outer_ring": {
        "r0": (3.0, 30.0),
        "alphaout": (-10.0, -1.0),
        "a_min": (1e-7, 1e-4),
        "a_max": 1e-3,
        "kappa": (1.0, 5.0),
        "h0": lambda p: 0.05 * p["r0"],
        "A_norm": (1e28, 1e38),
    },
}
```

The same component dictionaries can be used with SciPy, MCMC, or nested
sampling:

```python
from pyGraterFit import MultiRingSEDScipyFitter

fitter = MultiRingSEDScipyFitter(
    components=components,
    params_by_component=params_by_component,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    method="Nelder-Mead",
    use_log_params=True,
)

result = fitter.fit(maxiter=1000)
print(fitter.format_component_mass_abundances())
```

For nested sampling, replace the class and call `run`:

```python
from pyGraterFit import MultiRingSEDNestedFitter

fitter = MultiRingSEDNestedFitter(
    components=components,
    params_by_component=params_by_component,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    use_log_params=True,
)

fitter.run(npoints=800, method="multi", sample="rslice", dynamic=True)
```

## Multi-ring, multi-composition SED fits

For the common case where every ring contains the same set of dust
compositions, pass:

- `materials`: one grain object per composition;
- `ring_params`: one parameter dictionary per physical ring.

The fitter expands this into one backend component per ring/composition pair:

```text
inner_ring.olivine
inner_ring.silicate
outer_ring.olivine
outer_ring.silicate
```

This does not merge the material physics. Each component still has its own
`Grain` object, opacities, temperatures, sublimation behaviour, and emitted
flux. The shared ring parameters only mean that all materials in the same
physical ring use the same fitted geometry and size-distribution parameters.

```python
materials = {
    "olivine": olivine_grain,
    "silicate": silicate_grain,
}

ring_params = {
    "inner_ring": {
        "r0": (0.1, 3.0),
        "alphaout": (-10.0, -1.0),
        "a_min": (1e-7, 1e-4),
        "a_max": 1e-3,
        "kappa": (1.0, 5.0),
        "h0": lambda p: 0.05 * p["r0"],
    },
    "outer_ring": {
        "r0": (3.0, 30.0),
        "alphaout": (-10.0, -1.0),
        "a_min": (1e-7, 1e-4),
        "a_max": 1e-3,
        "kappa": (1.0, 5.0),
        "h0": lambda p: 0.05 * p["r0"],
    },
}

from pyGraterFit import MultiRingSEDNestedFitter

fitter = MultiRingSEDNestedFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    normalization_range=(1e28, 1e38),
)

fitter.run(npoints=1000, method="multi", sample="rslice", dynamic=True)
print(fitter.format_component_mass_abundances())
```

### How grouped composition normalization works

For a ring with several compositions, `normalization_range` is the prior range
for the ring's total normalization, not for each material independently. The
fitter samples:

- one `A_norm_total` per ring;
- `N - 1` fraction coordinates for `N` compositions.

It then sends ordinary per-material normalizations to pyGrater:

```text
A_norm(material) = fraction(material) × A_norm_total
```

The fractions are constructed so they are positive and sum to one. This keeps
wide normalization priors usable in nested sampling while preserving the
physical pyGrater model evaluation.

### Irregular composition sets

The `materials=...`, `ring_params=...` constructor assumes a rectangular grid:
every material is present in every ring. If your model is not rectangular, use
explicit component dictionaries.

This is useful when:

- the inner ring contains olivine and carbon, but the outer ring contains only
  olivine;
- one ring has a different material list from another ring;
- a component should be an empirical/template SED rather than a pyGrater grain;
- you need custom normalization groups;
- one component should have independent geometry rather than sharing ring
  parameters.

```python
components = {
    "inner_ring.olivine": olivine_grain,
    "inner_ring.carbon": carbon_grain,
    "outer_ring.olivine": olivine_grain,
}

params_by_component = {
    "inner_ring.olivine": {**inner_ring_params, "A_norm": (1e28, 1e38)},
    "inner_ring.carbon": {**inner_ring_params, "A_norm": (1e28, 1e38)},
    "outer_ring.olivine": {**outer_ring_params, "A_norm": (1e28, 1e38)},
}

component_groups = {
    "inner_ring.olivine": "inner_ring",
    "inner_ring.carbon": "inner_ring",
    "outer_ring.olivine": "outer_ring",
}

fitter = MultiRingSEDNestedFitter(
    components=components,
    params_by_component=params_by_component,
    component_groups=component_groups,
    group_shared_parameter_names=tuple(inner_ring_params),
    normalization_mode="group_total_fraction",
    normalization_groups=component_groups,
    normalization_total_ranges={"inner_ring": (1e28, 1e38)},
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
)
```

Only groups with two or more components receive fraction coordinates. A
single-component ring keeps its ordinary fitted `A_norm`.

## Correlated-flux fits

Correlated-flux fitters combine pyGrater fluxes with analytical ring
visibilities. They can fit correlated fluxes alone or correlated fluxes plus an
optional SED.

```python
from pyGraterFit import MultiRingSEDCorrelatedFluxNestedFitter

fitter = MultiRingSEDCorrelatedFluxNestedFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    correlated_flux=correlated_flux,
    sed_wavelengths=sed_wavelengths,
    sed_fluxes=sed_fluxes,
    sed_flux_errors=sed_flux_errors,
    stellar_angular_diameter_mas=stellar_diameter_mas,
    normalization_range=(1e28, 1e38),
    visibility_model="gaussian_ring",
)

fitter.run(npoints=800, method="multi", sample="rslice", dynamic=True)
```

For one-composition multi-ring correlated-flux fits, pass a single-entry
`materials` dictionary or use explicit `components` and
`params_by_component`.

## Image-based squared-visibility fits

For V2 data calculated from pyGrater images, use the same compact multi-ring,
multi-composition constructor as the additive SED fitters. Each ring/material
pair is still evaluated as its own physical component, but materials in the
same ring share geometry by default and only differ in their dust
normalization.

```python
from pyGraterFit import MultiRingSEDVisibilityMCMCFitter

fitter = MultiRingSEDVisibilityMCMCFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    sed_wavelengths=sed_wavelengths,
    sed_fluxes=sed_fluxes,
    sed_flux_errors=sed_flux_errors,
    vis2=vis2_data,
    image_settings={"nx": 128, "ny": 128, "FOV_AU": 20.0, "nl": 101},
    normalization_range=(1e28, 1e38),
    stellar_visibility_model="uniform_disk",
    stellar_angular_diameter_mas=stellar_diameter_mas,
)

fitter.fit(maxiter=500)
fitter.run_mcmc(
    n_steps=5000,
    backend_path="sed_vis2_backend.h5",
    initialization="best_fit",
)
```

Use explicit `components`, `params_by_component`, and `component_groups` only
for irregular models, for example when a material appears in one ring but not
another. Keep `stellar_angular_diameter_mas` in its dedicated constructor
argument instead of putting it inside `ring_params`.

## Plotting and command-line helpers

The package installs a few convenience scripts:

```bash
pyGraterFit-plot-mcmc
pyGraterFit-plot-all-mcmc
pyGraterFit-plot-nested
pyGraterFit-recover-backend
```

The fitters also expose plotting methods directly, for example:

```python
fitter.plot_best_fit().savefig("best_fit.png", dpi=150)
fitter.corner_plot(max_samples=5000).savefig("corner.png", dpi=150)
```

## Dust-mass output

For single-component fits:

```python
print(fitter.format_total_mass())
```

For additive multi-ring or multi-composition fits:

```python
print(fitter.format_component_mass_abundances())
```

The additive mass table reports:

- each component mass;
- each ring total mass;
- each composition percent of its ring;
- each composition percent of all fitted dust.

## Examples

Verbose templates are available in [`examples/`](examples/):

- `single_sed_scipy.py`
- `single_sed_mcmc.py`
- `single_sed_nested.py`
- `multi_ring_single_composition_sed_scipy.py`
- `multi_ring_single_composition_sed_nested.py`
- `additive_sed_mcmc.py`
- `additive_sed_nested.py`

Copy an example into your target-specific analysis folder and replace the toy
data arrays with your observations.

## Development checks

```bash
pip install -e ".[dev]"
pytest -q
```

Generated files such as `.DS_Store`, caches, build outputs, and `*.egg-info/`
directories are ignored and should not be committed.
