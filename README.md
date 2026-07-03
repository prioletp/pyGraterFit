# pyGraterFit

User-facing fitting tools for `pyGrater` dust-disk models.

`pyGraterFit` provides SciPy optimization, MCMC sampling, nested sampling, and
interferometric fitting interfaces around the physical SED/image machinery in
`pyGrater`.

## Installation

From a clone of this repository:

```bash
conda activate pyGrater
pip install -e .
```

## Recommended imports

Prefer the top-level imports:

```python
from pyGraterFit import SingleRingSEDScipyFitter
from pyGraterFit import SingleRingSEDMCMCFitter
from pyGraterFit import SingleRingSEDNestedFitter

from pyGraterFit import MultiRingSEDScipyFitter
from pyGraterFit import MultiRingSEDMCMCFitter
from pyGraterFit import MultiRingSEDNestedFitter
```

The older internal class names are kept only for compatibility with existing
scripts. New scripts should use the explicit `SingleRing...` and
`MultiRing...` names shown above.

## Parameter dictionaries

A parameter dictionary uses three conventions:

```python
params = {
    "r0": (0.1, 10.0),          # fitted uniformly between 0.1 and 10 au
    "alphaout": (-10.0, -1.0),  # fitted
    "a_min": (1e-7, 1e-4),      # fitted
    "a_max": 1e-3,              # fixed
    "kappa": (1.0, 5.0),        # fitted
    "h0": lambda p: 0.05 * p["r0"],  # dependent parameter
    "A_norm": (1e28, 1e38),     # fitted dust normalization
}
```

- A scalar is fixed.
- A two-value tuple is fitted.
- A callable is evaluated from the other parameters.

Positive scale parameters such as `r0`, `a_min`, `A_norm`, and `M_tot` are
sampled/optimized in log-space by default where the fitter supports log-space
coordinates.

## 1. Single-ring, one-composition SED fits

Use these fitters when the model contains one physical ring and one grain
composition.

### 1.1 SciPy optimization

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
fitter.summary()
print(fitter.format_total_mass())

fig = fitter.plot_best_fit()
fig.savefig("single_ring_scipy_best_fit.png", dpi=150)
```

### 1.2 MCMC

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
fitter.run_mcmc(nwalkers=40, nsteps=5000, burn_in=1000, thin=1)

fitter.mcmc_summary()
print(fitter.format_total_mass(use_mcmc_best=True))
```

### 1.3 Nested sampling

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
)

fitter.summary()
print(f"Total dust mass: {fitter.get_total_mass():.8g} Earth masses")
```

## 2. Multi-ring, one-composition SED fits

Use these fitters when the model contains several rings but the same dust
composition in each ring. In this case the normal constructor is still readable:
each ring is one additive component.

```python
components = {
    "inner_ring": grain,
    "outer_ring": grain,
}

inner_ring_params = {
    "r0": (0.1, 3.0),
    "alphaout": (-10.0, -1.0),
    "a_min": (1e-7, 1e-4),
    "a_max": 1e-3,
    "kappa": (1.0, 5.0),
    "h0": lambda p: 0.05 * p["r0"],
    "A_norm": (1e28, 1e38),
}

outer_ring_params = {
    "r0": (3.0, 30.0),
    "alphaout": (-10.0, -1.0),
    "a_min": (1e-7, 1e-4),
    "a_max": 1e-3,
    "kappa": (1.0, 5.0),
    "h0": lambda p: 0.05 * p["r0"],
    "A_norm": (1e28, 1e38),
}

params_by_component = {
    "inner_ring": inner_ring_params,
    "outer_ring": outer_ring_params,
}
```

### 2.1 SciPy optimization

```python
from pyGraterFit import MultiRingSEDScipyFitter

fitter = MultiRingSEDScipyFitter(
    components=components,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params_by_component=params_by_component,
    method="Nelder-Mead",
    use_log_params=True,
)

result = fitter.fit(maxiter=1000)
fitter.summary()
print(fitter.format_component_mass_abundances())

fig = fitter.plot_best_fit()
fig.savefig("multi_ring_one_composition_scipy_best_fit.png", dpi=150)
```

### 2.2 MCMC

```python
from pyGraterFit import MultiRingSEDMCMCFitter

fitter = MultiRingSEDMCMCFitter(
    components=components,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params_by_component=params_by_component,
    use_log_params=True,
)

fitter.fit(maxiter=1000)
fitter.run_mcmc(nwalkers=80, nsteps=5000, burn_in=1000, thin=1)

fitter.mcmc_summary()
print(fitter.format_component_mass_abundances())
```

### 2.3 Nested sampling

```python
from pyGraterFit import MultiRingSEDNestedFitter

fitter = MultiRingSEDNestedFitter(
    components=components,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
    params_by_component=params_by_component,
    use_log_params=True,
)

fitter.run(
    npoints=800,
    method="multi",
    sample="rslice",
    dynamic=True,
    dlogz=0.1,
)

fitter.summary()
print(fitter.format_component_mass_abundances())
```

## 3. Multi-ring, multi-composition SED fits

When each ring contains several dust compositions, use the same friendly
constructor as the one-composition case, but pass a `materials` dictionary and
a `ring_params` dictionary. In the backend, the fitter expands this into one
pyGrater component per ring/composition pair:

```text
inner_ring.olivine
inner_ring.silicate
outer_ring.olivine
outer_ring.silicate
```

This expansion does not change the physics: each ring/material component still
gets its own `Grain` object and its own pyGrater SED calculation. The shared
ring parameters only mean “use the same fitted geometric and size-distribution
parameters for all materials in this physical ring”; the material-dependent
temperatures, sublimation behaviour, opacities, and emitted fluxes remain
composition-specific.

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
```

### 3.1 SciPy optimization

```python
from pyGraterFit import MultiRingSEDScipyFitter

fitter = MultiRingSEDScipyFitter(
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

result = fitter.fit(maxiter=1000)
fitter.summary()
print(fitter.format_component_mass_abundances())
```

### 3.2 MCMC

```python
from pyGraterFit import MultiRingSEDMCMCFitter

fitter = MultiRingSEDMCMCFitter(
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

fitter.fit(maxiter=1000)
fitter.run_mcmc(nwalkers=120, nsteps=8000, burn_in=2000, thin=1)
fitter.mcmc_summary()
```

### 3.3 Nested sampling

```python
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

fitter.run(
    npoints=1000,
    method="multi",
    sample="rslice",
    dynamic=True,
    dlogz=0.1,
)

fitter.summary()
print(fitter.format_component_mass_abundances())
```

### 3.4 Non-rectangular and custom component models

The `materials=...`, `ring_params=...` constructor assumes a rectangular
ring/material grid: every listed material is present in every listed ring.
That is the most common debris-disk model and the easiest one to read.

For more unusual models, pass explicit `components` and
`params_by_component` dictionaries. This is useful when, for example:

- the inner ring contains olivine and carbon, but the outer ring contains only
  olivine;
- one ring has a different material list from another ring;
- one component is an empirical/template SED that should be added to physical
  pyGrater components;
- you want custom normalization groups that are not simply “one group per
  physical ring”;
- you want a component to have independent geometry instead of sharing all ring
  parameters with the other materials in that ring.

Example: inner ring with two compositions, outer ring with one composition:

```python
components = {
    "inner_ring.olivine": olivine_grain,
    "inner_ring.carbon": carbon_grain,
    "outer_ring.olivine": olivine_grain,
}

params_by_component = {
    "inner_ring.olivine": {
        **inner_ring_params,
        "A_norm": (1e28, 1e38),
    },
    "inner_ring.carbon": {
        **inner_ring_params,
        "A_norm": (1e28, 1e38),
    },
    "outer_ring.olivine": {
        **outer_ring_params,
        "A_norm": (1e28, 1e38),
    },
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
    normalization_total_ranges={
        "inner_ring": (1e28, 1e38),
    },
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=phase_function,
    wavelengths=wavelengths,
    fluxes=fluxes,
    fluxes_err=fluxes_err,
)
```

Here only `inner_ring` has more than one component, so only `inner_ring` gets
an `A_norm_total` plus fraction coordinates. The single-component `outer_ring`
keeps its ordinary fitted `outer_ring.olivine.A_norm`.

## Dust mass output

For single-component fits:

```python
print(fitter.format_total_mass())
```

For additive multi-ring fits:

```python
print(fitter.format_component_mass_abundances())
```

The additive mass table reports:

- each component mass;
- each ring total mass;
- each composition percent of its ring;
- each composition percent of all fitted dust.

## More examples

Small runnable templates are in the repository `examples/` directory.

HD113766-specific scripts are maintained separately from this package and can
use the same top-level `pyGraterFit` imports shown above.
