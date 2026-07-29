# Examples

These files are intentionally verbose templates. They are meant to be copied
into a target-specific folder and edited, not imported as a library.

Each script shows a different fitting workflow:

- `single_sed_scipy.py`: one-ring SED fit with deterministic SciPy
  optimization. Use this first to check that the data, star, grain, and model
  parameters are wired correctly.
- `single_sed_mcmc.py`: one-ring SED fit with emcee MCMC, persistent HDF5
  backend storage, and restart examples.
- `single_sed_nested.py`: one-ring SED fit with dynesty nested sampling,
  checkpointing, resume/load modes, and nested diagnostic plots.
- `multi_ring_single_composition_sed_scipy.py`: deterministic SciPy fit for
  several rings that all use the same grain composition.
- `multi_ring_single_composition_sed_nested.py`: dynesty version of the
  multi-ring, single-composition SED fit.
- `additive_sed_mcmc.py`: multi-ring/multi-composition additive SED fit using
  the friendly `materials=...` and `ring_params=...` constructor with grouped
  `A_norm_total + fractions` normalization.
- `additive_sed_nested.py`: nested-sampling version of the additive
  multi-ring/multi-composition workflow, including dynesty controls and
  checkpointing.

## How to read these examples

The examples use small placeholder data arrays so the file structure is clear.
For a real target, replace:

- `star_name`;
- grain compositions;
- wavelength/flux/error arrays;
- ring parameter ranges;
- output filenames.

For production fits, always check:

1. observational errors are positive and finite;
2. fitted scale parameters have physically sensible log-space ranges;
3. nested-sampling checkpoints are fresh after changing the parameterization;
4. the best likelihood is far above the finite rejection floor.

The main package README explains all developer options in more detail,
including `SAMPLE`, `METHOD`, grouped composition fractions, caching,
parallelization, checkpointing, and likelihood/evidence handling.

- `additive_sed_visibility_mcmc.py`: template for fitting image-based squared visibilities, optionally together with an SED, using the same `materials=...` and `ring_params=...` interface. Observed V2 wavelengths are associated with the nearest rendered image wavelength.

- `additive_sed_visibility_nested.py`: nested-sampling counterpart for image-based squared visibilities, with the same SED-optional workflow and checkpoint/resume/load sections.

- `additive_sed_correlated_flux_mcmc.py`: complete MCMC template for correlated fluxes alone or correlated fluxes plus an SED.

- `additive_sed_correlated_flux_nested.py`: nested-sampling counterpart for the correlated-flux fitter.

- `fov_additive_sed_mcmc.py`: complete template for fitting SED points with instrument-dependent field-of-view transmission.
