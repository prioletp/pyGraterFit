"""Additive multi-ring / multi-composition SED fit with MCMC.

This template shows the recommended compact interface for the common case:

    rings × grain materials

The fitter creates one pyGrater SED component per ring/material pair in the
backend, shares ring geometry among materials in the same ring, and samples
one total normalization plus composition fractions per ring.
"""

import numpy as np

from pyGraterFit import MultiRingSEDMCMCFitter
from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution


# ---------------------------------------------------------------------------
# 1. Star, compositions, and data
# ---------------------------------------------------------------------------

star = Star(star_name="HD113766")

# Add or remove materials here. The compact constructor automatically adds
# every material to every ring.
materials = {
    "olivine": Grain(composition="c_olivine_Fe_Poor"),
    "carbon": Grain(composition="aC_ACAR"),
}

wavelengths = np.array([8.0, 10.0, 12.0])
observed_flux = np.array([1.0, 1.2, 1.1])
flux_error = np.full_like(observed_flux, 0.1)


# ---------------------------------------------------------------------------
# 2. Ring parameters
# ---------------------------------------------------------------------------

def scale_height(parameters):
    """Keep h0 equal to 5% of the ring radius."""
    return 0.05 * parameters["r0"]


common_ring_parameters = {
    "h0": scale_height,
    "alphain": 10.0,
    "alphaout": (-10.0, -1.0001),
    "gamma": 2.0,
    "beta": 1.0,
    "itilt": 0.0,
    "PA": 90.0,
    "omega": 45.0,
    "a_min": (0.05e-6, 1e-4),
    "a_max": 1000e-6,
    "kappa": (1.0, 5.0),
    "N_sizes_integral": 100,
    "g": 0.0,
}

# Each top-level key is a ring. The compact constructor shares every parameter
# in one ring across all materials in that ring.
ring_params = {
    "inner_ring": {
        **common_ring_parameters,
        "r0": (0.01, 2.0),
    },
    "outer_ring": {
        **common_ring_parameters,
        "r0": (2.0, 100.0),
    },
}


# ---------------------------------------------------------------------------
# 3. Normalization parameterization
# ---------------------------------------------------------------------------

# In grouped normalization mode, this is the prior on each ring's TOTAL
# normalization, not an independent prior per material.
#
# For N materials, the fitter samples:
#   - A_norm_total
#   - N - 1 fraction_stick parameters
#
# It then constructs physical material normalizations:
#   A_material = fraction_material * A_norm_total
#
# This avoids prior draws where all materials independently receive enormous
# normalizations at the same time.
A_NORM_TOTAL_RANGE = (1e25, 1e38)


# ---------------------------------------------------------------------------
# 4. Build and run the additive fitter
# ---------------------------------------------------------------------------

fitter = MultiRingSEDMCMCFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    wavelengths=wavelengths,
    fluxes=observed_flux,
    fluxes_err=flux_error,
    normalization_range=A_NORM_TOTAL_RANGE,
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=300,
    parallel_components="auto",
    max_component_workers=4,
)

# Optional quick optimizer. This is often useful before MCMC because it gives a
# sensible starting point and catches wiring mistakes.
RUN_SCIPY_FIRST = True
if RUN_SCIPY_FIRST:
    fitter.fit(maxiter=300, verbose=True)

# Run MCMC from the best fit. For production, increase nsteps and inspect
# autocorrelation/acceptance diagnostics.
fitter.fit_then_mcmc(
    best_fit_values=fitter.best_params,
    nwalkers=max(32, 2 * fitter.ndim + 2),
    nsteps=2000,
    backend_path="additive_sed_mcmc_backend.h5",
    init="best_fit",
    best_fit_fwhm_frac=0.05,
)

fitter.save_results("additive_sed_mcmc_results.npz")
fitter.plot_corner("additive_sed_mcmc_corner.png")
fitter.plot_best_fit("additive_sed_mcmc_best_fit.png")


# ---------------------------------------------------------------------------
# 5. Explicit component interface, only if needed
# ---------------------------------------------------------------------------

# Use components=... and params_by_component=... only when the ring/material
# grid is not rectangular. For example, this is useful if the inner ring has
# olivine+carbon but the outer ring has only olivine, or if one component is an
# empirical template rather than a physical ring/material pair.
