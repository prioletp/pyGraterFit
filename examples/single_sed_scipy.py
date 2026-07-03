"""Single-component SED fit with deterministic SciPy optimization.

This is the simplest useful pyGrater fitter template. Use it when you want a
quick best fit before running MCMC or nested sampling.

The example is deliberately small and heavily commented. Replace the toy data
arrays with your observed dust fluxes and uncertainties.
"""

import numpy as np

from pyGraterFit import SEDFitter
from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution


# ---------------------------------------------------------------------------
# 1. Star and dust composition
# ---------------------------------------------------------------------------

# The Star object loads the stellar spectrum and stellar parameters used by
# pyGrater. The star name must exist in pyGrater's star table.
star = Star(star_name="HD113766")

# The Grain object controls the optical constants, Qabs/Qsca tables, and
# temperature/sublimation behavior. Changing the composition changes the
# radiative physics.
grain = Grain(composition="c_olivine_Fe_Poor")


# ---------------------------------------------------------------------------
# 2. Observational data
# ---------------------------------------------------------------------------

# Wavelengths are in microns. Fluxes/errors are in Jy in this template. They
# should be dust-only fluxes if you already subtracted the stellar photosphere.
wavelengths = np.array([8.0, 10.0, 12.0])
observed_flux = np.array([1.0, 1.2, 1.1])
flux_error = np.full_like(observed_flux, 0.1)


# ---------------------------------------------------------------------------
# 3. Model parameters
# ---------------------------------------------------------------------------

def scale_height(parameters):
    """Example dependent parameter: h0 is fixed to 5% of r0."""
    return 0.05 * parameters["r0"]


# Parameter convention:
# - scalar: fixed parameter;
# - two-value tuple/list: fitted range;
# - callable: dependent parameter evaluated from fitted/fixed values.
#
# By default, r0, a_min, A_norm, and M_tot are handled in log-space by the
# fitters that support log parameters. SciPy optimization uses transformed
# optimizer bounds internally where appropriate.
params = {
    "r0": (0.5, 5.0),          # characteristic radius [au]
    "h0": scale_height,        # scale height [au], dependent on r0
    "alphain": 10.0,           # inner radial density slope
    "alphaout": -5.0,          # outer radial density slope
    "beta": 1.0,               # flaring exponent
    "gamma": 1.0,              # vertical density exponent
    "itilt": 0.0,              # inclination [deg] for image-like models
    "PA": 90.0,                # position angle [deg]
    "omega": 45.0,             # argument of pericenter [deg]
    "a_min": (1e-7, 1e-5),     # minimum grain size [m]
    "a_max": 1e-3,             # maximum grain size [m]
    "kappa": (3.0, 4.5),       # size distribution power-law exponent
    "A_norm": (1e20, 1e35),    # dust normalization / emitting area scale
    "N_sizes_integral": 100,   # number of grain-size integration bins
    "g": 0.0,                  # scattering asymmetry parameter
}


# ---------------------------------------------------------------------------
# 4. Build and run the fitter
# ---------------------------------------------------------------------------

fitter = SEDFitter(
    grain,
    star,
    two_power_law,
    power_law_distribution,
    isotropic,
    wavelengths,
    observed_flux,
    flux_error,
    params,
)

result = fitter.fit(maxiter=1000, verbose=True)

print("Best-fit parameters:")
print(fitter.best_params)
print(f"Best chi2: {fitter.best_chi2:.4g}")
print(f"Reduced chi2: {result.chi2_red:.4g}")
