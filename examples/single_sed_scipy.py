"""Complete single-ring, single-composition SED SciPy example."""

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.single_ring_sed_scipy import SEDFitter


RUN_SCIPY = False
OUTPUT_PREFIX = "single_sed_scipy"


def scale_height(parameters):
    """Keep the scale height equal to 5 percent of r0."""
    return 0.05 * parameters["r0"]


star = Star(star_name="HD113766")
grain = Grain(redo_Q=False, composition="c_olivine_Fe_Poor")

wavelengths = np.array([8.0, 10.0, 12.0], dtype=float)
observed_flux = np.array([1.0, 1.2, 1.1], dtype=float)
flux_error = np.array([0.1, 0.1, 0.1], dtype=float)

params = {
    "r0": (0.5, 5.0),
    "h0": scale_height,
    "alphain": 10.0,
    "alphaout": (-10.0, -1.0),
    "beta": 1.0,
    "gamma": 1.0,
    "itilt": 0.0,
    "PA": 90.0,
    "omega": 45.0,
    "a_min": (1e-7, 1e-5),
    "a_max": 1e-3,
    "kappa": (3.0, 4.5),
    "A_norm": (1e20, 1e35),
    "N_sizes_integral": 100,
    "g": 0.0,
}

fitter = SEDFitter(
    grain=grain,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    wavelengths=wavelengths,
    fluxes=observed_flux,
    fluxes_err=flux_error,
    params=params,
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=800,
)

if RUN_SCIPY:
    result = fitter.fit(maxiter=1000, verbose=True)
    fitter.summary()
    print("Raw SciPy result:", result.message)
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)
else:
    print("Fitter built. Set RUN_SCIPY=True to run the optimizer.")
