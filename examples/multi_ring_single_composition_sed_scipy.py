"""Complete multi-ring, single-composition SED SciPy example."""

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.multi_component_sed_scipy import AdditiveSEDScipyFitter


RUN_SCIPY = False
OUTPUT_PREFIX = "multi_ring_single_composition_scipy"


def scale_height(parameters):
    return 0.05 * parameters["r0"]


star = Star(star_name="HD113766")
grain = Grain(redo_Q=False, composition="c_olivine_Fe_Poor")
materials = {"olivine": grain}

wavelengths = np.array([8.0, 10.0, 12.0], dtype=float)
observed_flux = np.array([1.0, 1.2, 1.1], dtype=float)
flux_error = np.array([0.1, 0.1, 0.1], dtype=float)

common = {
    "h0": scale_height,
    "alphain": 10.0,
    "alphaout": (-10.0, -1.0),
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

ring_params = {
    "inner_ring": {**common, "r0": (0.01, 2.0)},
    "outer_ring": {**common, "r0": (2.0, 100.0)},
}

fitter = AdditiveSEDScipyFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    wavelengths=wavelengths,
    fluxes=observed_flux,
    fluxes_err=flux_error,
    normalization_range=(1e25, 1e38),
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=300,
)

if RUN_SCIPY:
    fitter.fit(maxiter=500, verbose=True)
    fitter.summary()
    print(fitter.format_component_mass_abundances(), end="")
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)
else:
    print("Fitter built. Set RUN_SCIPY=True to optimize.")
