"""Multi-ring, single-composition SED fit with SciPy.

This is the simplest additive model: several rings, but the same grain
composition in each ring.  Because there is only one material per ring, the
fitter does not create composition fractions.  Each ring keeps its own
``A_norm`` range and is added linearly to the model SED.
"""

import numpy as np

from pyGraterFit import MultiRingSEDScipyFitter
from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution


star = Star(star_name="HD113766")
grain = Grain(composition="c_olivine_Fe_Poor")

wavelengths = np.array([8.0, 10.0, 12.0])
observed_flux = np.array([1.0, 1.2, 1.1])
flux_error = np.full_like(observed_flux, 0.1)


def scale_height(parameters):
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
    "A_norm": (1e25, 1e38),
}

components = {
    "inner_ring": grain,
    "outer_ring": grain,
}

params_by_component = {
    "inner_ring": {
        **common_ring_parameters,
        "r0": (0.01, 2.0),
    },
    "outer_ring": {
        **common_ring_parameters,
        "r0": (2.0, 100.0),
    },
}

fitter = MultiRingSEDScipyFitter(
    components=components,
    params_by_component=params_by_component,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    wavelengths=wavelengths,
    fluxes=observed_flux,
    fluxes_err=flux_error,
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=300,
)

result = fitter.fit(maxiter=500)
fitter.summary()
print(fitter.format_component_mass_abundances())
fitter.plot_best_fit().savefig(
    "multi_ring_single_composition_scipy_best_fit.png", dpi=150)
