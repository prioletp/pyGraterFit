"""Complete multi-ring, multi-composition correlated-flux nested example."""

from pathlib import Path

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.multi_component_sed_correlated_flux_dynesty import (
    AdditiveSEDCorrelatedFluxNestedFitter,
)
from pyGraterFit.fitters.single_ring_sed_correlated_flux_dynesty import (
    correlated_flux_from_vlti_loader,
)


RUN_MODE = "build_only"  # "build_only", "fresh", "resume", or "load"
FIT_SED_TOO = True
OUTPUT_PREFIX = "additive_sed_correlated_flux_nested"
CHECKPOINT = Path(f"{OUTPUT_PREFIX}.checkpoint")
RESULTS = Path(f"{OUTPUT_PREFIX}_results.npz")


def scale_height(parameters):
    return 0.05 * parameters["r0"]


star = Star(star_name="HD113766")
materials = {
    "olivine": Grain(redo_Q=False, composition="c_olivine_Fe_Poor"),
    "silicate": Grain(redo_Q=False, composition="astroSi"),
}
sed_wavelengths = np.array([8.0, 10.0, 12.0], dtype=float)
sed_fluxes = np.array([1.0, 1.2, 1.1], dtype=float)
sed_flux_errors = np.array([0.1, 0.1, 0.1], dtype=float)
correlated_flux = {
    "value": np.array([0.8, 0.6], dtype=float),
    "error": np.array([0.08, 0.07], dtype=float),
    "u_m": np.array([20.0, 0.0], dtype=float),
    "v_m": np.array([0.0, 20.0], dtype=float),
    "wavelength_m": np.array([10e-6, 10e-6], dtype=float),
}
common = {
    "h0": scale_height,
    "alphain": 10.0,
    "alphaout": (-10.0, -1.0),
    "gamma": 2.0,
    "beta": 1.0,
    "itilt": 55.0,
    "PA": (0.0, 180.0),
    "omega": 0.0,
    "ring_fwhm_au": (0.1, 5.0),
    "a_min": (0.05e-6, 1e-4),
    "a_max": 1000e-6,
    "kappa": (1.0, 5.0),
    "N_sizes_integral": 80,
    "g": 0.0,
}
ring_params = {
    "inner_ring": {**common, "r0": (0.1, 2.0)},
    "outer_ring": {**common, "r0": (2.0, 20.0)},
}

fitter = AdditiveSEDCorrelatedFluxNestedFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    correlated_flux=correlated_flux,
    sed_wavelengths=sed_wavelengths if FIT_SED_TOO else None,
    sed_fluxes=sed_fluxes if FIT_SED_TOO else None,
    sed_flux_errors=sed_flux_errors if FIT_SED_TOO else None,
    stellar_angular_diameter_mas=0.7,
    normalization_range=(1e25, 1e38),
    visibility_model="gaussian_ring",
    use_log_params=True,
    N_distances=300,
)

if RUN_MODE == "fresh":
    fitter.run(
        npoints=300, method="multi", sample="rslice", dynamic=True,
        dlogz=0.5, checkpoint_file=CHECKPOINT, checkpoint_every=300,
        resume=False)
    fitter.save_results(RESULTS)
elif RUN_MODE == "resume":
    fitter.resume_backend_nested(
        CHECKPOINT, npoints=300, method="multi", sample="rslice",
        dynamic=True, dlogz=0.5, checkpoint_every=300)
    fitter.save_results(RESULTS)
elif RUN_MODE == "load":
    fitter.load_results(RESULTS)

if RUN_MODE != "build_only":
    fitter.summary(include_mass_abundances=True)
    fitter.plot_nested_diagnostics(
        f"{OUTPUT_PREFIX}_plots", prefix=OUTPUT_PREFIX)
    fitter.corner_plot(max_samples=5000).savefig(
        f"{OUTPUT_PREFIX}_corner.png", dpi=150)
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)
else:
    print("Correlated-flux nested fitter built. Set RUN_MODE='fresh' to sample.")
