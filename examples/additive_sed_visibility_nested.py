"""Complete multi-ring, multi-composition image-V2 nested-sampling example.

This uses SEDVisibilityNestedFitter.  Set FIT_SED_TOO = False for a
visibility-only fit, or True to add the SED term to the same fit.
"""

from pathlib import Path

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import HenveyGreenstein
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.multi_component_sed_visibility_dynesty import (
    SEDVisibilityNestedFitter,
)
from pyGraterFit.fitters.multi_component_sed_visibility_mcmc import (
    vis2_from_vlti_loader,
)


FIT_SED_TOO = False
RUN_MODE = "build_only"  # "build_only", "fresh", "resume", or "load"

OUTPUT_PREFIX = "additive_sed_visibility_nested"
CHECKPOINT = Path(f"{OUTPUT_PREFIX}_checkpoint.pkl")
RESULTS_NPZ = Path(f"{OUTPUT_PREFIX}_results.npz")


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

# Normalized V2 dictionary. You can also use:
#     vis2_data = vis2_from_vlti_loader("/path/to/oifits_or_directory")
vis2_data = {
    "value": np.array([0.82, 0.65, 0.58], dtype=float),
    "error": np.array([0.05, 0.06, 0.07], dtype=float),
    "u_m": np.array([20.0, 0.0, 25.0], dtype=float),
    "v_m": np.array([0.0, 20.0, 10.0], dtype=float),
    "wavelength_m": np.array([8.35e-6, 10.6e-6, 12.8e-6], dtype=float),
}

# The observed wavelengths are assigned to the nearest rendered image plane.
image_wavelengths = np.array([8.0, 10.0, 13.0], dtype=float)

common = {
    "h0": scale_height,
    "alphain": 10.0,
    "alphaout": (-10.0, -1.0),
    "gamma": 2.0,
    "beta": 1.0,
    "itilt": 55.0,
    "PA": (0.0, 180.0),
    "omega": 0.0,
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

sed_arguments = {}
if FIT_SED_TOO:
    sed_arguments = {
        "sed_wavelengths": sed_wavelengths,
        "sed_fluxes": sed_fluxes,
        "sed_flux_errors": sed_flux_errors,
    }

fitter = SEDVisibilityNestedFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=HenveyGreenstein,
    vis2=vis2_data,
    image_settings={"nx": 64, "ny": 64, "FOV_AU": 20.0, "nl": 51},
    image_wavelengths=image_wavelengths,
    normalization_range=(1e25, 1e38),
    stellar_visibility_model="uniform_disk",
    stellar_angular_diameter_mas=0.7,
    use_log_params=True,
    N_distances=300,
    **sed_arguments,
)

if RUN_MODE == "fresh":
    fitter.run(
        npoints=400,
        dlogz=0.1,
        dynamic=True,
        checkpoint_file=CHECKPOINT,
        checkpoint_every=300,
        progress=True,
    )
    fitter.save_results(RESULTS_NPZ)
    fitter.summary()
    fitter.corner_plot(max_samples=5000).savefig(
        f"{OUTPUT_PREFIX}_corner.png", dpi=150)
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)

if RUN_MODE == "resume":
    fitter.resume_backend_nested(
        CHECKPOINT,
        npoints=400,
        dlogz=0.1,
        dynamic=True,
        progress=True,
    )
    fitter.save_results(RESULTS_NPZ)

if RUN_MODE == "load":
    fitter.load_results(RESULTS_NPZ)
    fitter.summary()
    fitter.corner_plot(max_samples=5000).savefig(
        f"{OUTPUT_PREFIX}_corner.png", dpi=150)
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)
