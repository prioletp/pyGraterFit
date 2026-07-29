"""Complete multi-ring, single-composition SED nested example."""

from pathlib import Path

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.multi_component_sed_dynesty import AdditiveSEDNestedFitter


RUN_MODE = "build_only"  # "build_only", "fresh", "resume", or "load"
OUTPUT_PREFIX = "multi_ring_single_composition_nested"
CHECKPOINT = Path(f"{OUTPUT_PREFIX}.checkpoint")
RESULTS = Path(f"{OUTPUT_PREFIX}_results.npz")


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

fitter = AdditiveSEDNestedFitter(
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
    print(fitter.format_component_mass_abundances(), end="")
    fitter.plot_nested_diagnostics(
        f"{OUTPUT_PREFIX}_plots", prefix=OUTPUT_PREFIX)
    fitter.corner_plot(max_samples=5000).savefig(
        f"{OUTPUT_PREFIX}_corner.png", dpi=150)
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)
else:
    print("Nested fitter built. Set RUN_MODE='fresh' to sample.")
