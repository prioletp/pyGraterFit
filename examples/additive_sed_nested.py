"""Additive multi-ring / multi-composition SED fit with dynesty.

This is the nested-sampling counterpart of `additive_sed_mcmc.py`. It uses the
friendly ring/material constructor, so the script only describes the
astrophysical model:

    - which rings exist;
    - which materials exist;
    - which data are fitted;
    - what total normalization range each ring should explore.

The fitter builds the lower-level component dictionaries internally.
"""

import numpy as np

from pyGraterFit import MultiRingSEDNestedFitter
from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution


# ---------------------------------------------------------------------------
# 1. Star, materials, and observed SED
# ---------------------------------------------------------------------------

star = Star(star_name="HD113766")

materials = {
    "olivine": Grain(composition="c_olivine_Fe_Poor"),
    "carbon": Grain(composition="aC_ACAR"),
    "silicate": Grain(composition="astroSi"),
}

# Replace these toy arrays with your dust-only fluxes and uncertainties.
wavelengths = np.array([8.0, 10.0, 12.0])
observed_flux = np.array([1.0, 1.2, 1.1])
flux_error = np.full_like(observed_flux, 0.1)


# ---------------------------------------------------------------------------
# 2. Ring model
# ---------------------------------------------------------------------------

def scale_height(parameters):
    """Dependent parameter evaluated separately for each ring."""
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

ring_params = {
    "ring1": {
        **common_ring_parameters,
        "r0": (0.01, 2.0),
    },
    "ring2": {
        **common_ring_parameters,
        "r0": (2.0, 100.0),
    },
}


# ---------------------------------------------------------------------------
# 3. Grouped normalization
# ---------------------------------------------------------------------------

# This is the prior range for each ring's total normalization. With three
# materials, each ring will have:
#
#     ring.A_norm_total
#     material_1.fraction_stick
#     material_2.fraction_stick
#
# The third material fraction is the leftover, so all physical fractions are
# positive and sum to one.
A_NORM_TOTAL_RANGE = (1e25, 1e38)


# ---------------------------------------------------------------------------
# 4. Build the nested fitter
# ---------------------------------------------------------------------------

fitter = MultiRingSEDNestedFitter(
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
    use_log_params=True,
    N_distances=300,
    parallel_components="auto",
    max_component_workers=4,
)


# ---------------------------------------------------------------------------
# 5. Dynesty controls
# ---------------------------------------------------------------------------

RUN_MODE = "fresh"  # "fresh", "resume", or "load"

N_LIVE_POINTS = 300
METHOD = "multi"       # multi-ellipsoid bounds
SAMPLE = "rslice"      # random-direction slice proposals
DYNAMIC = True         # dynesty DynamicNestedSampler
DLOGZ = 0.5            # evidence precision target
SEED = 8

CHECKPOINT = "additive_sed_nested.checkpoint"
RESULTS = "additive_sed_nested_results.npz"


# ---------------------------------------------------------------------------
# 6. Run, resume, or load
# ---------------------------------------------------------------------------

if RUN_MODE == "load":
    fitter.load_results(RESULTS)
elif RUN_MODE == "resume":
    fitter.resume_backend_nested(
        CHECKPOINT,
        npoints=N_LIVE_POINTS,
        method=METHOD,
        sample=SAMPLE,
        dynamic=DYNAMIC,
        dlogz=DLOGZ,
        seed=SEED,
        checkpoint_every=300,
    )
    fitter.save_results(RESULTS)
else:
    fitter.run(
        npoints=N_LIVE_POINTS,
        method=METHOD,
        sample=SAMPLE,
        dynamic=DYNAMIC,
        dlogz=DLOGZ,
        seed=SEED,
        checkpoint_file=CHECKPOINT,
        checkpoint_every=300,
        resume=False,
    )
    fitter.save_results(RESULTS)


# ---------------------------------------------------------------------------
# 7. Results
# ---------------------------------------------------------------------------

fitter.summary(include_mass_abundances=True)
print(fitter.format_component_mass_abundances(), end="")

fitter.plot_nested_diagnostics(
    "additive_sed_nested_plots",
    prefix="additive_sed_nested",
    seed=SEED,
)
fitter.plot_best_fit().savefig("additive_sed_nested_best_fit.png", dpi=150)
fitter.corner_plot(max_samples=5000, seed=SEED).savefig(
    "additive_sed_nested_corner.png", dpi=150)
