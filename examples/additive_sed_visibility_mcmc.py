"""Complete multi-ring, multi-composition SED + V2 MCMC example."""

from pathlib import Path

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import HenveyGreenstein
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.multi_component_sed_visibility_mcmc import (
    SEDVisibilityMCMCFitter,
    vis2_from_vlti_loader,
)


RUN_SCIPY_FIRST = False
RUN_MCMC = False
RESUME_HDF_BACKEND = False

OUTPUT_PREFIX = "additive_sed_visibility_mcmc"
BACKEND = Path(f"{OUTPUT_PREFIX}_backend.h5")
CHAIN_NPZ = Path(f"{OUTPUT_PREFIX}_chain.npz")


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
    "value": np.array([0.82, 0.65], dtype=float),
    "error": np.array([0.05, 0.06], dtype=float),
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

fitter = SEDVisibilityMCMCFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=HenveyGreenstein,
    sed_wavelengths=sed_wavelengths,
    sed_fluxes=sed_fluxes,
    sed_flux_errors=sed_flux_errors,
    vis2=vis2_data,
    image_settings={"nx": 64, "ny": 64, "FOV_AU": 20.0, "nl": 51},
    image_wavelengths=np.array([10.0]),
    normalization_range=(1e25, 1e38),
    stellar_visibility_model="uniform_disk",
    stellar_angular_diameter_mas=0.7,
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=300,
)

if RUN_SCIPY_FIRST:
    fitter.fit(maxiter=300, verbose=True)
    fitter.summary()
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)

if RUN_MCMC:
    fitter.run_mcmc(
        nwalkers=max(32, 2 * fitter.ndim + 2),
        nsteps=2000,
        burn_in=500,
        thin=1,
        init="best_fit" if RUN_SCIPY_FIRST else "prior",
        backend_path=BACKEND,
        reset_backend=True,
        save_path=CHAIN_NPZ,
        progress=True,
    )
    fitter.mcmc_summary()
    fitter.mcmc_walkers_plot(max_walkers=40).savefig(
        f"{OUTPUT_PREFIX}_walkers.png", dpi=150)
    fitter.mcmc_corner_plot(max_samples=5000).savefig(
        f"{OUTPUT_PREFIX}_corner.png", dpi=150)
    fitter.plot_best_fit().savefig(f"{OUTPUT_PREFIX}_best_fit.png", dpi=150)

if RESUME_HDF_BACKEND:
    fitter.resume_backend_mcmc(
        BACKEND, nsteps=1000, burn_in=500, thin=1, progress=True)
    fitter.save_chain(CHAIN_NPZ)
