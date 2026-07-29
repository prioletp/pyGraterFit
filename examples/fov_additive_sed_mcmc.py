"""Complete FOV-aware multi-ring, multi-composition SED MCMC example."""

from pathlib import Path

import numpy as np

from pyGrater import Grain, Star
from pyGrater.SED_fov import GaussianFieldOfView
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.multi_component_fov_sed_mcmc import (
    FOVAdditiveSEDMCMCFitter,
)


RUN_SCIPY_FIRST = False
RUN_MCMC = False
RESUME_HDF_BACKEND = False

OUTPUT_PREFIX = "fov_additive_sed_mcmc"
BACKEND = Path(f"{OUTPUT_PREFIX}_backend.h5")
CHAIN_NPZ = Path(f"{OUTPUT_PREFIX}_chain.npz")


def scale_height(parameters):
    return 0.05 * parameters["r0"]


star = Star(star_name="HD113766")
materials = {
    "olivine": Grain(redo_Q=False, composition="c_olivine_Fe_Poor"),
    "silicate": Grain(redo_Q=False, composition="astroSi"),
}
wavelengths = np.array([8.0, 10.0, 12.0], dtype=float)
instrument_names = np.array(["photometry", "midi", "midi"], dtype=str)
transmission_by_instrument = {
    "photometry": None,  # full transmission
    "midi": GaussianFieldOfView(fwhm_arcsec=0.5, cutoff_radius_arcsec=1.0),
}
observed_flux = np.array([1.0, 1.2, 1.1], dtype=float)
flux_error = np.array([0.1, 0.1, 0.1], dtype=float)

common = {
    "h0": scale_height,
    "alphain": 10.0,
    "alphaout": (-10.0, -1.0),
    "gamma": 2.0,
    "beta": 1.0,
    "itilt": 55.0,
    "PA": 90.0,
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

fitter = FOVAdditiveSEDMCMCFitter(
    materials=materials,
    ring_params=ring_params,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    wavelengths=wavelengths,
    instrument_names=instrument_names,
    transmission_by_instrument=transmission_by_instrument,
    fluxes=observed_flux,
    fluxes_err=flux_error,
    normalization_ranges=(1e25, 1e38),
    normalization_mode="group_total_fraction",
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=300,
    n_azimuth=64,
)

if RUN_SCIPY_FIRST:
    fitter.fit(maxiter=500, verbose=True)
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
