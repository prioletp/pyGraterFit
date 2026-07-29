"""Complete single-ring SED MCMC example with restart options."""

from pathlib import Path

import numpy as np

from pyGrater import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.phase_functions import isotropic
from pyGrater.size_distributions import power_law_distribution
from pyGraterFit.fitters.single_ring_sed_mcmc import SEDMCMCFitter


RUN_SCIPY_FIRST = False
RUN_MCMC = False
RESUME_HDF_BACKEND = False
RESTART_FROM_NPZ_CHAIN = False

OUTPUT_PREFIX = "single_sed_mcmc"
BACKEND = Path(f"{OUTPUT_PREFIX}_backend.h5")
CHAIN_NPZ = Path(f"{OUTPUT_PREFIX}_chain.npz")


def scale_height(parameters):
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

# Optional. If omitted, MCMC priors are taken from the fitted ranges in params.
prior_ranges = {
    name: value for name, value in params.items()
    if isinstance(value, tuple) and len(value) == 2
}

fitter = SEDMCMCFitter(
    grain=grain,
    star=star,
    density_distribution=two_power_law,
    size_distribution=power_law_distribution,
    scattering_phase_function=isotropic,
    wavelengths=wavelengths,
    fluxes=observed_flux,
    fluxes_err=flux_error,
    params=params,
    prior_ranges=prior_ranges,
    method="Nelder-Mead",
    use_log_params=True,
    N_distances=800,
)

if RUN_SCIPY_FIRST:
    fitter.fit(maxiter=1000, verbose=True)
    fitter.summary()

if RUN_MCMC:
    # If RUN_SCIPY_FIRST=True, init="best_fit" starts walkers around the SciPy
    # best fit. If not, switch to init="prior" or pass an explicit best_fit_values dictionary.
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
        BACKEND,
        nsteps=1000,
        burn_in=500,
        thin=1,
        progress=True,
    )
    fitter.save_chain(CHAIN_NPZ)

if RESTART_FROM_NPZ_CHAIN:
    fitter.restart_mcmc(
        CHAIN_NPZ,
        nsteps=1000,
        backend_path=f"{OUTPUT_PREFIX}_restarted_backend.h5",
        burn_in=500,
        thin=1,
        progress=True,
    )
