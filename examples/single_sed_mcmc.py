"""Single-component SED fit with emcee MCMC and a persistent backend.

Use this after checking the model with `single_sed_scipy.py`. The MCMC fitter
can first run a deterministic SciPy optimization, then initialize walkers near
that best fit and save the chain to an HDF5 backend.
"""

from pathlib import Path

from pyGraterFit import SEDMCMCFitter


# ---------------------------------------------------------------------------
# 1. Build the fitter
# ---------------------------------------------------------------------------

# In a real script, use the same objects and parameter dictionary shown in
# `single_sed_scipy.py`:
#
#     grain, star, two_power_law, power_law_distribution, isotropic,
#     wavelengths, observed_flux, flux_error, params
#
# The ellipsis keeps this file as a compact template; replace it with the full
# constructor call for your target.
fitter = SEDMCMCFitter(...)


# ---------------------------------------------------------------------------
# 2. Backend and run settings
# ---------------------------------------------------------------------------

# The backend stores the chain on disk, so interrupted runs can be restarted.
backend_path = Path("single_sed_backend.h5")

# Number of walkers. A common minimum is roughly 2-4 times the number of free
# parameters, but more walkers can help complicated posteriors.
N_WALKERS = 32

# Number of MCMC steps for this run. Production runs usually require many more
# steps than this template.
N_STEPS = 2000

# The initial walker cloud is drawn around the best SciPy fit. The FWHM
# fraction controls how tight that initial cloud is.
BEST_FIT_FWHM_FRAC = 0.02


# ---------------------------------------------------------------------------
# 3. Run SciPy initialization + MCMC
# ---------------------------------------------------------------------------

fitter.fit_then_mcmc(
    nwalkers=N_WALKERS,
    nsteps=N_STEPS,
    backend_path=backend_path,
    init="best_fit",
    best_fit_fwhm_frac=BEST_FIT_FWHM_FRAC,
)

fitter.save_results("single_sed_mcmc_results.npz")
fitter.plot_corner("single_sed_mcmc_corner.png")
fitter.plot_best_fit("single_sed_mcmc_best_fit.png")


# ---------------------------------------------------------------------------
# 4. Restarting a backend
# ---------------------------------------------------------------------------

# To continue from the last saved walker positions, uncomment this block.
# Use a new output backend if you want to preserve the old chain separately.
#
# fitter.restart_mcmc(
#     backend_path,
#     nsteps=1000,
#     backend_path=backend_path,
# )
# fitter.save_results("single_sed_mcmc_results_restarted.npz")
