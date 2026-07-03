#%%
"""
sed_mcmc.py — optimized SED MCMC fitter
============================================================

This fitter does not subclass the original simple fitter or any previous
MCMC fitter.  It contains the scipy fit, MCMC, restart, diagnostics, and
plotting logic needed to use the optimized SED engine.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing, minimize

from pyGrater import CachedSED
from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.utils.mass import (
    format_total_mass, single_component_total_mass)
from pyGraterFit.utils.mcmc_backend import write_parameter_names
from pyGraterFit.utils.parameter_handling import (
    resolve_parameters, split_parameter_specifications)


LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}


class SEDMCMCFitter:
    """Standalone SED fitter with scipy initialization and emcee MCMC."""

    _FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

    def __init__(self, grain, star, density_distribution, size_distribution,
                 scattering_phase_function, wavelengths, fluxes, fluxes_err,
                 params,
                 prior_ranges=None, best_fit_values=None,
                 method='Nelder-Mead', use_log_params=True,
                 N_distances=800, sed_model_class=CachedSED,
                 sed_model_kwargs=None):
        self.grain = grain
        self.star = star
        self.density_distribution = density_distribution
        self.size_distribution = size_distribution
        self.scattering_phase_function = scattering_phase_function
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.obs = np.asarray(fluxes, dtype=np.float64)
        self.obs_err = np.asarray(fluxes_err, dtype=np.float64)
        self._obs = self.obs
        self._inv_obs_err = 1.0 / self.obs_err
        self.method = method

        sed_model_kwargs = dict(sed_model_kwargs or {})
        sed_model_kwargs.setdefault('N_distances', N_distances)
        self.sed_obj = sed_model_class(
            grain, star, density_distribution, size_distribution,
            self.wavelengths, **sed_model_kwargs)

        (self.free_params_range, self.fixed_params_value,
         self.dependent_params) = split_parameter_specifications(params)

        self.param_names = list(self.free_params_range.keys())
        self.ndim = len(self.param_names)
        self.log_params = (
            {k for k in self.param_names if k in LOG_SPACE_PARAMS}
            if use_log_params else set())
        self._bounds = []
        for name in self.param_names:
            lo, hi = self.free_params_range[name]
            if name in self.log_params:
                self._bounds.append((np.log10(lo), np.log10(hi)))
            else:
                self._bounds.append((lo, hi))

        print(f'Free parameters ({self.ndim}): {self.free_params_range}')
        print(f'Fixed parameters: {self.fixed_params_value}')
        print(f'Dependent parameters: {list(self.dependent_params)}')
        print(f'Log-space parameters: {self.log_params}')
        print(f'Method: {method}')

        self.prior_ranges = self._validate_prior_ranges(prior_ranges)
        self.input_best_fit_values = (
            None if best_fit_values is None
            else self._validate_best_fit(best_fit_values))

        self.n_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None
        self.param_errors = None
        self.mcmc_sampler = None
        self.mcmc_chain = None
        self.mcmc_log_prob = None
        self.flat_samples = None
        self.mcmc_best_params = None
        self.mcmc_best_log_prob = None
        self.mcmc_best_chi2 = None
        self.mcmc_param_summary = None
        self.n_mcmc_evaluations = 0
        self.loaded_chain = None
        self.loaded_log_prob = None
        self.loaded_last_positions = None
        self.initial_walker_summary = None
        self.initial_walker_positions = None

    # -- scipy parameter conversions ---------------------------------------

    def _to_dict(self, x):
        d = {}
        for i, name in enumerate(self.param_names):
            d[name] = 10**x[i] if name in self.log_params else x[i]
        return d

    def _complete_parameters(self, free_parameters):
        """Add fixed values and evaluate dependent parameter callables."""
        return resolve_parameters(
            free_parameters, self.fixed_params_value, self.dependent_params)

    def _to_vec(self, d):
        return np.array([
            np.log10(d[k]) if k in self.log_params else d[k]
            for k in self.param_names])

    # -- validation and MCMC conversions ------------------------------------

    def _validate_prior_ranges(self, prior_ranges):
        if prior_ranges is None:
            prior_ranges = self.free_params_range
        missing = [name for name in self.param_names if name not in prior_ranges]
        if missing:
            raise ValueError(
                'Missing MCMC prior ranges for free parameters: '
                + ', '.join(missing))

        cleaned = {}
        for name in self.param_names:
            val = prior_ranges[name]
            if not isinstance(val, (list, tuple, np.ndarray)) or len(val) != 2:
                raise ValueError(
                    f'Prior range for {name} must be a two-element range.')
            lo, hi = float(val[0]), float(val[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(
                    f'Prior range for {name} must have finite low < high.')
            cleaned[name] = (lo, hi)
        return cleaned

    def _validate_best_fit(self, best_fit_values):
        missing = [name for name in self.param_names if name not in best_fit_values]
        if missing:
            raise ValueError(
                'Missing best-fit values for free parameters: '
                + ', '.join(missing))
        return {name: float(best_fit_values[name]) for name in self.param_names}

    def _phys_to_mcmc_vec(self, params_dict):
        return np.array([params_dict[name] for name in self.param_names],
                        dtype=np.float64)

    def _mcmc_vec_to_dict(self, theta):
        return {
            name: float(theta[i])
            for i, name in enumerate(self.param_names)
        }

    # -- objective -----------------------------------------------------------

    def chi_squared(self, x):
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        x = np.clip(x, lo, hi)
        params_dict = self._complete_parameters(self._to_dict(x))
        try:
            model = self.sed_obj.get_SED(
                keep_separate_fluxes=False, **params_dict)
            residual = (self._obs - model) * self._inv_obs_err
            chi2 = float(np.dot(residual, residual))
        except Exception as exc:
            print(f'[ERROR] eval failed: {exc}')
            chi2 = 1e10

        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_params = self._to_dict(x)
        if self.n_evaluations % 20 == 0:
            print(f'  eval {self.n_evaluations}: chi2={chi2:.4f}')
        return chi2

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        self.n_evaluations = 0
        self.best_chi2 = np.inf
        if initial_guess is not None:
            x0 = self._to_vec(initial_guess)
        else:
            x0 = np.array([0.5 * (b[0] + b[1]) for b in self._bounds])

        print(f'\nStarting {self.method} optimisation ...')
        if self.method == 'differential_evolution':
            result = differential_evolution(
                self.chi_squared, bounds=self._bounds,
                maxiter=maxiter, disp=verbose, seed=42)
        elif self.method == 'dual_annealing':
            result = dual_annealing(
                self.chi_squared, bounds=self._bounds,
                maxiter=maxiter, seed=42)
        else:
            result = minimize(
                self.chi_squared, x0, method=self.method,
                bounds=self._bounds,
                options={'maxiter': maxiter, 'disp': verbose})
        result.best_params = self.best_params
        result.best_chi2 = self.best_chi2
        result.chi2_red = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        return result

    def chi_squared_physical(self, params_dict):
        full = self._complete_parameters(params_dict)
        try:
            model = self.sed_obj.get_SED(
                keep_separate_fluxes=False, **full)
            residual = (self._obs - model) * self._inv_obs_err
            chi2 = float(np.dot(residual, residual))
        except Exception as exc:
            print(f'[ERROR] MCMC eval failed: {exc}')
            chi2 = np.inf
        return chi2

    def log_prior(self, theta):
        for i, name in enumerate(self.param_names):
            lo, hi = self.prior_ranges[name]
            if not np.isfinite(theta[i]) or theta[i] < lo or theta[i] > hi:
                return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        chi2 = self.chi_squared_physical(self._mcmc_vec_to_dict(theta))
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        self.n_mcmc_evaluations += 1
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # -- walker initialization ---------------------------------------------

    def set_best_fit_values(self, best_fit_values):
        self.best_params = self._validate_best_fit(best_fit_values)
        self.best_chi2 = self.chi_squared_physical(self.best_params)
        return self.best_params

    @staticmethod
    def _draw_truncated_normal(rng, center, low, high, sigma, n):
        if sigma <= 0:
            raise ValueError('Best-fit walker FWHM must be positive.')
        draws = np.empty(n, dtype=np.float64)
        filled = 0
        attempts = 0
        while filled < n:
            batch = max(4 * (n - filled), 64)
            cand = rng.normal(center, sigma, size=batch)
            cand = cand[(cand >= low) & (cand <= high)]
            take = min(len(cand), n - filled)
            if take:
                draws[filled:filled + take] = cand[:take]
                filled += take
            attempts += batch
            if filled == 0 and attempts > 100000:
                half_width = min(2.0 * sigma, high - low)
                lo = max(low, center - half_width)
                hi = min(high, center + half_width)
                draws[:] = rng.uniform(lo, hi, size=n)
                filled = n
        return draws

    def _record_initial_walker_summary(self, p0, label):
        summary = {}
        print(f'\nInitial walker spread ({label}):')
        for i, name in enumerate(self.param_names):
            vals = p0[:, i]
            entry = {
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'std': float(np.std(vals)),
            }
            if name in self.log_params and np.all(vals > 0):
                log_vals = np.log10(vals)
                entry['log10_min'] = float(np.min(log_vals))
                entry['log10_max'] = float(np.max(log_vals))
                entry['log10_std'] = float(np.std(log_vals))
                print(
                    f'  {name:<15} min={entry["min"]:.6g} '
                    f'max={entry["max"]:.6g} '
                    f'log10_std={entry["log10_std"]:.6g}')
            else:
                print(
                    f'  {name:<15} min={entry["min"]:.6g} '
                    f'max={entry["max"]:.6g} std={entry["std"]:.6g}')
            summary[name] = entry
        self.initial_walker_summary = summary
        self.initial_walker_positions = p0.copy()

    def _initial_walker_positions(self, center, nwalkers, scatter_frac, seed):
        rng = np.random.default_rng(seed)
        center_vec = self._phys_to_mcmc_vec(center)
        p0 = np.empty((nwalkers, self.ndim), dtype=np.float64)
        for i, name in enumerate(self.param_names):
            lo, hi = self.prior_ranges[name]
            c = center_vec[i]
            if c < lo or c > hi:
                raise ValueError(
                    f'Best-fit value for {name}={c} is outside its prior '
                    f'range ({lo}, {hi}).')
            if name in self.log_params:
                if lo <= 0 or hi <= 0 or c <= 0:
                    raise ValueError(
                        f'Cannot initialize {name} in log space because lo, '
                        f'hi, and center must be positive.')
                log_lo, log_hi, log_c = np.log10([lo, hi, c])
                sigma = scatter_frac * (log_hi - log_lo) * self._FWHM_TO_SIGMA
                draws = self._draw_truncated_normal(
                    rng, log_c, log_lo, log_hi, sigma, nwalkers)
                p0[:, i] = 10**draws
            else:
                sigma = scatter_frac * (hi - lo) * self._FWHM_TO_SIGMA
                p0[:, i] = self._draw_truncated_normal(
                    rng, c, lo, hi, sigma, nwalkers)
        if np.unique(p0, axis=0).shape[0] < min(nwalkers, self.ndim + 1):
            raise RuntimeError(
                'Best-fit initialization produced too few distinct walkers. '
                'Increase best_fit_fwhm_frac or check prior boundaries.')
        self._record_initial_walker_summary(p0, 'best-fit truncated Gaussian')
        return p0

    def _prior_walker_positions(self, nwalkers, seed):
        rng = np.random.default_rng(seed)
        p0 = np.empty((nwalkers, self.ndim), dtype=np.float64)
        for i, name in enumerate(self.param_names):
            lo, hi = self.prior_ranges[name]
            if name in self.log_params:
                if lo <= 0 or hi <= 0:
                    raise ValueError(
                        f'Cannot draw {name} uniformly in log space because '
                        f'its prior range is not strictly positive.')
                log_lo, log_hi = np.log10([lo, hi])
                p0[:, i] = 10**rng.uniform(log_lo, log_hi, size=nwalkers)
            else:
                p0[:, i] = rng.uniform(lo, hi, size=nwalkers)
        self._record_initial_walker_summary(p0, 'prior')
        return p0

    # -- MCMC run and results ----------------------------------------------

    def run_mcmc(self, best_fit_values=None, nwalkers=None, nsteps=1000,
                 burn_in=0, thin=1, best_fit_fwhm_frac=0.02,
                 scatter_frac=None, seed=42, init='best_fit',
                 restart_from=None, initial_state=None, save_path=None,
                 backend_path=None, resume_backend=False,
                 reset_backend=True, progress=True, **run_kwargs):
        try:
            import emcee
        except ImportError as exc:
            raise ImportError(
                'SEDMCMCFitter requires emcee. Install it with '
                '`pip install emcee` to run MCMC sampling.') from exc
        if scatter_frac is not None:
            best_fit_fwhm_frac = scatter_frac
        if nwalkers is None:
            nwalkers = max(32, 2 * self.ndim + 2)
        if nwalkers < 2 * self.ndim:
            raise ValueError('nwalkers should be at least 2 * ndim.')

        backend = None
        backend_last_sample = None
        if backend_path is not None:
            backend = emcee.backends.HDFBackend(str(backend_path))
            if resume_backend:
                if backend.iteration <= 0:
                    raise ValueError(
                        f'Cannot resume empty emcee backend: {backend_path}')
                backend_last_sample = backend.get_last_sample()
                nwalkers = backend_last_sample.coords.shape[0]
                if backend_last_sample.coords.shape[1] != self.ndim:
                    raise ValueError('Backend ndim does not match this fitter.')
                print(
                    f'Resuming HDF5 backend {backend_path} from '
                    f'{backend.iteration} saved steps.')
            elif reset_backend:
                backend.reset(nwalkers, self.ndim)
                write_parameter_names(backend_path, self.param_names)

        if backend_last_sample is not None:
            p0 = backend_last_sample
        elif restart_from is not None:
            p0 = self.load_chain(restart_from)
        elif initial_state is not None:
            p0 = np.asarray(initial_state, dtype=np.float64)
        elif init == 'prior':
            p0 = self._prior_walker_positions(nwalkers, seed)
        elif init == 'best_fit':
            if best_fit_values is not None:
                center = self.set_best_fit_values(best_fit_values)
            elif self.best_params is not None:
                center = self.best_params
            elif self.input_best_fit_values is not None:
                center = self.set_best_fit_values(self.input_best_fit_values)
            else:
                raise RuntimeError(
                    'No best-fit values available. Run fit(), call '
                    'fit_then_mcmc(), pass best_fit_values, or use init="prior".')
            p0 = self._initial_walker_positions(
                center, nwalkers, best_fit_fwhm_frac, seed)
        else:
            raise ValueError("init must be either 'best_fit' or 'prior'.")

        p0_coords = (
            np.asarray(backend_last_sample.coords, dtype=np.float64)
            if backend_last_sample is not None
            else np.asarray(p0, dtype=np.float64))
        if p0_coords.ndim != 2 or p0_coords.shape[1] != self.ndim:
            raise ValueError(
                f'Initial walker positions must have shape (nwalkers, {self.ndim}).')
        nwalkers = p0_coords.shape[0]
        if nwalkers < 2 * self.ndim:
            raise ValueError('nwalkers should be at least 2 * ndim.')
        if not np.all(np.isfinite([self.log_prior(theta) for theta in p0_coords])):
            raise ValueError('Initial walker positions must be inside prior ranges.')

        print(f'\nStarting MCMC with {nwalkers} walkers, {nsteps} steps.')
        print(f'Prior ranges: {self.prior_ranges}')
        self.n_mcmc_evaluations = 0
        sampler = emcee.EnsembleSampler(
            nwalkers, self.ndim, self.log_probability, backend=backend)
        sampler.run_mcmc(p0, nsteps, progress=progress, **run_kwargs)

        self.mcmc_sampler = sampler
        self.mcmc_chain = sampler.get_chain()
        self.mcmc_log_prob = sampler.get_log_prob()
        self.flat_samples = sampler.get_chain(
            discard=burn_in, thin=thin, flat=True)
        flat_log_prob = sampler.get_log_prob(
            discard=burn_in, thin=thin, flat=True)
        best_idx = int(np.nanargmax(flat_log_prob))
        self.mcmc_best_log_prob = float(flat_log_prob[best_idx])
        self.mcmc_best_params = self._mcmc_vec_to_dict(
            self.flat_samples[best_idx])
        self.mcmc_best_chi2 = -2.0 * self.mcmc_best_log_prob
        if self.best_params is None or self.mcmc_best_chi2 < self.best_chi2:
            self.best_params = self.mcmc_best_params
            self.best_chi2 = self.mcmc_best_chi2
        self.mcmc_param_summary = self._summarise_samples(self.flat_samples)
        if save_path is not None:
            self.save_chain(save_path)
        return sampler

    def save_chain(self, filename):
        if self.mcmc_sampler is None:
            raise RuntimeError('Run run_mcmc() before saving a chain.')
        np.savez(
            filename,
            chain=self.mcmc_chain,
            log_prob=self.mcmc_log_prob,
            last_positions=self.mcmc_chain[-1],
            param_names=np.array(self.param_names, dtype=str),
            prior_lows=np.array(
                [self.prior_ranges[name][0] for name in self.param_names],
                dtype=np.float64),
            prior_highs=np.array(
                [self.prior_ranges[name][1] for name in self.param_names],
                dtype=np.float64),
        )
        return filename

    def load_chain(self, filename):
        data = np.load(filename)
        saved_names = list(data['param_names'].astype(str))
        if saved_names != self.param_names:
            raise ValueError(
                'Saved chain parameter order does not match this fitter. '
                f'Saved={saved_names}, current={self.param_names}')
        self.loaded_chain = data['chain']
        self.loaded_log_prob = data['log_prob']
        self.loaded_last_positions = data['last_positions']
        return np.array(self.loaded_last_positions, dtype=np.float64)

    def fit_then_mcmc(self, best_fit_values=None, initial_guess=None,
                      maxiter=1000, fit_verbose=True, **mcmc_kwargs):
        if best_fit_values is not None:
            self.set_best_fit_values(best_fit_values)
            fit_result = None
        elif self.input_best_fit_values is not None:
            self.set_best_fit_values(self.input_best_fit_values)
            fit_result = None
        else:
            fit_result = self.fit(
                initial_guess=initial_guess, maxiter=maxiter,
                verbose=fit_verbose)
        sampler = self.run_mcmc(**mcmc_kwargs)
        return fit_result, sampler

    def run_prior_mcmc(self, **mcmc_kwargs):
        return self.run_mcmc(init='prior', **mcmc_kwargs)

    def restart_mcmc(self, filename, **mcmc_kwargs):
        return self.run_mcmc(restart_from=filename, **mcmc_kwargs)

    def resume_backend_mcmc(self, backend_path, **mcmc_kwargs):
        return self.run_mcmc(
            backend_path=backend_path, resume_backend=True, **mcmc_kwargs)

    # -- summaries and plots -----------------------------------------------

    def _summarise_samples(self, samples):
        summary = {}
        q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
        for i, name in enumerate(self.param_names):
            summary[name] = {
                'median': float(q50[i]),
                'minus_1sigma': float(q50[i] - q16[i]),
                'plus_1sigma': float(q84[i] - q50[i]),
                'q16': float(q16[i]),
                'q84': float(q84[i]),
            }
        return summary

    def mcmc_diagnostics(self):
        if self.mcmc_sampler is None:
            raise RuntimeError('Run run_mcmc() first.')
        diagnostics = {
            'acceptance_fraction_mean': float(
                np.mean(self.mcmc_sampler.acceptance_fraction)),
            'acceptance_fraction_min': float(
                np.min(self.mcmc_sampler.acceptance_fraction)),
            'acceptance_fraction_max': float(
                np.max(self.mcmc_sampler.acceptance_fraction)),
        }
        try:
            tau = self.mcmc_sampler.get_autocorr_time(tol=0)
            diagnostics['autocorr_time'] = {
                name: float(tau[i])
                for i, name in enumerate(self.param_names)
            }
        except Exception as exc:
            diagnostics['autocorr_time_error'] = str(exc)
        return diagnostics

    def summary(self):
        if self.best_params is None:
            print('No fit result yet.')
            return
        chi2r = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        print(f'\nBest chi2 = {self.best_chi2:.4f}  (reduced chi2 = {chi2r:.2f})')
        print(f'Evaluations: {self.n_evaluations}')
        print(f'\n{"Parameter":<15} {"Value":>14}')
        print('-' * 30)
        for k, v in self.best_params.items():
            print(f'{k:<15} {v:>14.6g}')
        print(self.format_total_mass())
        print()

    def get_total_mass(self, values=None, use_mcmc_best=False):
        """Return the surviving dust mass in Earth masses.

        By default this uses ``best_params`` from the SciPy initialization or
        best available plotted point.  Set ``use_mcmc_best=True`` to use the
        best posterior sample after ``run_mcmc``.
        """
        if values is None:
            values = (
                self.mcmc_best_params
                if use_mcmc_best and self.mcmc_best_params is not None
                else self.best_params)
        return single_component_total_mass(
            self.sed_obj, values, self._complete_parameters,
            missing_message='Run fit/MCMC first or pass parameter values.')

    def format_total_mass(self, values=None, label='Total dust mass',
                          use_mcmc_best=False):
        """Return a one-line mass summary for logs and output files."""
        return format_total_mass(
            self.get_total_mass(values, use_mcmc_best=use_mcmc_best),
            label=label)

    def mcmc_summary(self):
        if self.mcmc_param_summary is None:
            print('No MCMC result yet.')
            return
        dof = max(len(self.obs) - self.ndim, 1)
        mcmc_chi2_red = self.mcmc_best_chi2 / dof
        best_chi2_red = self.best_chi2 / dof
        print('\nMCMC posterior summary:')
        print(f'{"Parameter":<15} {"median":>14} {"-1sigma":>14} {"+1sigma":>14}')
        print('-' * 61)
        for name in self.param_names:
            vals = self.mcmc_param_summary[name]
            print(f'{name:<15} {vals["median"]:>14.6g} '
                  f'{vals["minus_1sigma"]:>14.6g} '
                  f'{vals["plus_1sigma"]:>14.6g}')
        print(
            f'\nBest sampled chi2 = {self.mcmc_best_chi2:.6g}  '
            f'(reduced chi2 = {mcmc_chi2_red:.6g}, dof = {dof})')
        print(
            f'Best plotted chi2 = {self.best_chi2:.6g}  '
            f'(reduced chi2 = {best_chi2_red:.6g})')
        print('\nBest sampled point:')
        for k, v in self.mcmc_best_params.items():
            print(f'  {k:<15} {v:>14.6g}')
        print(self.format_total_mass(
            use_mcmc_best=True, label='Total dust mass at best sampled point'))
        print()

    def plot_best_fit(self, wavelengths_plot=None):
        if self.best_params is None:
            raise RuntimeError('Run fit() or MCMC first.')
        idx = np.argsort(self.wavelengths)
        full = self._complete_parameters(self.best_params)
        therm, scat = self.sed_obj.get_SED(
            keep_separate_fluxes=True, **full)
        model = np.real(therm) + np.real(scat)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(self.wavelengths[idx], self.obs[idx], yerr=self.obs_err[idx],
                    fmt='o', color='black', capsize=4, zorder=5,
                    label='Observations')
        ax.plot(self.wavelengths[idx], np.real(therm)[idx], '--', color='red',
                lw=1.5, label='Thermal')
        ax.plot(self.wavelengths[idx], np.real(scat)[idx], '--', color='blue',
                lw=1.5, label='Scattered')
        ax.plot(self.wavelengths[idx], model[idx], '-', color='black', lw=2,
                label='Total')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength [um]')
        ax.set_ylabel('Flux [Jy]')
        chi2r = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        ax.set_title(f'Best-fit SED  (reduced chi2 = {chi2r:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def mcmc_walkers_plot(self, discard=0, thin=1, max_walkers=None,
                          alpha=0.35, **plot_kwargs):
        if self.mcmc_chain is None:
            raise RuntimeError('Run run_mcmc() first.')
        chain = self.mcmc_chain[discard::thin]
        if chain.size == 0:
            raise ValueError('No samples left after discard/thin.')
        nsteps, nwalkers, _ = chain.shape
        walker_idx = np.arange(nwalkers)
        if max_walkers is not None and nwalkers > max_walkers:
            walker_idx = np.linspace(0, nwalkers - 1, max_walkers, dtype=int)

        fig, axes = plt.subplots(
            self.ndim, 1, figsize=(10, max(2.0 * self.ndim, 3.0)),
            sharex=True, squeeze=False)
        x = np.arange(discard, discard + nsteps * thin, thin)
        defaults = dict(color='black', lw=0.6, alpha=alpha)
        defaults.update(plot_kwargs)
        for i, name in enumerate(self.param_names):
            ax = axes[i, 0]
            for w in walker_idx:
                ax.plot(x, chain[:, w, i], **defaults)
            ax.set_ylabel(name)
            lo, hi = self.prior_ranges[name]
            ax.axhline(lo, color='tab:red', lw=0.7, alpha=0.5)
            ax.axhline(hi, color='tab:red', lw=0.7, alpha=0.5)
            vals = chain[:, walker_idx, i]
            if name in self.log_params and np.all(vals > 0):
                ax.set_yscale('log')
            ax.grid(True, alpha=0.25)
        axes[-1, 0].set_xlabel('MCMC step')
        fig.suptitle('MCMC walker traces')
        fig.tight_layout()
        return fig

    def mcmc_corner_plot(self, max_samples=None, seed=42, **corner_kwargs):
        if self.flat_samples is None:
            raise RuntimeError('Run run_mcmc() first.')
        samples = self.flat_samples
        if max_samples is not None and len(samples) > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(samples), size=max_samples, replace=False)
            samples = samples[np.sort(idx)]
        defaults = dict(title_kwargs={'fontsize': 12})
        if self.best_params is not None:
            defaults['truths'] = [self.best_params[k] for k in self.param_names]
        defaults.update(corner_kwargs)
        return make_corner_plot(
            samples, self.param_names, self.log_params, **defaults)
