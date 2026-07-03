#%%
"""
sed_scipy.py - optimized SciPy-only SED fitter
==============================================

This is the optimized simple fitter for users who only want a scipy best fit,
not MCMC.  It keeps the original fixed/free parameter dictionary convention,
and uses the optimized SED engine's total-flux path for the objective.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing, minimize

from pyGrater import CachedSED
from pyGraterFit.utils.mass import (
    format_total_mass, single_component_total_mass)
from pyGraterFit.utils.parameter_handling import (
    resolve_parameters, split_parameter_specifications)


LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}


class SEDFitter:
    """SciPy-only fitter using the optimized SED engine."""

    def __init__(self, grain, star, density_distribution, size_distribution,
                 scattering_phase_function, wavelengths, fluxes, fluxes_err,
                 params,
                 method='Nelder-Mead', use_log_params=True,
                 N_distances=8000, sed_model_class=CachedSED,
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
                if lo <= 0 or hi <= 0:
                    raise ValueError(
                        f'Log-space parameter {name} needs positive bounds.')
                self._bounds.append((np.log10(lo), np.log10(hi)))
            else:
                self._bounds.append((lo, hi))

        print(f'Free parameters ({self.ndim}): {self.free_params_range}')
        print(f'Fixed parameters: {self.fixed_params_value}')
        print(f'Dependent parameters: {list(self.dependent_params)}')
        print(f'Log-space parameters: {self.log_params}')
        print(f'Method: {method}')

        self.n_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None
        self.param_errors = None

    def _to_dict(self, x):
        d = {}
        for i, name in enumerate(self.param_names):
            d[name] = 10**x[i] if name in self.log_params else x[i]
        return d

    def _to_vec(self, d):
        return np.array([
            np.log10(d[k]) if k in self.log_params else d[k]
            for k in self.param_names])

    def _complete_parameters(self, free_parameters):
        return resolve_parameters(
            free_parameters, self.fixed_params_value, self.dependent_params)

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

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')
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

    def get_total_mass(self, values=None):
        """Return the surviving dust mass in Earth masses.

        The calculation is delegated to ``SED.get_total_mass`` so SciPy,
        MCMC, and nested fitters all use the same pyGrater mass convention.
        ``values`` may be supplied as a free-parameter dictionary; otherwise
        the current best-fit parameters are used.
        """
        values = self.best_params if values is None else values
        return single_component_total_mass(
            self.sed_obj, values, self._complete_parameters,
            missing_message='Run fit() first or pass parameter values.')

    def format_total_mass(self, values=None, label='Total dust mass'):
        """Return a one-line mass summary for logs and output files."""
        return format_total_mass(self.get_total_mass(values), label=label)

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
