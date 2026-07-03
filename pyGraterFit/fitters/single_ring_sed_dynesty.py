"""Single-ring, single-composition optimized SED fitter using nested sampling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyGrater import CachedSED
from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.utils.dynesty_backend import (
    resample_equal, run_dynesty)
from pyGraterFit.utils.parameter_handling import (
    resolve_parameters, split_parameter_specifications)


LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}
FINITE_CHI2_CEILING = 2.0e6
MAX_SAFE_RESIDUAL = np.sqrt(FINITE_CHI2_CEILING)
LOG_LIKELIHOOD_FLOOR = -1.0e6


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


class SEDNestedFitter:
    """Fit one optimized SED with dynamic ``dynesty`` nested sampling.

    Parameter dictionary syntax is identical to the optimized scipy/MCMC fitters:
    scalars are fixed, two-value ranges are priors, and callables are
    dependencies such as ``h0=lambda p: 0.05 * p['r0']``.
    """

    def __init__(
            self, grain, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, fluxes, fluxes_err,
            params, use_log_params=True, N_distances=800,
            sed_model_class=CachedSED, sed_model_kwargs=None,
            include_likelihood_normalization=True):
        self.grain = grain
        self.star = star
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.obs = np.asarray(fluxes, dtype=np.float64)
        self.obs_err = np.asarray(fluxes_err, dtype=np.float64)
        if not (self.wavelengths.shape == self.obs.shape == self.obs_err.shape):
            raise ValueError('wavelengths, fluxes, and fluxes_err must match.')
        if np.any(~np.isfinite(self.obs_err)) or np.any(self.obs_err <= 0):
            raise ValueError('All observational errors must be finite and positive.')
        self._inverse_obs_err = 1.0 / self.obs_err
        self._log_likelihood_normalization = (
            -np.sum(np.log(self.obs_err * np.sqrt(2.0 * np.pi)))
            if include_likelihood_normalization else 0.0)

        sed_model_kwargs = dict(sed_model_kwargs or {})
        sed_model_kwargs.setdefault('N_distances', N_distances)
        self.sed_obj = sed_model_class(
            grain, star, density_distribution, size_distribution,
            self.wavelengths, **sed_model_kwargs)

        (self.free_params_range, self.fixed_params_value,
         self.dependent_params) = split_parameter_specifications(params)
        self.param_names = list(self.free_params_range)
        self.ndim = len(self.param_names)
        if self.ndim == 0:
            raise ValueError('Nested sampling requires at least one free parameter.')
        self.log_params = (
            set(self.param_names).intersection(LOG_SPACE_PARAMS)
            if use_log_params else set())
        for name in self.log_params:
            low, high = self.free_params_range[name]
            if low <= 0 or high <= 0:
                raise ValueError(
                    f'Log-space parameter {name} needs positive prior bounds.')

        self.result = None
        self.samples = None
        self.weights = None
        self.log_likelihood_values = None
        self.equal_weight_samples = None
        self.best_params = None
        self.best_chi2 = np.inf
        self.posterior_summary = None
        self.log_evidence = None
        self.log_evidence_error = None
        self.n_likelihood_calls = 0
        self.sampler = None
        self.sampling_diagnostics = None

        print(f'Free parameters ({self.ndim}): {self.free_params_range}')
        print(f'Fixed parameters: {self.fixed_params_value}')
        print(f'Dependent parameters: {list(self.dependent_params)}')
        print(f'Log-uniform prior parameters: {self.log_params}')

    def _vector_to_dict(self, values):
        return {
            name: float(values[index])
            for index, name in enumerate(self.param_names)}

    def _complete_parameters(self, free_parameters):
        return resolve_parameters(
            free_parameters, self.fixed_params_value, self.dependent_params)

    def prior_transform(self, unit_cube):
        """Transform the unit cube into physical parameters."""
        unit_cube = np.asarray(unit_cube, dtype=np.float64)
        physical = np.empty(self.ndim, dtype=np.float64)
        for index, name in enumerate(self.param_names):
            low, high = self.free_params_range[name]
            if name in self.log_params:
                physical[index] = 10.0 ** (
                    np.log10(low) + unit_cube[index]
                    * (np.log10(high) - np.log10(low)))
            else:
                physical[index] = low + unit_cube[index] * (high - low)
        return physical

    def chi_squared_physical(self, free_parameters):
        parameters = self._complete_parameters(free_parameters)
        model = self.sed_obj.get_SED(
            keep_separate_fluxes=False, **parameters)
        if not np.all(np.isfinite(model)):
            return FINITE_CHI2_CEILING
        residual = (self.obs - np.real(model)) * self._inverse_obs_err
        if not np.all(np.isfinite(residual)):
            return FINITE_CHI2_CEILING
        if np.max(np.abs(residual)) > MAX_SAFE_RESIDUAL:
            return FINITE_CHI2_CEILING
        chi2 = float(np.dot(residual, residual))
        if not np.isfinite(chi2):
            return FINITE_CHI2_CEILING
        return min(chi2, FINITE_CHI2_CEILING)

    def log_likelihood(self, physical_values):
        self.n_likelihood_calls += 1
        try:
            chi2 = self.chi_squared_physical(
                self._vector_to_dict(physical_values))
        except Exception as exc:
            if self.n_likelihood_calls <= 5:
                print(f'[ERROR] nested likelihood evaluation failed: {exc}')
            return LOG_LIKELIHOOD_FLOOR
        if not np.isfinite(chi2):
            return LOG_LIKELIHOOD_FLOOR
        log_likelihood = self._log_likelihood_normalization - 0.5 * chi2
        if not np.isfinite(log_likelihood):
            return LOG_LIKELIHOOD_FLOOR
        return max(float(log_likelihood), LOG_LIKELIHOOD_FLOOR)

    def _set_results(self, samples, weights, log_likelihood_values,
                     log_evidence, log_evidence_error, result=None):
        self.result = result
        self.samples = np.asarray(samples, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.log_likelihood_values = np.asarray(
            log_likelihood_values, dtype=np.float64)
        self.log_evidence = float(log_evidence)
        self.log_evidence_error = float(log_evidence_error)

        best_index = int(np.nanargmax(self.log_likelihood_values))
        self.best_params = self._vector_to_dict(self.samples[best_index])
        self.best_chi2 = float(
            -2.0 * (self.log_likelihood_values[best_index]
                    - self._log_likelihood_normalization))
        self.posterior_summary = {}
        for index, name in enumerate(self.param_names):
            q16, median, q84 = _weighted_quantile(
                self.samples[:, index], self.weights, [0.16, 0.5, 0.84])
            self.posterior_summary[name] = {
                'median': float(median),
                'minus_1sigma': float(median - q16),
                'plus_1sigma': float(q84 - median),
                'q16': float(q16), 'q84': float(q84),
            }
        self.equal_weight_samples = None

    def run(self, npoints=300, method='multi', dlogz=0.1, maxiter=None,
            maxcall=None, seed=8, update_interval=None, progress=True,
            dynamic=True, sample='rslice', checkpoint_file=None,
            checkpoint_every=300, resume=False, walks=None, slices=None,
            n_effective=None, maxbatch=None):
        """Run or resume dynesty nested sampling.

        ``method`` is retained as the bounding method for compatibility with
        previous scripts; use ``dynamic=False`` for static nested sampling.
        """
        if method == 'classic':
            method = 'none'
        if method not in {'none', 'single', 'multi', 'balls', 'cubes'}:
            raise ValueError('Unknown dynesty bounding method.')
        self.n_likelihood_calls = 0
        (self.sampler, self.result, weights,
         self.sampling_diagnostics) = run_dynesty(
            self.log_likelihood, self.prior_transform, self.ndim,
            npoints=npoints, bound=method, sample=sample, dynamic=dynamic,
            dlogz=dlogz, maxiter=maxiter, maxcall=maxcall, seed=seed,
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every, resume=resume,
            progress=progress, update_interval=update_interval, walks=walks,
            slices=slices, n_effective=n_effective, maxbatch=maxbatch)
        self.n_likelihood_calls = self.sampling_diagnostics[
            'n_likelihood_calls']
        self._set_results(
            self.result.samples, weights, self.result.logl,
            self.result.logz[-1], self.result.logzerr[-1],
            result=self.result)
        return self.result

    def resume_backend_nested(self, checkpoint_file, **run_kwargs):
        """Continue a dynesty checkpoint using this fitter's likelihood."""
        return self.run(
            checkpoint_file=checkpoint_file, resume=True, **run_kwargs)

    def plot_nested_diagnostics(
            self, output_directory, prefix='single_ring_nested',
            max_corner_samples=50000, seed=8):
        """Save trace, likelihood/weight, and corner diagnostics."""
        from pyGraterFit.utils.nested_plotting import (
            plot_nested_results)
        return plot_nested_results(
            self, output_directory, prefix=prefix,
            max_corner_samples=max_corner_samples, seed=seed)

    def posterior_samples(self, max_samples=None, seed=8):
        """Return an equal-weight posterior sample for plotting."""
        if self.samples is None:
            raise RuntimeError('Run nested sampling or load results first.')
        equal = resample_equal(self.samples, self.weights, seed=seed)
        if max_samples is not None and len(equal) > int(max_samples):
            rng = np.random.RandomState(seed)
            equal = equal[rng.choice(
                len(equal), size=int(max_samples), replace=False)]
        self.equal_weight_samples = equal
        return equal

    def save_results(self, filename):
        if self.samples is None:
            raise RuntimeError('Run nested sampling before saving results.')
        filename = Path(filename)
        np.savez_compressed(
            filename, samples=self.samples, weights=self.weights,
            log_likelihood=self.log_likelihood_values,
            param_names=np.asarray(self.param_names),
            log_params=np.asarray(sorted(self.log_params)),
            log_evidence=self.log_evidence,
            log_evidence_error=self.log_evidence_error,
            n_likelihood_calls=self.n_likelihood_calls)
        return filename

    def load_results(self, filename):
        with np.load(filename, allow_pickle=False) as saved:
            saved_names = list(saved['param_names'].astype(str))
            if saved_names != self.param_names:
                raise ValueError(
                    f'Saved parameters {saved_names} do not match '
                    f'{self.param_names}.')
            saved_log = set(saved['log_params'].astype(str))
            if saved_log != self.log_params:
                raise ValueError('Saved log-prior parameters do not match fitter.')
            self.n_likelihood_calls = int(saved['n_likelihood_calls'])
            self._set_results(
                saved['samples'], saved['weights'], saved['log_likelihood'],
                saved['log_evidence'], saved['log_evidence_error'])
        return self

    def corner_plot(self, max_samples=50000, seed=8, **corner_kwargs):
        samples = self.posterior_samples(max_samples=max_samples, seed=seed)
        return make_corner_plot(
            samples, self.param_names, self.log_params, **corner_kwargs)

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run nested sampling or load results first.')
        parameters = self._complete_parameters(self.best_params)
        thermal, scattered = self.sed_obj.get_SED(
            keep_separate_fluxes=True, **parameters)
        thermal, scattered = np.real(thermal), np.real(scattered)
        total = thermal + scattered
        order = np.argsort(self.wavelengths)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(
            self.wavelengths[order], self.obs[order],
            yerr=self.obs_err[order], fmt='o', color='black', capsize=4,
            zorder=5, label='Observations')
        ax.plot(self.wavelengths[order], thermal[order], '--', color='red',
                label='Thermal')
        ax.plot(self.wavelengths[order], scattered[order], '--', color='blue',
                label='Scattered')
        ax.plot(self.wavelengths[order], total[order], color='black',
                linewidth=2, label='Total')
        ax.set(xscale='log', yscale='log', xlabel='Wavelength [um]',
               ylabel='Flux [Jy]')
        reduced = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        ax.set_title(f'Best nested sample (reduced chi2 = {reduced:.3g})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def get_total_mass(self):
        if self.best_params is None:
            raise RuntimeError('Run nested sampling or load results first.')
        parameters = self._complete_parameters(self.best_params)
        if getattr(self.sed_obj, 'sizes_for_integral', None) is None:
            self.sed_obj.get_SED(keep_separate_fluxes=False, **parameters)
        return self.sed_obj.get_total_mass(**parameters)

    def summary(self):
        if self.posterior_summary is None:
            print('No nested-sampling result yet.')
            return
        degrees_of_freedom = max(len(self.obs) - self.ndim, 1)
        print(f'log(Z) = {self.log_evidence:.8g} '
              f'+/- {self.log_evidence_error:.3g}')
        print(f'Best chi2 = {self.best_chi2:.8g}')
        print(f'Reduced chi2 = {self.best_chi2 / degrees_of_freedom:.8g}')
        print(f'Likelihood calls = {self.n_likelihood_calls}')
        if self.sampling_diagnostics is not None:
            print(
                'Calls / saved sample = '
                f'{self.sampling_diagnostics["calls_per_iteration"]:.4g}')
            print(
                'Sampling efficiency = '
                f'{self.sampling_diagnostics["sampling_efficiency_percent"]:.3g}%')
            print(
                'Effective posterior samples = '
                f'{self.sampling_diagnostics["effective_sample_size"]:.4g}')
        print(f'\n{"Parameter":<18} {"median":>14} {"-1sigma":>14} '
              f'{"+1sigma":>14} {"best chi2":>14}')
        print('-' * 78)
        for name in self.param_names:
            values = self.posterior_summary[name]
            print(f'{name:<18} {values["median"]:>14.6g} '
                  f'{values["minus_1sigma"]:>14.6g} '
                  f'{values["plus_1sigma"]:>14.6g} '
                  f'{self.best_params[name]:>14.6g}')
