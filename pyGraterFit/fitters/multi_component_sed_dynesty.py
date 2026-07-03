"""Nested sampling for any number of additive optimized SED components."""

from pathlib import Path

import numpy as np

from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.fitters.multi_component_sed_mcmc import (
    AdditiveSEDMCMCFitter)
from pyGraterFit.utils.dynesty_backend import (
    resample_equal, run_dynesty)


# Keep rejected samples finite without pushing dynesty's evidence arithmetic
# into extreme-number pathologies.  This is far below any useful posterior
# region for the HD113766 SED fits, but not so low that logz/logzerr overflow.
LOG_LIKELIHOOD_FLOOR = -1.0e6


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


class AdditiveSEDNestedFitter:
    """Fit additive rings/compositions with dynamic ``dynesty``.

    This uses the same component dictionaries, shared parameters, group-shared
    parameters, and dependent-parameter callables as the additive optimized MCMC
    fitter. Each component must retain its own ``A_norm``.
    """

    def __init__(
            self, components=None, star=None, density_distribution=None,
            size_distribution=None, scattering_phase_function=None,
            wavelengths=None, fluxes=None, fluxes_err=None,
            params_by_component=None, shared_parameter_names=(),
            prior_ranges_by_component=None, shared_prior_ranges=None,
            use_log_params=True, N_distances=400,
            parallel_components='auto', max_component_workers=2,
            component_groups=None, group_shared_parameter_names=(),
            sed_model_class=None, sed_model_kwargs=None,
            mass_abundance_groups=None,
            include_likelihood_normalization=True,
            normalization_mode='independent',
            normalization_groups=None,
            normalization_total_ranges=None,
            materials=None, ring_params=None,
            normalization_range=(1e25, 1e38),
            mass_abundance_by_ring=True):
        component_arguments = {}
        if sed_model_class is not None:
            component_arguments['sed_model_class'] = sed_model_class
        self.component_fitter = AdditiveSEDMCMCFitter(
            components, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, fluxes, fluxes_err,
            params_by_component,
            shared_parameter_names=shared_parameter_names,
            prior_ranges_by_component=prior_ranges_by_component,
            shared_prior_ranges=shared_prior_ranges,
            use_log_params=use_log_params, N_distances=N_distances,
            parallel_components=parallel_components,
            max_component_workers=max_component_workers,
            component_groups=component_groups,
            group_shared_parameter_names=group_shared_parameter_names,
            sed_model_kwargs=sed_model_kwargs,
            mass_abundance_groups=mass_abundance_groups,
            normalization_mode=normalization_mode,
            normalization_groups=normalization_groups,
            normalization_total_ranges=normalization_total_ranges,
            materials=materials,
            ring_params=ring_params,
            normalization_range=normalization_range,
            mass_abundance_by_ring=mass_abundance_by_ring,
            **component_arguments)

        self.components = self.component_fitter.components
        self.component_names = self.component_fitter.component_names
        self.param_names = self.component_fitter.param_names
        self.ndim = self.component_fitter.ndim
        self.log_params = self.component_fitter.log_params
        self.prior_ranges = self.component_fitter.prior_ranges
        self.obs = self.component_fitter.obs
        self.obs_err = self.component_fitter.obs_err
        self.wavelengths = self.component_fitter.wavelengths
        self._log_likelihood_normalization = (
            -np.sum(np.log(self.obs_err * np.sqrt(2.0 * np.pi)))
            if include_likelihood_normalization else 0.0)

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

    def close(self):
        self.component_fitter.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Avoid recursive/pickle-hostile checkpoint state. dynesty pickles the
        # sampler separately; keeping it on the likelihood object can also pull
        # in thread locks through the active sampler/executor.
        state['sampler'] = None
        return state

    def prior_transform(self, unit_cube):
        """Transform a unit cube into the configured physical priors."""
        unit_cube = np.asarray(unit_cube, dtype=np.float64)
        physical = np.empty(self.ndim, dtype=np.float64)
        for index, entry in enumerate(self.component_fitter._entries):
            low, high = entry['prior']
            if entry['log']:
                physical[index] = 10.0 ** (
                    np.log10(low) + unit_cube[index]
                    * (np.log10(high) - np.log10(low)))
            else:
                physical[index] = low + unit_cube[index] * (high - low)
        return physical

    def _vector_to_values(self, physical_values):
        return self.component_fitter._vector_to_values(
            physical_values, optimizer_space=False)

    def chi_squared_physical(self, values):
        return self.component_fitter.chi_squared_physical(values)

    def log_likelihood(self, physical_values):
        self.n_likelihood_calls += 1
        chi2 = self.chi_squared_physical(
            self._vector_to_values(physical_values))
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
        self.best_params = self._vector_to_values(self.samples[best_index])
        self.best_chi2 = float(
            -2.0 * (self.log_likelihood_values[best_index]
                    - self._log_likelihood_normalization))
        self.component_fitter.best_params = self.best_params
        self.component_fitter.best_chi2 = self.best_chi2
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

    def run(self, npoints=500, method='multi', dlogz=0.1, maxiter=None,
            maxcall=None, seed=8, update_interval=None, progress=True,
            dynamic=True, sample='rslice', checkpoint_file=None,
            checkpoint_every=300, resume=False, walks=None, slices=None,
            n_effective=None, maxbatch=None):
        """Run or resume dynesty nested sampling."""
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
            self, output_directory, prefix='multi_component_nested',
            max_corner_samples=50000, seed=8):
        """Save trace, likelihood/weight, and corner diagnostics."""
        from pyGraterFit.utils.nested_plotting import (
            plot_nested_results)
        return plot_nested_results(
            self, output_directory, prefix=prefix,
            max_corner_samples=max_corner_samples, seed=seed)

    def posterior_samples(self, max_samples=None, seed=8):
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
            names = list(saved['param_names'].astype(str))
            if names != self.param_names:
                raise ValueError(
                    f'Saved parameters {names} do not match {self.param_names}.')
            if set(saved['log_params'].astype(str)) != self.log_params:
                raise ValueError('Saved log-prior parameters do not match fitter.')
            self.n_likelihood_calls = int(saved['n_likelihood_calls'])
            self._set_results(
                saved['samples'], saved['weights'], saved['log_likelihood'],
                saved['log_evidence'], saved['log_evidence_error'])
        return self

    def corner_plot(self, max_samples=50000, seed=8, **corner_kwargs):
        return make_corner_plot(
            self.posterior_samples(max_samples=max_samples, seed=seed),
            self.param_names, self.log_params, **corner_kwargs)

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run nested sampling or load results first.')
        return self.component_fitter.plot_best_fit()

    def get_component_masses(self):
        if self.best_params is None:
            raise RuntimeError('Run nested sampling or load results first.')
        return self.component_fitter.get_component_masses(self.best_params)

    def get_component_mass_abundances(self, masses=None):
        return self.component_fitter.get_component_mass_abundances(
            self.best_params, masses=masses)

    def format_component_mass_abundances(self, masses=None):
        return self.component_fitter.format_component_mass_abundances(
            self.best_params, masses=masses)

    def summary(self, include_mass_abundances=True):
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
        print(f'\n{"Parameter":<32} {"median":>14} {"-1sigma":>14} '
              f'{"+1sigma":>14} {"best chi2":>14}')
        print('-' * 92)
        best_vector = self.component_fitter._values_to_vector(self.best_params)
        for index, name in enumerate(self.param_names):
            values = self.posterior_summary[name]
            print(f'{name:<32} {values["median"]:>14.6g} '
                  f'{values["minus_1sigma"]:>14.6g} '
                  f'{values["plus_1sigma"]:>14.6g} '
                  f'{best_vector[index]:>14.6g}')
        if include_mass_abundances:
            print(self.format_component_mass_abundances(), end='')
