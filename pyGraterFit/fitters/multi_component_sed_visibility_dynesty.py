"""Dynamic nested sampling for additive SED plus image-V2 models."""

from pathlib import Path

import numpy as np

from pyGraterFit.fitters.multi_component_sed_correlated_flux_dynesty import (
    _weighted_quantile,
)
from pyGraterFit.fitters.single_ring_sed_correlated_flux_dynesty import (
    LOG_LIKELIHOOD_FLOOR,
)
from pyGraterFit.fitters.multi_component_sed_visibility_mcmc import (
    SEDVisibilityMCMCFitter,
    vis2_from_vlti_loader,
)
from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.utils.dynesty_backend import resample_equal, run_dynesty


class SEDVisibilityNestedFitter:
    """Nested sampler for image-based V2 fits, optionally including an SED.

    The constructor is the same as :class:`SEDVisibilityMCMCFitter`.  Omit
    ``sed_wavelengths``, ``sed_fluxes``, and ``sed_flux_errors`` for V2-only
    fitting.
    """

    def __init__(self, *args, include_likelihood_normalization=True, **kwargs):
        self.component_fitter = SEDVisibilityMCMCFitter(*args, **kwargs)
        self.include_likelihood_normalization = bool(
            include_likelihood_normalization)
        self._mirror_component_fitter_state()
        self.result = None
        self.samples = None
        self.weights = None
        self.log_likelihood_values = None
        self.equal_weight_samples = None
        self.best_params = None
        self.best_chi2 = np.inf
        self.best_chi2_components = None
        self.posterior_summary = None
        self.log_evidence = None
        self.log_evidence_error = None
        self.n_likelihood_calls = 0
        self.sampler = None
        self.sampling_diagnostics = None

    def _mirror_component_fitter_state(self):
        self.components = self.component_fitter.components
        self.component_names = self.component_fitter.component_names
        self.param_names = self.component_fitter.param_names
        self.ndim = self.component_fitter.ndim
        self.log_params = self.component_fitter.log_params
        self.prior_ranges = self.component_fitter.prior_ranges
        self.fit_sed = self.component_fitter.fit_sed
        self.n_sed_points = self.component_fitter.n_sed_points
        self.n_vis2_points = self.component_fitter.n_vis2_points
        self.image_wavelengths_micron = (
            self.component_fitter.image_wavelengths_micron)
        self.vis2 = self.component_fitter.vis2
        self.wavelengths = self.component_fitter.wavelengths

    def close(self):
        self.component_fitter.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sampler'] = None
        return state

    def prior_transform(self, unit_cube):
        unit_cube = np.asarray(unit_cube, dtype=np.float64)
        physical = np.empty(self.ndim, dtype=np.float64)
        for index, entry in enumerate(self.component_fitter._entries):
            low, high = entry['prior']
            if entry['log']:
                physical[index] = 10.0 ** (
                    np.log10(low)
                    + unit_cube[index] * (np.log10(high) - np.log10(low)))
            else:
                physical[index] = low + unit_cube[index] * (high - low)
        return physical

    def _vector_to_values(self, physical_values):
        return self.component_fitter._vector_to_values(
            physical_values, optimizer_space=False)

    def chi_squared_physical(self, values):
        return self.component_fitter.chi_squared_physical(values)

    def chi2_breakdown(self, values=None):
        return self.component_fitter.chi2_breakdown(values)

    def log_likelihood(self, physical_values):
        self.n_likelihood_calls += 1
        try:
            values = self._vector_to_values(physical_values)
            chi2 = self.chi_squared_physical(values)
        except Exception as exc:
            if self.n_likelihood_calls <= 5:
                print(f'[ERROR] additive SED/V2 likelihood failed: {exc}')
            return LOG_LIKELIHOOD_FLOOR
        if not np.isfinite(chi2):
            return LOG_LIKELIHOOD_FLOOR
        if self.include_likelihood_normalization:
            log_norm = -np.sum(np.log(
                self.component_fitter.vis2['error'] * np.sqrt(2.0 * np.pi)))
            if self.component_fitter.fit_sed:
                log_norm -= np.sum(np.log(
                    self.component_fitter.obs_err * np.sqrt(2.0 * np.pi)))
        else:
            log_norm = 0.0
        log_likelihood = log_norm - 0.5 * chi2
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
        self.best_chi2_components = self.chi2_breakdown(self.best_params)
        self.best_chi2 = self.best_chi2_components['objective']
        self.component_fitter.best_params = self.best_params
        self.component_fitter.best_chi2 = self.best_chi2
        self.component_fitter.best_chi2_components = self.best_chi2_components
        self.posterior_summary = {}
        for index, name in enumerate(self.param_names):
            q16, median, q84 = _weighted_quantile(
                self.samples[:, index], self.weights, [0.16, 0.5, 0.84])
            self.posterior_summary[name] = {
                'median': float(median),
                'minus_1sigma': float(median - q16),
                'plus_1sigma': float(q84 - median),
                'q16': float(q16),
                'q84': float(q84),
            }
        self.equal_weight_samples = None

    def run(self, npoints=500, method='multi', dlogz=0.1, maxiter=None,
            maxcall=None, seed=8, update_interval=None, progress=True,
            dynamic=True, sample='rslice', checkpoint_file=None,
            checkpoint_every=300, resume=False, walks=None, slices=None,
            n_effective=None, maxbatch=None):
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
        return self.run(
            checkpoint_file=checkpoint_file, resume=True, **run_kwargs)

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
        if self.best_params is None:
            print('No nested result yet.')
            return
        print('\nBest nested SED/V2 fit:')
        self.component_fitter._print_chi2_breakdown(
            self.chi2_breakdown(self.best_params))
        print(f'logz = {self.log_evidence:.6g} +/- '
              f'{self.log_evidence_error:.3g}')
        for name, values in self.posterior_summary.items():
            print(
                f'{name:<28} {values["median"]:>14.6g} '
                f'-{values["minus_1sigma"]:>10.4g} '
                f'+{values["plus_1sigma"]:>10.4g}')
        if include_mass_abundances:
            print(self.format_component_mass_abundances(), end='')


AdditiveSEDVisibilityNestedFitter = SEDVisibilityNestedFitter
MultiComponentSEDVisibilityNestedFitter = SEDVisibilityNestedFitter


__all__ = [
    'SEDVisibilityNestedFitter',
    'AdditiveSEDVisibilityNestedFitter',
    'MultiComponentSEDVisibilityNestedFitter',
    'vis2_from_vlti_loader',
]
