"""Nested fitter for a pyGrater SED and analytical correlated fluxes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyGrater import CachedSED
from pyGraterFit.utils.dynesty_backend import (
    resample_equal, run_dynesty)
from pyGraterFit.utils.analytical_visibilities import (
    analytical_disk_visibility,
    correlated_flux_from_components,
)
from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.utils.parameter_handling import (
    resolve_parameters,
    split_parameter_specifications,
)
from pyGraterFit.utils.interferometry import (
    uniform_disk_argument_per_mas,
    uniform_disk_visibility_from_argument,
)


STELLAR_DIAMETER_PARAMETER = 'stellar_angular_diameter_mas'
RING_FWHM_PARAMETER = 'ring_fwhm_au'
ANALYTICAL_ONLY_PARAMETERS = {
    STELLAR_DIAMETER_PARAMETER,
    RING_FWHM_PARAMETER,
}
LOG_SPACE_PARAMETERS = {
    'M_tot', 'A_norm', 'a_min', 'r0', RING_FWHM_PARAMETER,
}
FINITE_CHI2_CEILING = 2.0e6
MAX_SAFE_RESIDUAL = np.sqrt(FINITE_CHI2_CEILING)
LOG_LIKELIHOOD_FLOOR = -1.0e6


def _safe_chi_squared_from_residual(residual):
    """Return a finite chi-squared for numerically extreme residual arrays."""
    residual = np.asarray(residual, dtype=np.float64)
    if not np.all(np.isfinite(residual)):
        return FINITE_CHI2_CEILING
    if residual.size and np.max(np.abs(residual)) > MAX_SAFE_RESIDUAL:
        return FINITE_CHI2_CEILING
    chi2 = float(np.dot(residual, residual))
    if not np.isfinite(chi2):
        return FINITE_CHI2_CEILING
    return min(chi2, FINITE_CHI2_CEILING)


def _one_dimensional_float_array(values, name):
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f'{name} is empty.')
    return array


def correlated_flux_from_vlti_loader(source, **loader_arguments):
    """Return correlated-flux arrays from ``vlti_loader.Observations``.

    The preferred loader keys are ``corrflux`` and ``corrflux_err``, as
    requested for the updated loader. The current ``CorrFlux`` names are also
    accepted so this fitter works before that loader change is merged.
    Baselines use ``Bu`` (East), ``Bv`` (North), and ``VIS2_waves`` (metres).
    """
    if hasattr(source, 'data'):
        if loader_arguments:
            raise ValueError(
                'loader_arguments cannot be used with an Observations object.')
        data = source.data
    elif isinstance(source, dict):
        if loader_arguments:
            raise ValueError(
                'loader_arguments cannot be used with a data dictionary.')
        data = source
    else:
        try:
            from vlti_loader import Observations
        except ImportError as exc:
            raise ImportError(
                'Install prioletp/vlti_loader or pass an Observations object.'
            ) from exc
        data = Observations(source, **loader_arguments).data

    def first_available(*names):
        for name in names:
            if name in data:
                return data[name]
        raise KeyError(
            f'vlti_loader data need one of these keys: {list(names)}.')

    arrays = {
        'value': _one_dimensional_float_array(
            first_available('corrflux', 'CorrFlux'), 'corrflux'),
        'error': _one_dimensional_float_array(
            first_available('corrflux_err', 'CorrFlux_err'), 'corrflux_err'),
        'u_m': _one_dimensional_float_array(
            first_available('corrflux_u_m', 'Bu'), 'corrflux u coordinate'),
        'v_m': _one_dimensional_float_array(
            first_available('corrflux_v_m', 'Bv'), 'corrflux v coordinate'),
        'wavelength_m': _one_dimensional_float_array(
            first_available('corrflux_waves', 'VIS2_waves'),
            'corrflux wavelength'),
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(
            'Correlated flux, errors, baselines, and wavelengths must match.')

    valid = np.ones(arrays['value'].size, dtype=bool)
    for array in arrays.values():
        valid &= np.isfinite(array)
    valid &= arrays['error'] > 0
    valid &= arrays['wavelength_m'] > 0
    arrays = {name: array[valid] for name, array in arrays.items()}
    if arrays['value'].size == 0:
        raise ValueError('No valid correlated-flux points remain.')

    instrument = data.get('INS_CORRFLUX', data.get('INS_VIS2'))
    if instrument is not None:
        instrument = np.asarray(instrument).ravel()
        if instrument.size == valid.size:
            arrays['instrument'] = instrument[valid].astype(str)
    return arrays


def _validate_correlated_flux(source):
    if (hasattr(source, 'data')
            or isinstance(source, (str, bytes, list, tuple))
            or hasattr(source, '__fspath__')):
        return correlated_flux_from_vlti_loader(source)
    if not isinstance(source, dict):
        raise TypeError(
            'correlated_flux must be a normalized dictionary or vlti_loader source.')
    if {'value', 'error', 'u_m', 'v_m', 'wavelength_m'} <= set(source):
        normalized = source
    else:
        return correlated_flux_from_vlti_loader(source)

    arrays = {
        name: _one_dimensional_float_array(normalized[name], name)
        for name in ('value', 'error', 'u_m', 'v_m', 'wavelength_m')}
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError('All correlated-flux arrays must have the same length.')
    valid = np.ones(arrays['value'].size, dtype=bool)
    for array in arrays.values():
        valid &= np.isfinite(array)
    valid &= arrays['error'] > 0
    valid &= arrays['wavelength_m'] > 0
    arrays = {name: array[valid] for name, array in arrays.items()}
    if arrays['value'].size == 0:
        raise ValueError('No valid correlated-flux points remain.')
    instrument = normalized.get('instrument')
    if instrument is not None:
        instrument = np.asarray(instrument).ravel()
        if instrument.size != valid.size:
            raise ValueError('Instrument labels must match correlated fluxes.')
        arrays['instrument'] = instrument[valid].astype(str)
    return arrays


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


class SEDCorrelatedFluxNestedFitter:
    """Jointly fit a dust SED and absolute VLTI correlated fluxes.

    The dust flux at every interferometric wavelength comes from pyGrater.
    The disk visibility is analytical, and the stellar flux is interpolated
    from the supplied pyGrater ``Star`` spectrum. By default the objective is

        chi2_SED / N_SED + chi2_corrflux / N_corrflux,

    preserving equal average weight for the two datasets regardless of their
    number of points.
    """

    def __init__(
            self, grain, star, density_distribution, size_distribution,
            scattering_phase_function, sed_wavelengths, sed_fluxes,
            sed_flux_errors, correlated_flux, params,
            visibility_model='gaussian_ring', use_log_params=True,
            fit_sed=True, sed_includes_star=False,
            normalize_each_dataset=True,
            N_distances=800, sed_model_class=CachedSED,
            sed_model_kwargs=None):
        self.grain = grain
        self.star = star
        self.visibility_model = str(visibility_model)
        self.fit_sed = bool(fit_sed)
        self.sed_includes_star = bool(sed_includes_star)
        self.normalize_each_dataset = bool(normalize_each_dataset)

        self.correlated_flux = _validate_correlated_flux(correlated_flux)
        self._inverse_correlated_flux_error = (
            1.0 / self.correlated_flux['error'])
        correlated_log_normalization = -np.sum(np.log(
            self.correlated_flux['error'] * np.sqrt(2.0 * np.pi)))
        correlated_wavelengths_micron = (
            self.correlated_flux['wavelength_m'] * 1e6)

        if self.fit_sed:
            self.sed_wavelengths_micron = _one_dimensional_float_array(
                sed_wavelengths, 'sed_wavelengths')
            self.observed_sed_jy = _one_dimensional_float_array(
                sed_fluxes, 'sed_fluxes')
            self.sed_error_jy = _one_dimensional_float_array(
                sed_flux_errors, 'sed_flux_errors')
            if not (self.sed_wavelengths_micron.size
                    == self.observed_sed_jy.size == self.sed_error_jy.size):
                raise ValueError(
                    'SED wavelength, flux, and error arrays must match.')
            if np.any(~np.isfinite(self.observed_sed_jy)):
                raise ValueError('Observed SED contains non-finite values.')
            if np.any(~np.isfinite(self.sed_error_jy)) or np.any(
                    self.sed_error_jy <= 0):
                raise ValueError('SED errors must be finite and positive.')
            if np.any(self.sed_wavelengths_micron <= 0):
                raise ValueError('SED wavelengths must be positive.')
            self._inverse_sed_error = 1.0 / self.sed_error_jy
            sed_log_normalization = -np.sum(np.log(
                self.sed_error_jy * np.sqrt(2.0 * np.pi)))
        else:
            self.sed_wavelengths_micron = np.empty(0, dtype=np.float64)
            self.observed_sed_jy = np.empty(0, dtype=np.float64)
            self.sed_error_jy = np.empty(0, dtype=np.float64)
            self._inverse_sed_error = np.empty(0, dtype=np.float64)
            sed_log_normalization = 0.0

        if not self.fit_sed:
            self._log_likelihood_normalization = (
                correlated_log_normalization)
        elif self.normalize_each_dataset:
            self._log_likelihood_normalization = (
                sed_log_normalization / self.observed_sed_jy.size
                + correlated_log_normalization
                / self.correlated_flux['value'].size)
        else:
            self._log_likelihood_normalization = (
                sed_log_normalization + correlated_log_normalization)

        # One pyGrater call supplies both the fitted SED and the dust flux used
        # in the correlated-flux calculation.
        self.model_wavelengths_micron = np.unique(
            np.concatenate((
                self.sed_wavelengths_micron,
                correlated_wavelengths_micron)))
        self._sed_wavelength_indices = np.searchsorted(
            self.model_wavelengths_micron, self.sed_wavelengths_micron)
        self._correlated_wavelength_indices = np.searchsorted(
            self.model_wavelengths_micron, correlated_wavelengths_micron)

        sed_model_kwargs = dict(sed_model_kwargs or {})
        sed_model_kwargs.setdefault('N_distances', N_distances)
        self.sed_obj = sed_model_class(
            grain, star, density_distribution, size_distribution,
            self.model_wavelengths_micron, **sed_model_kwargs)

        self.stellar_flux_at_model_wavelengths_jy = np.interp(
            self.model_wavelengths_micron,
            np.asarray(star.waves, dtype=np.float64),
            np.asarray(star.flux, dtype=np.float64),
            left=0.0, right=0.0)
        self.stellar_flux_at_sed_wavelengths_jy = (
            self.stellar_flux_at_model_wavelengths_jy[
                self._sed_wavelength_indices])
        self.stellar_flux_at_correlated_wavelengths_jy = (
            self.stellar_flux_at_model_wavelengths_jy[
                self._correlated_wavelength_indices])

        (self.free_params_range, self.fixed_params_value,
         self.dependent_params) = split_parameter_specifications(params)
        self.param_names = list(self.free_params_range)
        self.ndim = len(self.param_names)
        if self.ndim == 0:
            raise ValueError('Nested sampling requires at least one free parameter.')
        self.log_params = (
            set(self.param_names).intersection(LOG_SPACE_PARAMETERS)
            if use_log_params else set())
        for name in self.log_params:
            low, high = self.free_params_range[name]
            if low <= 0 or high <= 0:
                raise ValueError(
                    f'Log-space parameter {name} needs positive prior bounds.')
        required = {
            'r0', RING_FWHM_PARAMETER, STELLAR_DIAMETER_PARAMETER,
        }
        missing = required.difference(params)
        if missing:
            raise ValueError(f'Missing required model parameters: {sorted(missing)}')

        self._stellar_visibility_argument_per_mas = (
            uniform_disk_argument_per_mas(
                self.correlated_flux['u_m'], self.correlated_flux['v_m'],
                self.correlated_flux['wavelength_m']))
        self._fixed_stellar_visibility = None
        if (STELLAR_DIAMETER_PARAMETER not in self.free_params_range
                and STELLAR_DIAMETER_PARAMETER not in self.dependent_params):
            diameter = self.fixed_params_value[STELLAR_DIAMETER_PARAMETER]
            self._fixed_stellar_visibility = (
                uniform_disk_visibility_from_argument(
                    self._stellar_visibility_argument_per_mas * diameter))

        self.result = None
        self.samples = None
        self.weights = None
        self.log_likelihood_values = None
        self.best_params = None
        self.best_chi2 = np.inf
        self.best_fit_statistic = np.inf
        self.best_chi2_components = None
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
        print(f'Analytical visibility model: {self.visibility_model}')
        print(f'Fit standalone SED observations: {self.fit_sed}')
        print(f'SED points in likelihood: {self.observed_sed_jy.size}')
        print(f'Correlated-flux points: {self.correlated_flux["value"].size}')

    def _vector_to_dict(self, values):
        return {
            name: float(values[index])
            for index, name in enumerate(self.param_names)}

    def _complete_parameters(self, free_parameters):
        return resolve_parameters(
            free_parameters, self.fixed_params_value, self.dependent_params)

    @staticmethod
    def _sed_parameters(parameters):
        return {
            name: value for name, value in parameters.items()
            if name not in ANALYTICAL_ONLY_PARAMETERS}

    def prior_transform(self, unit_cube):
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

    def _disk_visibility(self, parameters):
        """Evaluate the selected disk visibility model.

        Add a future model by registering its function in
        ``ANALYTICAL_DISK_VISIBILITY_MODELS`` and adding its explicit,
        readable parameter mapping here.
        """
        if self.visibility_model == 'gaussian_ring':
            return analytical_disk_visibility(
                'gaussian_ring',
                u_m=self.correlated_flux['u_m'],
                v_m=self.correlated_flux['v_m'],
                wavelengths_m=self.correlated_flux['wavelength_m'],
                distance_pc=self.star.distance,
                ring_radius_au=parameters['r0'],
                ring_fwhm_au=parameters[RING_FWHM_PARAMETER],
                inclination_degrees=parameters.get('itilt', 0.0),
                position_angle_degrees=parameters.get('PA', 0.0))
        raise ValueError(
            f'No fitter parameter mapping exists for {self.visibility_model!r}.')

    def _stellar_visibility(self, parameters):
        if self._fixed_stellar_visibility is not None:
            return self._fixed_stellar_visibility
        diameter_mas = parameters[STELLAR_DIAMETER_PARAMETER]
        if diameter_mas < 0:
            raise ValueError('Stellar angular diameter must be non-negative.')
        return uniform_disk_visibility_from_argument(
            self._stellar_visibility_argument_per_mas * diameter_mas)

    def model(self, free_parameters):
        parameters = self._complete_parameters(free_parameters)
        dust_flux_all_wavelengths_jy = np.real(self.sed_obj.get_SED(
            keep_separate_fluxes=False,
            **self._sed_parameters(parameters)))
        model_sed_jy = dust_flux_all_wavelengths_jy[
            self._sed_wavelength_indices].copy()
        if self.sed_includes_star:
            model_sed_jy += self.stellar_flux_at_sed_wavelengths_jy

        disk_flux_at_correlated_wavelengths_jy = (
            dust_flux_all_wavelengths_jy[self._correlated_wavelength_indices])
        disk_visibility = self._disk_visibility(parameters)
        stellar_visibility = self._stellar_visibility(parameters)
        model_correlated_flux_jy = correlated_flux_from_components(
            disk_flux_at_correlated_wavelengths_jy, disk_visibility,
            self.stellar_flux_at_correlated_wavelengths_jy,
            stellar_visibility)
        return {
            'sed_jy': model_sed_jy,
            'correlated_flux_jy': model_correlated_flux_jy,
            'dust_flux_at_correlated_wavelengths_jy':
                disk_flux_at_correlated_wavelengths_jy,
            'disk_visibility': disk_visibility,
            'stellar_visibility': stellar_visibility,
        }

    def chi_squared_physical(self, free_parameters, return_models=False):
        models = self.model(free_parameters)
        if not all(np.all(np.isfinite(value)) for value in models.values()):
            components = {
                'sed': FINITE_CHI2_CEILING if self.fit_sed else 0.0,
                'correlated_flux': FINITE_CHI2_CEILING,
                'fit_statistic': FINITE_CHI2_CEILING,
            }
            if return_models:
                return components, models
            return components
        correlated_residual = (
            (self.correlated_flux['value']
             - models['correlated_flux_jy'])
            * self._inverse_correlated_flux_error)
        if self.fit_sed:
            sed_residual = (
                (self.observed_sed_jy - models['sed_jy'])
                * self._inverse_sed_error)
            chi2_sed = _safe_chi_squared_from_residual(sed_residual)
        else:
            chi2_sed = 0.0
        chi2_correlated_flux = _safe_chi_squared_from_residual(
            correlated_residual)
        if not self.fit_sed:
            fit_statistic = chi2_correlated_flux
        elif self.normalize_each_dataset:
            fit_statistic = (
                chi2_sed / self.observed_sed_jy.size
                + chi2_correlated_flux / self.correlated_flux['value'].size)
        else:
            fit_statistic = chi2_sed + chi2_correlated_flux
        components = {
            'sed': chi2_sed,
            'correlated_flux': chi2_correlated_flux,
            'fit_statistic': float(fit_statistic),
        }
        if return_models:
            return components, models
        return components

    def log_likelihood(self, physical_values):
        self.n_likelihood_calls += 1
        try:
            components = self.chi_squared_physical(
                self._vector_to_dict(physical_values))
        except Exception as exc:
            if self.n_likelihood_calls <= 5:
                print(f'[ERROR] correlated-flux likelihood failed: {exc}')
            return LOG_LIKELIHOOD_FLOOR
        statistic = components['fit_statistic']
        if not np.isfinite(statistic):
            return LOG_LIKELIHOOD_FLOOR
        log_likelihood = self._log_likelihood_normalization - 0.5 * statistic
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
        self.best_chi2_components = self.chi_squared_physical(self.best_params)
        self.best_fit_statistic = self.best_chi2_components['fit_statistic']
        self.best_chi2 = (
            self.best_chi2_components['sed']
            + self.best_chi2_components['correlated_flux'])
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

    def run(self, npoints=400, method='multi', dlogz=0.1, maxiter=None,
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
            self, output_directory, prefix='correlated_flux_nested',
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
        if max_samples is not None and equal.shape[0] > int(max_samples):
            rng = np.random.RandomState(seed)
            equal = equal[rng.choice(
                equal.shape[0], int(max_samples), replace=False)]
        return equal

    def save_results(self, filename):
        if self.samples is None:
            raise RuntimeError('Run nested sampling before saving results.')
        filename = Path(filename)
        np.savez_compressed(
            filename,
            samples=self.samples,
            weights=self.weights,
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
        samples = self.posterior_samples(max_samples=max_samples, seed=seed)
        return make_corner_plot(
            samples, self.param_names, self.log_params, **corner_kwargs)

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run nested sampling or load results first.')
        _, models = self.chi_squared_physical(
            self.best_params, return_models=True)
        sed_order = np.argsort(self.sed_wavelengths_micron)
        spatial_frequency_mlambda = (
            np.hypot(
                self.correlated_flux['u_m'], self.correlated_flux['v_m'])
            / self.correlated_flux['wavelength_m'] / 1e6)
        corr_order = np.argsort(spatial_frequency_mlambda)

        if self.fit_sed:
            figure, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].errorbar(
                self.sed_wavelengths_micron[sed_order],
                self.observed_sed_jy[sed_order],
                yerr=self.sed_error_jy[sed_order],
                fmt='o', color='black', capsize=3, label='Observed SED')
            axes[0].plot(
                self.sed_wavelengths_micron[sed_order],
                models['sed_jy'][sed_order],
                color='tab:red', linewidth=2, label='pyGrater model')
            axes[0].set(
                xscale='log', yscale='log', xlabel='Wavelength [um]',
                ylabel='Flux [Jy]')
            correlated_axis = axes[1]
        else:
            figure, correlated_axis = plt.subplots(figsize=(7, 5))
            axes = [correlated_axis]

        correlated_axis.errorbar(
            spatial_frequency_mlambda[corr_order],
            self.correlated_flux['value'][corr_order],
            yerr=self.correlated_flux['error'][corr_order],
            fmt='o', color='black', capsize=3, label='Observed correlated flux')
        correlated_axis.scatter(
            spatial_frequency_mlambda[corr_order],
            models['correlated_flux_jy'][corr_order],
            color='tab:red', marker='x', label='Analytical model')
        correlated_axis.set(
            xlabel='Spatial frequency [Mlambda]',
            ylabel='Correlated flux [Jy]')
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        figure.tight_layout()
        return figure

    def get_total_mass(self):
        if self.best_params is None:
            raise RuntimeError('Run nested sampling or load results first.')
        parameters = self._complete_parameters(self.best_params)
        return self.sed_obj.get_total_mass(**self._sed_parameters(parameters))

    def summary(self):
        if self.posterior_summary is None:
            print('No nested-sampling result yet.')
            return
        total_points = (
            self.observed_sed_jy.size + self.correlated_flux['value'].size)
        degrees_of_freedom = max(total_points - self.ndim, 1)
        print(f'log(Z) = {self.log_evidence:.8g} '
              f'+/- {self.log_evidence_error:.3g}')
        if self.fit_sed:
            print(f'Raw chi2 SED = {self.best_chi2_components["sed"]:.8g}')
        print('Raw chi2 correlated flux = '
              f'{self.best_chi2_components["correlated_flux"]:.8g}')
        print(f'Combined raw reduced chi2 = '
              f'{self.best_chi2 / degrees_of_freedom:.8g}')
        print(f'Nested fit statistic = {self.best_fit_statistic:.8g}')
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
              f'{"+1sigma":>14} {"best":>14}')
        print('-' * 92)
        for name in self.param_names:
            values = self.posterior_summary[name]
            print(f'{name:<32} {values["median"]:>14.6g} '
                  f'{values["minus_1sigma"]:>14.6g} '
                  f'{values["plus_1sigma"]:>14.6g} '
                  f'{self.best_params[name]:>14.6g}')
