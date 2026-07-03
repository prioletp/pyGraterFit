"""Joint SED and squared-visibility fitter for additive optimized disk models.

Each component is one ring/composition pair.  Ring parameters can be shared
between compositions with ``component_groups`` while every composition keeps
its own ``A_norm``.  The objective gives equal weight to the SED and V2 data:

    chi2_fit = chi2_sed / N_sed + chi2_vis2 / N_vis2

Closure phases are deliberately not evaluated by this fitter.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize

from pyGraterFit.fitters.multi_component_sed_mcmc import (
    AdditiveSEDMCMCFitter,
)
from pyGrater import Image
from pyGraterFit.utils.interferometry import (
    observables_from_image,
    uniform_disk_argument_per_mas,
    uniform_disk_visibility,
    uniform_disk_visibility_from_argument,
)


STELLAR_DIAMETER_PARAMETER = 'stellar_angular_diameter_mas'


def vis2_from_vlti_loader(source, **loader_arguments):
    """Convert ``vlti_loader.Observations`` data to the fitter's V2 format.

    ``source`` may be an existing ``Observations`` instance, a FITS path, a
    directory, or a list of FITS paths.  Passing an existing instance is useful
    because filtering and spectral binning performed with ``vlti_loader`` are
    retained.  Invalid values and non-positive uncertainties are removed once.
    """
    if hasattr(source, 'data'):
        if loader_arguments:
            raise ValueError(
                'loader_arguments cannot be used with an Observations object.')
        observations = source
    else:
        try:
            from vlti_loader import Observations
        except ImportError as exc:
            raise ImportError(
                'Install prioletp/vlti_loader or pass a V2 dictionary.') from exc
        observations = Observations(source, **loader_arguments)

    data = observations.data
    required = ('VIS2', 'VIS2_err', 'Bu', 'Bv', 'VIS2_waves')
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f'vlti_loader data are missing {missing}.')

    arrays = {
        'value': np.asarray(data['VIS2'], dtype=np.float64).ravel(),
        'error': np.asarray(data['VIS2_err'], dtype=np.float64).ravel(),
        'u_m': np.asarray(data['Bu'], dtype=np.float64).ravel(),
        'v_m': np.asarray(data['Bv'], dtype=np.float64).ravel(),
        'wavelength_m': np.asarray(
            data['VIS2_waves'], dtype=np.float64).ravel(),
    }
    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError('vlti_loader V2 arrays do not have matching lengths.')
    valid = np.ones(arrays['value'].size, dtype=bool)
    for values in arrays.values():
        valid &= np.isfinite(values)
    valid &= arrays['error'] > 0
    valid &= arrays['wavelength_m'] > 0
    arrays = {name: values[valid] for name, values in arrays.items()}
    if arrays['value'].size == 0:
        raise ValueError('No valid squared-visibility points remain.')

    if 'INS_VIS2' in data:
        instruments = np.asarray(data['INS_VIS2']).ravel()
        if instruments.size == valid.size:
            arrays['instrument'] = instruments[valid].astype(str)
    return arrays


def _validate_vis2(vis2):
    if hasattr(vis2, 'data') or isinstance(vis2, (str, bytes, list, tuple)) \
            or hasattr(vis2, '__fspath__'):
        vis2 = vis2_from_vlti_loader(vis2)
    if isinstance(vis2, dict) and 'vis2' in vis2:
        vis2 = vis2['vis2']
    if not isinstance(vis2, dict):
        raise TypeError('vis2 must be a V2 dictionary or vlti_loader source.')

    required = ('value', 'error', 'u_m', 'v_m', 'wavelength_m')
    arrays = {
        name: np.asarray(vis2[name], dtype=np.float64).ravel()
        for name in required}
    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError('All V2 arrays must have the same length.')
    valid = np.ones(arrays['value'].size, dtype=bool)
    for values in arrays.values():
        valid &= np.isfinite(values)
    valid &= arrays['error'] > 0
    valid &= arrays['wavelength_m'] > 0
    arrays = {name: values[valid] for name, values in arrays.items()}
    if arrays['value'].size == 0:
        raise ValueError('No valid squared-visibility points remain.')
    instrument = vis2.get('instrument')
    if instrument is not None:
        instrument = np.asarray(instrument).ravel()
        if instrument.size != valid.size:
            raise ValueError('V2 instrument labels must match the data length.')
        arrays['instrument'] = instrument[valid].astype(str)
    return arrays


class SEDVisibilityMCMCFitter(
        AdditiveSEDMCMCFitter):
    """Fit an SED and V2 data with any number of rings and compositions."""

    _SPATIAL_PARAMETER_NAMES = (
        'r0', 'h0', 'alphain', 'alphaout', 'beta', 'gamma',
        'itilt', 'PA', 'omega')

    def __init__(
            self, components, star, density_distribution, size_distribution,
            scattering_phase_function, sed_wavelengths, sed_fluxes,
            sed_flux_errors, vis2, params_by_component,
            shared_parameter_names=(), prior_ranges_by_component=None,
            shared_prior_ranges=None, best_fit_values=None,
            method='Nelder-Mead', use_log_params=True, N_distances=400,
            parallel_components='auto', max_component_workers=2,
            component_groups=None, group_shared_parameter_names=(),
            image_wavelengths=None, image_settings=None,
            include_unresolved_star=True, maximum_wavelength_mismatch=0.0,
            fft_padding_factor=4, sed_model_kwargs=None,
            mass_abundance_groups=None,
            stellar_visibility_model='point_source',
            stellar_angular_diameter_mas=0.0):
        if stellar_visibility_model not in ('point_source', 'uniform_disk'):
            raise ValueError(
                'stellar_visibility_model must be "point_source" or '
                '"uniform_disk".')
        if stellar_visibility_model == 'uniform_disk':
            if not include_unresolved_star:
                raise ValueError(
                    'A uniform-disk star requires include_unresolved_star=True.')
            if STELLAR_DIAMETER_PARAMETER in shared_parameter_names:
                raise ValueError(
                    f'Pass {STELLAR_DIAMETER_PARAMETER} through its dedicated '
                    'constructor argument, not shared_parameter_names.')
            params_by_component = {
                name: {**values, STELLAR_DIAMETER_PARAMETER:
                       stellar_angular_diameter_mas}
                for name, values in params_by_component.items()}
            shared_parameter_names = (
                *tuple(shared_parameter_names), STELLAR_DIAMETER_PARAMETER)
        self.stellar_visibility_model = stellar_visibility_model
        self._stellar_diameter_is_variable = (
            stellar_visibility_model == 'uniform_disk'
            and (callable(stellar_angular_diameter_mas)
                 or isinstance(stellar_angular_diameter_mas,
                               (tuple, list, np.ndarray))))
        super().__init__(
            components, star, density_distribution, size_distribution,
            scattering_phase_function, sed_wavelengths, sed_fluxes,
            sed_flux_errors, params_by_component,
            shared_parameter_names=shared_parameter_names,
            prior_ranges_by_component=prior_ranges_by_component,
            shared_prior_ranges=shared_prior_ranges,
            best_fit_values=best_fit_values, method=method,
            use_log_params=use_log_params, N_distances=N_distances,
            parallel_components=parallel_components,
            max_component_workers=max_component_workers,
            component_groups=component_groups,
            group_shared_parameter_names=group_shared_parameter_names,
            sed_model_kwargs=sed_model_kwargs,
            mass_abundance_groups=mass_abundance_groups)

        self.vis2 = _validate_vis2(vis2)
        self._inv_vis2_error = 1.0 / self.vis2['error']
        self.n_sed_points = self.obs.size
        self.n_vis2_points = self.vis2['value'].size
        self.image_settings = dict(image_settings or {})
        self.include_unresolved_star = bool(include_unresolved_star)
        self.fft_padding_factor = float(fft_padding_factor)
        if self.fft_padding_factor < 1:
            raise ValueError('fft_padding_factor must be at least 1.')

        if image_wavelengths is None:
            image_wavelengths = np.unique(self.vis2['wavelength_m']) * 1e6
        self.image_wavelengths_micron = np.asarray(
            image_wavelengths, dtype=np.float64).ravel()
        if (self.image_wavelengths_micron.size == 0
                or np.any(~np.isfinite(self.image_wavelengths_micron))
                or np.any(self.image_wavelengths_micron <= 0)):
            raise ValueError('Image wavelengths must be finite and positive.')

        self.image_objects = {
            name: Image(
                grain, star, density_distribution, size_distribution,
                scattering_phase_function, self.image_wavelengths_micron)
            for name, grain in self.components.items()}
        first_image = self.image_objects[self.component_names[0]]
        self._stellar_flux_interpolator = (
            first_image.radiative_transfer.stellar_spectrum_interpolator)
        self._assign_vis2_to_images(maximum_wavelength_mismatch)
        self._stellar_flux_at_vis2 = (
            np.asarray(self._stellar_flux_interpolator(
                self.vis2['wavelength_m'] * 1e6), dtype=np.float64)
            if self.include_unresolved_star
            else np.zeros(self.n_vis2_points, dtype=np.float64))
        self._fixed_stellar_visibility_at_vis2 = None
        self._stellar_uniform_disk_argument_per_mas = (
            uniform_disk_argument_per_mas(
                self.vis2['u_m'], self.vis2['v_m'],
                self.vis2['wavelength_m']))
        if not self._stellar_diameter_is_variable:
            fixed_diameter = (
                0.0 if stellar_visibility_model == 'point_source'
                else float(stellar_angular_diameter_mas))
            self._fixed_stellar_visibility_at_vis2 = uniform_disk_visibility(
                self.vis2['u_m'], self.vis2['v_m'],
                self.vis2['wavelength_m'], fixed_diameter)
        self.timings = {
            'sed': 0.0, 'images': 0.0, 'fourier': 0.0, 'total': 0.0}
        self.best_chi2_components = None
        self.last_chi2_components = None
        print(f'SED points: {self.n_sed_points}')
        print(f'V2 points: {self.n_vis2_points}')
        print(f'Monochromatic image planes: '
              f'{self.image_wavelengths_micron.size}')
        print('Objective: chi2_SED/N_SED + chi2_V2/N_V2')
        print(f'Stellar visibility model: {self.stellar_visibility_model}')

    def _full_component_params(self, values, name):
        parameters = super()._full_component_params(values, name)
        parameters.pop(STELLAR_DIAMETER_PARAMETER, None)
        return parameters

    def _stellar_angular_diameter(self, values):
        if self.stellar_visibility_model == 'point_source':
            return 0.0
        first = self.component_names[0]
        diameter = super()._full_component_params(values, first)[
            STELLAR_DIAMETER_PARAMETER]
        if diameter < 0:
            raise ValueError('Stellar angular diameter must be non-negative.')
        return float(diameter)

    def _assign_vis2_to_images(self, maximum_relative_mismatch):
        model_wavelengths_m = self.image_wavelengths_micron * 1e-6
        observed = self.vis2['wavelength_m']
        nearest = np.argmin(
            np.abs(observed[:, None] - model_wavelengths_m[None, :]), axis=1)
        mismatch = np.abs(observed - model_wavelengths_m[nearest]) / observed
        tolerance = float(maximum_relative_mismatch)
        numerical_tolerance = 32 * np.finfo(np.float64).eps
        if np.any(mismatch > max(tolerance, numerical_tolerance)):
            worst = int(np.argmax(mismatch))
            raise ValueError(
                'No image wavelength is sufficiently close to V2 wavelength '
                f'{observed[worst] * 1e6:.8g} micron (relative mismatch '
                f'{mismatch[worst]:.3g}).')
        self._vis2_image_index = nearest
        self._vis2_indices_by_image = [
            np.flatnonzero(nearest == image_index)
            for image_index in range(model_wavelengths_m.size)]

    def _evaluate_one_image(self, item):
        name, params, prepared_spatial_disk = item
        return self.image_objects[name].get_image(
            keep_separate_fluxes=False,
            prepared_spatial_disk=prepared_spatial_disk,
            **self.image_settings, **params)

    def component_images(self, values):
        parameters = {
            name: self._full_component_params(values, name)
            for name in self.component_names}
        prepared_by_geometry = {}
        items = []
        image_grid_key = tuple(
            self.image_settings.get(name)
            for name in ('nx', 'ny', 'pixAU', 'FOV_AU', 'nl'))
        for name in self.component_names:
            params = parameters[name]
            geometry_key = image_grid_key + tuple(
                float(params[parameter])
                for parameter in self._SPATIAL_PARAMETER_NAMES)
            prepared_spatial_disk = prepared_by_geometry.get(geometry_key)
            if prepared_spatial_disk is None:
                prepared_spatial_disk = self.image_objects[
                    name].prepare_spatial_disk(
                        **self.image_settings, **params)
                prepared_by_geometry[geometry_key] = prepared_spatial_disk
            items.append((name, params, prepared_spatial_disk))
        if self._executor is not None:
            return list(self._executor.map(self._evaluate_one_image, items))
        return [self._evaluate_one_image(item) for item in items]

    def model_images(self, values):
        outputs = self.component_images(values)
        total = np.asarray(outputs[0], dtype=np.float64).copy()
        for output in outputs[1:]:
            total += np.asarray(output, dtype=np.float64)
        return total

    def model_vis2(self, values, model_images=None):
        if model_images is None:
            model_images = self.model_images(values)
        calculated = np.empty(self.n_vis2_points, dtype=np.float64)
        stellar_visibility = self._fixed_stellar_visibility_at_vis2
        if stellar_visibility is None:
            stellar_visibility = uniform_disk_visibility_from_argument(
                self._stellar_uniform_disk_argument_per_mas
                * self._stellar_angular_diameter(values))
        pixel_scale_au = self.image_objects[self.component_names[0]].pixAU
        for image_index, indices in enumerate(self._vis2_indices_by_image):
            if indices.size == 0:
                continue
            values_at_baselines, _, _ = observables_from_image(
                model_images[image_index], pixel_scale_au, self.star.distance,
                vis2_u_m=self.vis2['u_m'][indices],
                vis2_v_m=self.vis2['v_m'][indices],
                vis2_wavelength_m=self.vis2['wavelength_m'][indices],
                unresolved_flux_jy=self._stellar_flux_at_vis2[indices],
                vis2_stellar_visibility=stellar_visibility[indices],
                padding_factor=self.fft_padding_factor)
            calculated[indices] = values_at_baselines
        return calculated

    def evaluate_physical_parameters(self, values, return_models=False):
        start = time.perf_counter()
        sed_start = start
        model_sed = self.model(values)
        sed_elapsed = time.perf_counter() - sed_start
        sed_residual = (self.obs - model_sed) * self._inv_obs_err
        sed_chi2 = float(np.dot(sed_residual, sed_residual))

        image_start = time.perf_counter()
        model_images = self.model_images(values)
        image_elapsed = time.perf_counter() - image_start
        fourier_start = time.perf_counter()
        model_vis2 = self.model_vis2(values, model_images)
        fourier_elapsed = time.perf_counter() - fourier_start
        vis2_residual = (
            (self.vis2['value'] - model_vis2) * self._inv_vis2_error)
        vis2_chi2 = float(np.dot(vis2_residual, vis2_residual))

        components = {
            'sed_chi2': sed_chi2,
            'sed_chi2_per_point': sed_chi2 / self.n_sed_points,
            'vis2_chi2': vis2_chi2,
            'vis2_chi2_per_point': vis2_chi2 / self.n_vis2_points,
        }
        objective = (
            components['sed_chi2_per_point']
            + components['vis2_chi2_per_point'])
        self.last_chi2_components = components
        self.timings['sed'] += sed_elapsed
        self.timings['images'] += image_elapsed
        self.timings['fourier'] += fourier_elapsed
        self.timings['total'] += time.perf_counter() - start
        if return_models:
            models = {'sed_jy': model_sed, 'vis2': model_vis2,
                      'images_jy_per_pixel': model_images}
            return objective, components, models
        return objective, components

    def chi_squared_physical(self, values):
        try:
            objective, components = self.evaluate_physical_parameters(values)
            if objective < self.best_chi2:
                self.best_chi2_components = components.copy()
            return float(objective)
        except Exception as exc:
            print(f'[ERROR] joint SED/V2 evaluation failed: {exc}')
            return np.inf

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        self.n_evaluations, self.best_chi2 = 0, np.inf
        x0 = (self._values_to_vector(initial_guess, optimizer_space=True)
              if initial_guess is not None
              else 0.5 * (self._bounds_lo + self._bounds_hi))
        print(f'\nStarting {self.method} joint SED/V2 optimization ...')
        if self.method == 'differential_evolution':
            result = differential_evolution(
                self.chi_squared, self._bounds, maxiter=maxiter,
                disp=verbose, seed=42)
        elif self.method == 'dual_annealing':
            result = dual_annealing(
                self.chi_squared, self._bounds, maxiter=maxiter, seed=42)
        else:
            result = minimize(
                self.chi_squared, x0, method=self.method, bounds=self._bounds,
                options={'maxiter': maxiter, 'disp': verbose})
        result.best_params = self.best_params
        result.best_chi2 = self.best_chi2
        result.chi2_components = self.chi2_breakdown(self.best_params)
        return result

    def chi2_breakdown(self, values=None):
        values = self.best_params if values is None else values
        if values is None:
            raise RuntimeError('No fitted parameters are available.')
        objective, components = self.evaluate_physical_parameters(values)
        return {'objective': objective, 'n_sed': self.n_sed_points,
                'n_vis2': self.n_vis2_points, **components}

    @staticmethod
    def _print_chi2_breakdown(breakdown):
        print(f'Joint objective = {breakdown["objective"]:.8g}')
        print(f'  SED: chi2={breakdown["sed_chi2"]:.8g}, '
              f'N={breakdown["n_sed"]}, '
              f'chi2/N={breakdown["sed_chi2_per_point"]:.8g}')
        print(f'  V2:  chi2={breakdown["vis2_chi2"]:.8g}, '
              f'N={breakdown["n_vis2"]}, '
              f'chi2/N={breakdown["vis2_chi2_per_point"]:.8g}')

    def summary(self, include_mass_abundances=True):
        if self.best_params is None:
            print('No fit result yet.')
            return
        print('\nBest joint SED/V2 fit:')
        self._print_chi2_breakdown(self.chi2_breakdown())
        for name in self.component_names:
            print(f'{name}:')
            for parameter, value in self.best_params[name].items():
                print(f'  {parameter:<18} {value:>14.6g}')
        if include_mass_abundances:
            print(self.format_component_mass_abundances(), end='')

    def mcmc_summary(self):
        if self.mcmc_param_summary is None:
            print('No MCMC result yet.')
            return
        print('\nJoint SED/V2 posterior summary:')
        print(f'{"Parameter":<28} {"median":>14} '
              f'{"-1sigma":>14} {"+1sigma":>14}')
        print('-' * 73)
        for name in self.param_names:
            value = self.mcmc_param_summary[name]
            print(f'{name:<28} {value["median"]:>14.6g} '
                  f'{value["minus_1sigma"]:>14.6g} '
                  f'{value["plus_1sigma"]:>14.6g}')
        print('\nBest posterior sample:')
        self._print_chi2_breakdown(self.chi2_breakdown(
            self.mcmc_best_params))
        print(self.format_component_mass_abundances(
            values=self.mcmc_best_params), end='')

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run a fit or restore an MCMC chain first.')
        _, breakdown, models = self.evaluate_physical_parameters(
            self.best_params, return_models=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].errorbar(
            self.wavelengths, self.obs, yerr=self.obs_err, fmt='o', ms=4,
            color='black', label='SED data')
        axes[0].plot(self.wavelengths, models['sed_jy'], color='tab:red',
                     lw=1.7, label='Best model')
        axes[0].set(xscale='log', yscale='log', xlabel='Wavelength [micron]',
                    ylabel='Dust flux density [Jy]')
        axes[0].legend()
        axes[0].grid(True, which='both', alpha=0.25)

        baseline = np.hypot(self.vis2['u_m'], self.vis2['v_m'])
        spatial_frequency = baseline / self.vis2['wavelength_m'] / 1e6
        axes[1].errorbar(
            spatial_frequency, self.vis2['value'], yerr=self.vis2['error'],
            fmt='o', ms=3, color='black', alpha=0.7, label='V2 data')
        order = np.argsort(spatial_frequency)
        axes[1].scatter(
            spatial_frequency[order], models['vis2'][order], s=11,
            color='tab:red', label='Best model')
        axes[1].set(xlabel='Spatial frequency [Mlambda]', ylabel='V2')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()
        fig.suptitle(
            f'chi2_SED/N={breakdown["sed_chi2_per_point"]:.4g}; '
            f'chi2_V2/N={breakdown["vis2_chi2_per_point"]:.4g}')
        fig.tight_layout()
        return fig
