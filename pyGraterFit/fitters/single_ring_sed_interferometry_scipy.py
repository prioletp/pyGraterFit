"""Optimized simultaneous SED and interferometric fitter.

The fitter uses ``SED`` for unresolved dust fluxes and ``Image`` for
resolved disk images.  OIFITS squared visibilities and closure phases are
calculated with the conventions documented in
``pyGraterFit.utils.interferometry``.

Parameters represented by a scalar are fixed. A two-element range marks a
free parameter, and a callable defines a dependency, for example
``h0=lambda p: 0.05 * p['r0']``. Positive scale parameters are optimized in
log10 space by default, while reported values remain physical.
"""

import time

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize

from pyGrater import CachedSED
from pyGrater import Image
from pyGraterFit.utils.interferometry import (
    load_oifits_observations,
    observables_from_image,
    uniform_disk_argument_per_mas,
    uniform_disk_visibility_from_argument,
    wrap_phase_degrees,
)
from pyGraterFit.utils.mass import (
    format_total_mass, single_component_total_mass)
from pyGraterFit.utils.parameter_handling import (
    resolve_parameters, split_parameter_specifications)


LOG_SPACE_PARAMETERS = {'M_tot', 'A_norm', 'a_min', 'a_max', 'r0', 'h0'}
STELLAR_DIAMETER_PARAMETER = 'stellar_angular_diameter_mas'


def _as_float_array(values, name):
    array = np.asarray(values, dtype=np.float64).ravel()
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains non-finite values.')
    return array


def _validate_observations(observations):
    """Validate the dictionary returned by ``load_oifits_observations``."""
    if not isinstance(observations, dict):
        raise TypeError('observations must be a dictionary or OIFITS path.')
    cleaned = {'vis2': None, 'closure_phase': None}

    vis2 = observations.get('vis2')
    if vis2 is not None:
        required = ('value', 'error', 'u_m', 'v_m', 'wavelength_m')
        cleaned['vis2'] = {
            name: _as_float_array(vis2[name], f'vis2.{name}')
            for name in required}
        lengths = {len(values) for values in cleaned['vis2'].values()}
        if len(lengths) != 1:
            raise ValueError('All squared-visibility arrays must match.')

    closure = observations.get('closure_phase')
    if closure is not None:
        required = ('value_degrees', 'error_degrees', 'u1_m', 'v1_m',
                    'u2_m', 'v2_m', 'wavelength_m')
        cleaned['closure_phase'] = {
            name: _as_float_array(closure[name], f'closure_phase.{name}')
            for name in required}
        lengths = {len(values) for values in cleaned['closure_phase'].values()}
        if len(lengths) != 1:
            raise ValueError('All closure-phase arrays must match.')

    if cleaned['vis2'] is None and cleaned['closure_phase'] is None:
        raise ValueError('No squared visibilities or closure phases were supplied.')
    return cleaned


class SEDInterferometryFitter:
    """Fit dust SED, squared visibility, and closure phase simultaneously."""

    def __init__(
            self, grain, star, density_distribution, size_distribution,
            phase_function, sed_wavelengths, sed_fluxes, sed_flux_errors,
            observations, image_wavelengths, params, method='Nelder-Mead',
            use_log_params=True, image_settings=None, sed_weight=1.0,
            vis2_weight=1.0, closure_phase_weight=1.0,
            include_unresolved_star=True, maximum_wavelength_mismatch=0.1,
            fft_padding_factor=4, N_distances=800,
            sed_model_class=CachedSED, sed_model_kwargs=None,
            stellar_visibility_model='point_source'):
        self.grain = grain
        self.star = star
        self.method = method
        self.sed_weight = float(sed_weight)
        self.vis2_weight = float(vis2_weight)
        self.closure_phase_weight = float(closure_phase_weight)
        self.include_unresolved_star = bool(include_unresolved_star)
        if stellar_visibility_model not in ('point_source', 'uniform_disk'):
            raise ValueError(
                'stellar_visibility_model must be "point_source" or '
                '"uniform_disk".')
        if stellar_visibility_model == 'uniform_disk':
            if not self.include_unresolved_star:
                raise ValueError(
                    'A uniform-disk star requires include_unresolved_star=True.')
            if STELLAR_DIAMETER_PARAMETER not in params:
                raise ValueError(
                    f'Uniform-disk mode requires {STELLAR_DIAMETER_PARAMETER} '
                    'in params.')
        self.stellar_visibility_model = stellar_visibility_model
        self.fft_padding_factor = float(fft_padding_factor)
        self.image_settings = dict(image_settings or {})

        self.sed_wavelengths_micron = _as_float_array(
            sed_wavelengths, 'sed_wavelengths')
        self.observed_sed_jy = _as_float_array(sed_fluxes, 'sed_fluxes')
        self.sed_error_jy = _as_float_array(
            sed_flux_errors, 'sed_flux_errors')
        if not (len(self.sed_wavelengths_micron)
                == len(self.observed_sed_jy) == len(self.sed_error_jy)):
            raise ValueError('SED wavelength, flux, and error arrays must match.')
        if np.any(self.sed_error_jy <= 0):
            raise ValueError('SED uncertainties must be positive.')
        self.inverse_sed_error = 1.0 / self.sed_error_jy

        if isinstance(observations, (str, bytes)) or hasattr(
                observations, '__fspath__'):
            observations = load_oifits_observations(observations)
        self.observations = _validate_observations(observations)

        self.image_wavelengths_micron = _as_float_array(
            image_wavelengths, 'image_wavelengths')
        if np.any(self.image_wavelengths_micron <= 0):
            raise ValueError('Image wavelengths must be positive.')

        sed_model_kwargs = dict(sed_model_kwargs or {})
        sed_model_kwargs.setdefault('N_distances', N_distances)
        self.sed_model = sed_model_class(
            grain, star, density_distribution, size_distribution,
            self.sed_wavelengths_micron, **sed_model_kwargs)
        self.image_model = Image(
            grain, star, density_distribution, size_distribution,
            phase_function, self.image_wavelengths_micron)
        self._stellar_flux_interpolator = (
            self.image_model.radiative_transfer.stellar_spectrum_interpolator)
        self._stellar_flux_by_observable = {}
        for observable_name, observable in self.observations.items():
            if observable is not None:
                self._stellar_flux_by_observable[observable_name] = np.asarray(
                    self._stellar_flux_interpolator(
                        observable['wavelength_m'] * 1e6),
                    dtype=np.float64)

        (self.free_parameter_ranges, self.fixed_parameter_values,
         self.dependent_parameters) = split_parameter_specifications(params)

        self._fixed_vis2_stellar_visibility = None
        self._vis2_uniform_disk_argument_per_mas = None
        diameter_is_variable = (
            STELLAR_DIAMETER_PARAMETER in self.free_parameter_ranges
            or STELLAR_DIAMETER_PARAMETER in self.dependent_parameters)
        if self.observations['vis2'] is not None:
            vis2_data = self.observations['vis2']
            self._vis2_uniform_disk_argument_per_mas = (
                uniform_disk_argument_per_mas(
                    vis2_data['u_m'], vis2_data['v_m'],
                    vis2_data['wavelength_m']))
        if self.observations['vis2'] is not None and not diameter_is_variable:
            fixed_diameter = (
                0.0 if self.stellar_visibility_model == 'point_source'
                else float(self.fixed_parameter_values[
                    STELLAR_DIAMETER_PARAMETER]))
            self._fixed_vis2_stellar_visibility = (
                uniform_disk_visibility_from_argument(
                    self._vis2_uniform_disk_argument_per_mas
                    * fixed_diameter))

        self.parameter_names = list(self.free_parameter_ranges)
        self.n_dimensions = len(self.parameter_names)
        self.log_space_parameters = (
            set(self.parameter_names) & LOG_SPACE_PARAMETERS
            if use_log_params else set())
        self.optimization_bounds = []
        for name in self.parameter_names:
            low, high = self.free_parameter_ranges[name]
            if name in self.log_space_parameters:
                if low <= 0:
                    raise ValueError(f'{name} must be positive for log-space fitting.')
                low, high = np.log10([low, high])
            self.optimization_bounds.append((low, high))

        self._assign_observations_to_images(maximum_wavelength_mismatch)
        self.n_evaluations = 0
        self.best_chi_squared = np.inf
        self.best_parameters = None
        self.best_chi_squared_components = None
        self.timings = {'sed': 0.0, 'images': 0.0, 'visibilities': 0.0,
                        'total': 0.0}

    def _assign_observations_to_images(self, maximum_relative_mismatch):
        """Precompute which monochromatic model image serves each data point."""
        self.observation_image_index = {}
        model_wavelengths_m = self.image_wavelengths_micron * 1e-6
        for observable_name in ('vis2', 'closure_phase'):
            observable = self.observations[observable_name]
            if observable is None:
                self.observation_image_index[observable_name] = None
                continue
            wavelengths = observable['wavelength_m']
            nearest = np.argmin(
                np.abs(wavelengths[:, None] - model_wavelengths_m[None, :]),
                axis=1)
            mismatch = np.abs(wavelengths - model_wavelengths_m[nearest]) / wavelengths
            if np.any(mismatch > maximum_relative_mismatch):
                worst = int(np.argmax(mismatch))
                raise ValueError(
                    'No model image is sufficiently close to observation '
                    f'wavelength {wavelengths[worst] * 1e6:.6g} micron; '
                    f'nearest mismatch is {mismatch[worst]:.1%}.')
            self.observation_image_index[observable_name] = nearest

    def optimization_vector_to_parameters(self, vector):
        parameters = {}
        for index, name in enumerate(self.parameter_names):
            value = vector[index]
            parameters[name] = (
                10.0**value if name in self.log_space_parameters else value)
        return parameters

    def parameters_to_optimization_vector(self, parameters):
        return np.array([
            np.log10(parameters[name])
            if name in self.log_space_parameters else parameters[name]
            for name in self.parameter_names], dtype=np.float64)

    def _complete_parameters(self, free_parameters):
        return resolve_parameters(
            free_parameters, self.fixed_parameter_values,
            self.dependent_parameters)

    @staticmethod
    def _disk_parameters(parameters):
        return {name: value for name, value in parameters.items()
                if name != STELLAR_DIAMETER_PARAMETER}

    def _stellar_angular_diameter(self, parameters):
        if self.stellar_visibility_model == 'point_source':
            return 0.0
        diameter = float(parameters[STELLAR_DIAMETER_PARAMETER])
        if diameter < 0:
            raise ValueError('Stellar angular diameter must be non-negative.')
        return diameter

    def _sed_chi_squared(self, parameters):
        model_sed_jy = self.sed_model.get_SED(
            keep_separate_fluxes=False, **self._disk_parameters(parameters))
        residual = ((self.observed_sed_jy - model_sed_jy)
                    * self.inverse_sed_error)
        return float(np.dot(residual, residual)), model_sed_jy

    def _interferometric_observables(self, parameters):
        image_start = time.perf_counter()
        model_images_jy = self.image_model.get_image(
            keep_separate_fluxes=False,
            **self.image_settings, **self._disk_parameters(parameters))
        image_time = time.perf_counter() - image_start

        model_vis2 = (None if self.observations['vis2'] is None
                      else np.empty(len(self.observations['vis2']['value'])))
        model_closure = (
            None if self.observations['closure_phase'] is None else
            np.empty(len(self.observations['closure_phase']['value_degrees'])))
        visibility_start = time.perf_counter()
        vis2_stellar_visibility = self._fixed_vis2_stellar_visibility
        if (model_vis2 is not None and vis2_stellar_visibility is None):
            vis2_stellar_visibility = uniform_disk_visibility_from_argument(
                self._vis2_uniform_disk_argument_per_mas
                * self._stellar_angular_diameter(parameters))
        for image_index, image in enumerate(model_images_jy):
            vis2_mask = (
                None if model_vis2 is None else
                self.observation_image_index['vis2'] == image_index)
            closure_mask = (
                None if model_closure is None else
                self.observation_image_index['closure_phase'] == image_index)
            if ((vis2_mask is None or not np.any(vis2_mask))
                    and (closure_mask is None or not np.any(closure_mask))):
                continue

            vis2_data = self.observations['vis2']
            closure_data = self.observations['closure_phase']
            vis2_star_flux = 0.0
            closure_star_flux = 0.0
            if self.include_unresolved_star:
                if vis2_mask is not None and np.any(vis2_mask):
                    vis2_star_flux = self._stellar_flux_by_observable[
                        'vis2'][vis2_mask]
                if closure_mask is not None and np.any(closure_mask):
                    closure_star_flux = self._stellar_flux_by_observable[
                        'closure_phase'][closure_mask]

            calculated_vis2, calculated_closure, _ = observables_from_image(
                image, self.image_model.pixAU, self.star.distance,
                vis2_u_m=(None if vis2_mask is None else
                          vis2_data['u_m'][vis2_mask]),
                vis2_v_m=(None if vis2_mask is None else
                          vis2_data['v_m'][vis2_mask]),
                vis2_wavelength_m=(None if vis2_mask is None else
                                   vis2_data['wavelength_m'][vis2_mask]),
                closure_u1_m=(None if closure_mask is None else
                              closure_data['u1_m'][closure_mask]),
                closure_v1_m=(None if closure_mask is None else
                              closure_data['v1_m'][closure_mask]),
                closure_u2_m=(None if closure_mask is None else
                              closure_data['u2_m'][closure_mask]),
                closure_v2_m=(None if closure_mask is None else
                              closure_data['v2_m'][closure_mask]),
                closure_wavelength_m=(None if closure_mask is None else
                                      closure_data['wavelength_m'][closure_mask]),
                unresolved_flux_jy=vis2_star_flux,
                closure_unresolved_flux_jy=closure_star_flux,
                stellar_angular_diameter_mas=(
                    self._stellar_angular_diameter(parameters)),
                vis2_stellar_visibility=(
                    None if vis2_mask is None else
                    vis2_stellar_visibility[vis2_mask]),
                padding_factor=self.fft_padding_factor)
            if vis2_mask is not None and np.any(vis2_mask):
                model_vis2[vis2_mask] = calculated_vis2
            if closure_mask is not None and np.any(closure_mask):
                model_closure[closure_mask] = calculated_closure

        return model_vis2, model_closure, image_time, (
            time.perf_counter() - visibility_start)

    def evaluate_physical_parameters(self, free_parameters):
        """Return total chi-squared, components, and model observables."""
        parameters = self._complete_parameters(free_parameters)
        sed_chi2, model_sed = self._sed_chi_squared(parameters)
        model_vis2, model_closure, image_time, visibility_time = (
            self._interferometric_observables(parameters))

        vis2_chi2 = 0.0
        if model_vis2 is not None:
            data = self.observations['vis2']
            error = np.maximum(data['error'], np.finfo(float).tiny)
            residual = (data['value'] - model_vis2) / error
            vis2_chi2 = float(np.dot(residual, residual))

        closure_chi2 = 0.0
        if model_closure is not None:
            data = self.observations['closure_phase']
            error = np.maximum(data['error_degrees'], np.finfo(float).tiny)
            residual = wrap_phase_degrees(
                data['value_degrees'] - model_closure) / error
            closure_chi2 = float(np.dot(residual, residual))

        components = {'sed': sed_chi2, 'vis2': vis2_chi2,
                      'closure_phase': closure_chi2}
        total = (self.sed_weight * sed_chi2
                 + self.vis2_weight * vis2_chi2
                 + self.closure_phase_weight * closure_chi2)
        models = {'sed_jy': model_sed, 'vis2': model_vis2,
                  'closure_phase_degrees': model_closure}
        return float(total), components, models, image_time, visibility_time

    def chi_squared(self, optimization_vector):
        start = time.perf_counter()
        low = np.array([bound[0] for bound in self.optimization_bounds])
        high = np.array([bound[1] for bound in self.optimization_bounds])
        optimization_vector = np.clip(optimization_vector, low, high)
        free_parameters = self.optimization_vector_to_parameters(
            optimization_vector)
        sed_start = time.perf_counter()
        try:
            total, components, _, image_time, visibility_time = (
                self.evaluate_physical_parameters(free_parameters))
        except Exception as error:
            print(f'Likelihood evaluation failed: {error}')
            return 1e100
        elapsed = time.perf_counter() - start
        self.timings['images'] += image_time
        self.timings['visibilities'] += visibility_time
        self.timings['sed'] += max(0.0, elapsed - image_time - visibility_time)
        self.timings['total'] += time.perf_counter() - sed_start
        self.n_evaluations += 1
        if total < self.best_chi_squared:
            self.best_chi_squared = total
            self.best_parameters = free_parameters.copy()
            self.best_chi_squared_components = components.copy()
        return total

    @property
    def n_observations(self):
        count = len(self.observed_sed_jy)
        if self.observations['vis2'] is not None:
            count += len(self.observations['vis2']['value'])
        if self.observations['closure_phase'] is not None:
            count += len(self.observations['closure_phase']['value_degrees'])
        return count

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        """Run the selected SciPy optimization and return its result."""
        self.n_evaluations = 0
        self.best_chi_squared = np.inf
        if initial_guess is None:
            initial_vector = np.array([
                0.5 * (low + high)
                for low, high in self.optimization_bounds])
        else:
            initial_vector = self.parameters_to_optimization_vector(initial_guess)

        if self.method == 'differential_evolution':
            result = differential_evolution(
                self.chi_squared, self.optimization_bounds,
                maxiter=maxiter, disp=verbose, seed=42)
        elif self.method == 'dual_annealing':
            result = dual_annealing(
                self.chi_squared, self.optimization_bounds,
                maxiter=maxiter, seed=42)
        else:
            result = minimize(
                self.chi_squared, initial_vector, method=self.method,
                bounds=self.optimization_bounds,
                options={'maxiter': maxiter, 'disp': verbose})
        result.best_params = self.best_parameters
        result.best_chi2 = self.best_chi_squared
        result.chi2_components = self.best_chi_squared_components
        result.chi2_red = self.best_chi_squared / max(
            self.n_observations - self.n_dimensions, 1)
        return result

    def get_total_mass(self, values=None):
        """Return the surviving dust mass in Earth masses.

        The mass is computed by the same pyGrater ``SED.get_total_mass``
        routine used by SED-only fitters.  Interferometric-only parameters
        such as stellar angular diameter are removed before the SED mass
        calculation.
        """
        values = self.best_parameters if values is None else values
        return single_component_total_mass(
            self.sed_model, values,
            lambda p: self._disk_parameters(self._complete_parameters(p)),
            missing_message='Run fit() first or pass parameter values.')

    def format_total_mass(self, values=None, label='Total dust mass'):
        """Return a one-line mass summary for logs and output files."""
        return format_total_mass(self.get_total_mass(values), label=label)

    def summary(self):
        """Print the best parameters and chi-squared contributions."""
        if self.best_parameters is None:
            print('No fit result is available.')
            return
        degrees_of_freedom = max(
            self.n_observations - self.n_dimensions, 1)
        print(f'Best chi2: {self.best_chi_squared:.6g}')
        print(f'Reduced chi2: {self.best_chi_squared / degrees_of_freedom:.6g}')
        for name, value in self.best_chi_squared_components.items():
            print(f'  {name}: {value:.6g}')
        for name, value in self.best_parameters.items():
            print(f'  {name}: {value:.8g}')
        print(self.format_total_mass())
