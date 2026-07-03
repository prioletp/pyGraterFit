"""Additive pyGrater SED plus analytical correlated-flux fitting.

This module extends the additive SED fitter to interferometric correlated
fluxes.  Each component still owns its own pyGrater SED object, grain
composition, sublimation behaviour, and density calculation.  The only extra
step is ring-level: components that belong to the same physical ring are added
at the correlated-flux wavelengths and multiplied by one analytical visibility
for that ring.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize

import pyGraterFit.fitters.multi_component_sed_mcmc as additive_sed_mcmc
from pyGraterFit.fitters.multi_component_sed_mcmc import (
    AdditiveSEDMCMCFitter,
    FINITE_CHI2_CEILING,
    MAX_SAFE_RESIDUAL,
    _components_from_ring_materials,
)
from pyGraterFit.fitters.single_ring_sed_correlated_flux_dynesty import (
    ANALYTICAL_ONLY_PARAMETERS,
    RING_FWHM_PARAMETER,
    STELLAR_DIAMETER_PARAMETER,
    _safe_chi_squared_from_residual,
    _validate_correlated_flux,
)
from pyGraterFit.utils.analytical_visibilities import (
    analytical_disk_visibility,
)
from pyGraterFit.utils.interferometry import (
    uniform_disk_argument_per_mas,
    uniform_disk_visibility_from_argument,
)

additive_sed_mcmc.LOG_SPACE_PARAMS.add(RING_FWHM_PARAMETER)


class AdditiveSEDCorrelatedFluxFitter(AdditiveSEDMCMCFitter):
    """Fit additive pyGrater components to correlated flux, optionally with SED.

    Parameters are the same as :class:`AdditiveSEDMCMCFitter`, with these
    additions:

    ``correlated_flux``
        Dictionary, ``vlti_loader.Observations`` object, or OIFITS path accepted
        by ``correlated_flux_from_vlti_loader``.  The normalized dictionary
        contains wavelengths in metres, baseline coordinates in metres, values
        in Jy, and errors in Jy.
    ``ring_visibility_groups``
        Mapping from component name to physical ring name.  Components with the
        same ring name are summed and assigned one analytical visibility.  By
        default this reuses ``component_groups``.
    ``sed_wavelengths``/``sed_fluxes``/``sed_flux_errors``
        Optional standalone SED data.  If omitted, the likelihood uses only the
        correlated fluxes.

    Required model parameters are ``r0``, ``ring_fwhm_au``, and
    ``stellar_angular_diameter_mas``.  ``r0``, inclination, and PA may differ by
    ring.  The stellar diameter should normally be fixed or globally shared.
    """

    def __init__(
            self, components=None, star=None, density_distribution=None,
            size_distribution=None, scattering_phase_function=None,
            correlated_flux=None, params_by_component=None,
            sed_wavelengths=None, sed_fluxes=None, sed_flux_errors=None,
            shared_parameter_names=(), prior_ranges_by_component=None,
            shared_prior_ranges=None, best_fit_values=None,
            method='Nelder-Mead', use_log_params=True, N_distances=400,
            parallel_components='auto', max_component_workers=2,
            component_groups=None, group_shared_parameter_names=(),
            ring_visibility_groups=None, sed_model_class=None,
            sed_model_kwargs=None, mass_abundance_groups=None,
            share_spatial_grid=True, normalization_mode='independent',
            normalization_groups=None, normalization_total_ranges=None,
            visibility_model='gaussian_ring', sed_includes_star=False,
            normalize_each_dataset=True,
            include_likelihood_normalization=True,
            materials=None, ring_params=None,
            stellar_angular_diameter_mas=0.0,
            normalization_range=(1e25, 1e38),
            mass_abundance_by_ring=True):
        if components is None and materials is not None and ring_params is not None:
            if any(STELLAR_DIAMETER_PARAMETER in parameters
                   for parameters in ring_params.values()):
                raise ValueError(
                    f'Pass {STELLAR_DIAMETER_PARAMETER!r} as '
                    'stellar_angular_diameter_mas, not inside ring_params.')
            expanded = _components_from_ring_materials(
                materials, ring_params,
                normalization_range=normalization_range,
                normalization_total_ranges=normalization_total_ranges,
                mass_abundance_by_ring=mass_abundance_by_ring)
            components = expanded['components']
            params_by_component = expanded['params_by_component']
            for parameters in params_by_component.values():
                parameters[STELLAR_DIAMETER_PARAMETER] = (
                    stellar_angular_diameter_mas)
            if component_groups is None:
                component_groups = expanded['component_groups']
            if ring_visibility_groups is None:
                ring_visibility_groups = expanded['component_groups']
            if not group_shared_parameter_names:
                group_shared_parameter_names = expanded[
                    'group_shared_parameter_names']
            if mass_abundance_groups is None:
                mass_abundance_groups = expanded['mass_abundance_groups']
            if normalization_groups is None:
                normalization_groups = expanded['normalization_groups']
            if normalization_total_ranges is None:
                normalization_total_ranges = expanded[
                    'normalization_total_ranges']
            if normalization_mode == 'independent':
                normalization_mode = 'group_total_fraction'
            if STELLAR_DIAMETER_PARAMETER in shared_parameter_names:
                shared_parameter_names = tuple(shared_parameter_names)
            else:
                shared_parameter_names = (
                    tuple(shared_parameter_names)
                    + (STELLAR_DIAMETER_PARAMETER,))
        self.correlated_flux = _validate_correlated_flux(correlated_flux)
        self._inverse_correlated_flux_error = (
            1.0 / self.correlated_flux['error'])
        correlated_log_normalization = -np.sum(np.log(
            self.correlated_flux['error'] * np.sqrt(2.0 * np.pi)))
        correlated_wavelengths_micron = (
            self.correlated_flux['wavelength_m'] * 1.0e6)

        self.fit_sed = sed_wavelengths is not None
        self.sed_includes_star = bool(sed_includes_star)
        self.normalize_each_dataset = bool(normalize_each_dataset)
        self.visibility_model = visibility_model
        if self.visibility_model != 'gaussian_ring':
            raise ValueError("Only visibility_model='gaussian_ring' is mapped.")

        if self.fit_sed:
            self.sed_wavelengths_micron = np.asarray(
                sed_wavelengths, dtype=np.float64)
            self.observed_sed_jy = np.asarray(sed_fluxes, dtype=np.float64)
            self.sed_error_jy = np.asarray(sed_flux_errors, dtype=np.float64)
            if not (
                    self.sed_wavelengths_micron.shape
                    == self.observed_sed_jy.shape
                    == self.sed_error_jy.shape):
                raise ValueError('SED wavelength, flux, and error arrays must match.')
            if np.any(~np.isfinite(self.sed_error_jy)) or np.any(
                    self.sed_error_jy <= 0.0):
                raise ValueError('SED errors must be finite and positive.')
            self._inverse_sed_error = 1.0 / self.sed_error_jy
            sed_log_normalization = -np.sum(np.log(
                self.sed_error_jy * np.sqrt(2.0 * np.pi)))
        else:
            self.sed_wavelengths_micron = np.empty(0, dtype=np.float64)
            self.observed_sed_jy = np.empty(0, dtype=np.float64)
            self.sed_error_jy = np.empty(0, dtype=np.float64)
            self._inverse_sed_error = np.empty(0, dtype=np.float64)
            sed_log_normalization = 0.0

        self.model_wavelengths_micron = np.unique(np.concatenate((
            self.sed_wavelengths_micron, correlated_wavelengths_micron)))
        self._sed_wavelength_indices = np.searchsorted(
            self.model_wavelengths_micron, self.sed_wavelengths_micron)
        self._correlated_wavelength_indices = np.searchsorted(
            self.model_wavelengths_micron, correlated_wavelengths_micron)

        dummy_flux = np.zeros_like(self.model_wavelengths_micron)
        dummy_error = np.ones_like(self.model_wavelengths_micron)
        component_arguments = {}
        if sed_model_class is not None:
            component_arguments['sed_model_class'] = sed_model_class
        super().__init__(
            components, star, density_distribution, size_distribution,
            scattering_phase_function, self.model_wavelengths_micron,
            dummy_flux, dummy_error, params_by_component,
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
            mass_abundance_groups=mass_abundance_groups,
            share_spatial_grid=share_spatial_grid,
            normalization_mode=normalization_mode,
            normalization_groups=normalization_groups,
            normalization_total_ranges=normalization_total_ranges,
            **component_arguments)

        if ring_visibility_groups is None:
            ring_visibility_groups = self.component_groups
        if set(ring_visibility_groups) != set(self.component_names):
            raise ValueError(
                'ring_visibility_groups labels must match components.')
        self.ring_visibility_groups = dict(ring_visibility_groups)
        self.ring_visibility_names = list(
            dict.fromkeys(self.ring_visibility_groups.values()))
        self.components_by_ring_visibility = {
            ring_name: [
                name for name in self.component_names
                if self.ring_visibility_groups[name] == ring_name]
            for ring_name in self.ring_visibility_names}

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
        self._stellar_visibility_argument_per_mas = (
            uniform_disk_argument_per_mas(
                self.correlated_flux['u_m'], self.correlated_flux['v_m'],
                self.correlated_flux['wavelength_m']))

        if include_likelihood_normalization:
            if not self.fit_sed:
                self._log_likelihood_normalization = correlated_log_normalization
            elif self.normalize_each_dataset:
                self._log_likelihood_normalization = (
                    sed_log_normalization / self.observed_sed_jy.size
                    + correlated_log_normalization
                    / self.correlated_flux['value'].size)
            else:
                self._log_likelihood_normalization = (
                    sed_log_normalization + correlated_log_normalization)
        else:
            self._log_likelihood_normalization = 0.0

        self.n_observations = (
            self.observed_sed_jy.size + self.correlated_flux['value'].size)
        self.best_chi2_components = None
        self._validate_visibility_parameters()

    def _full_component_params_with_analytical(self, values, name):
        return super()._full_component_params(values, name)

    def _full_component_params(self, values, name):
        params = self._full_component_params_with_analytical(values, name)
        return {
            parameter: value for parameter, value in params.items()
            if parameter not in ANALYTICAL_ONLY_PARAMETERS}

    def _validate_visibility_parameters(self):
        missing = []
        for ring_name, component_names in self.components_by_ring_visibility.items():
            representative = component_names[0]
            for parameter in ('r0', RING_FWHM_PARAMETER):
                if not self._component_has_parameter(representative, parameter):
                    missing.append(f'{ring_name}.{parameter}')
        found_stellar_diameter = any(
            self._component_has_parameter(name, STELLAR_DIAMETER_PARAMETER)
            for name in self.component_names)
        if not found_stellar_diameter:
            missing.append(STELLAR_DIAMETER_PARAMETER)
        if missing:
            raise ValueError(
                'Missing analytical correlated-flux parameters: '
                f'{sorted(set(missing))}')

    def _component_has_parameter(self, component_name, parameter):
        if parameter in self.fixed_params_by_component[component_name]:
            return True
        if parameter in self.dependent_params_by_component[component_name]:
            return True
        return any(
            entry['parameter'] == parameter
            and component_name in entry.get('targets', ())
            for entry in self._entries)

    def _ring_visibility(self, values, ring_name):
        representative = self.components_by_ring_visibility[ring_name][0]
        parameters = self._full_component_params_with_analytical(
            values, representative)
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
            f'No parameter mapping exists for {self.visibility_model!r}.')

    def _stellar_visibility(self, values):
        for name in self.component_names:
            parameters = self._full_component_params_with_analytical(
                values, name)
            if STELLAR_DIAMETER_PARAMETER in parameters:
                diameter_mas = parameters[STELLAR_DIAMETER_PARAMETER]
                if diameter_mas < 0.0:
                    raise ValueError(
                        'Stellar angular diameter must be non-negative.')
                return uniform_disk_visibility_from_argument(
                    self._stellar_visibility_argument_per_mas * diameter_mas)
        raise ValueError(
            f'Missing {STELLAR_DIAMETER_PARAMETER!r} for stellar visibility.')

    def component_fluxes(self, values):
        outputs = self.component_seds(values, keep_separate_fluxes=False)
        return {
            name: np.real(output)
            for name, output in zip(self.component_names, outputs)}

    def model(self, values, keep_separate_fluxes=False):
        if keep_separate_fluxes:
            return super().model(values, keep_separate_fluxes=True)
        component_fluxes = self.component_fluxes(values)
        total_dust = np.zeros_like(self.model_wavelengths_micron)
        for flux in component_fluxes.values():
            total_dust += flux

        model_sed_jy = total_dust[self._sed_wavelength_indices].copy()
        if self.sed_includes_star:
            model_sed_jy += self.stellar_flux_at_sed_wavelengths_jy

        coherent_flux = (
            self.stellar_flux_at_correlated_wavelengths_jy
            * self._stellar_visibility(values))
        ring_fluxes = {}
        ring_visibilities = {}
        for ring_name, component_names in (
                self.components_by_ring_visibility.items()):
            ring_flux = np.zeros(self.correlated_flux['value'].shape)
            for component_name in component_names:
                ring_flux += component_fluxes[component_name][
                    self._correlated_wavelength_indices]
            visibility = self._ring_visibility(values, ring_name)
            ring_fluxes[ring_name] = ring_flux
            ring_visibilities[ring_name] = visibility
            coherent_flux = coherent_flux + ring_flux * visibility

        return {
            'sed_jy': model_sed_jy,
            'correlated_flux_jy': np.abs(coherent_flux),
            'component_fluxes_jy': component_fluxes,
            'ring_fluxes_at_correlated_wavelengths_jy': ring_fluxes,
            'ring_visibilities': ring_visibilities,
            'stellar_visibility': self._stellar_visibility(values),
        }

    def chi_squared_components(self, values, return_models=False):
        try:
            models = self.model(values)
            model_arrays = [models['sed_jy'], models['correlated_flux_jy']]
            model_arrays += list(models['ring_fluxes_at_correlated_wavelengths_jy'].values())
            model_arrays += list(models['ring_visibilities'].values())
            model_arrays.append(models['stellar_visibility'])
            if not all(np.all(np.isfinite(array)) for array in model_arrays):
                raise FloatingPointError('non-finite model')
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
            chi2_correlated = _safe_chi_squared_from_residual(
                correlated_residual)
            if not self.fit_sed:
                fit_statistic = chi2_correlated
            elif self.normalize_each_dataset:
                fit_statistic = (
                    chi2_sed / self.observed_sed_jy.size
                    + chi2_correlated / self.correlated_flux['value'].size)
            else:
                fit_statistic = chi2_sed + chi2_correlated
            components = {
                'sed': float(chi2_sed),
                'correlated_flux': float(chi2_correlated),
                'fit_statistic': float(fit_statistic),
            }
        except Exception as exc:
            print(f'[ERROR] additive correlated-flux evaluation failed: {exc}')
            components = {
                'sed': FINITE_CHI2_CEILING if self.fit_sed else 0.0,
                'correlated_flux': FINITE_CHI2_CEILING,
                'fit_statistic': FINITE_CHI2_CEILING,
            }
            models = None
        if return_models:
            return components, models
        return components

    def chi_squared_physical(self, values):
        components = self.chi_squared_components(values)
        chi2 = components['fit_statistic']
        if not np.isfinite(chi2):
            return FINITE_CHI2_CEILING
        return min(float(chi2), FINITE_CHI2_CEILING)

    def chi_squared(self, vector):
        vector = np.clip(vector, self._bounds_lo, self._bounds_hi)
        values = self._vector_to_values(vector, optimizer_space=True)
        components = self.chi_squared_components(values)
        chi2 = components['fit_statistic']
        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_params = values
            self.best_chi2_components = components
        if self.n_evaluations % 20 == 0:
            print(f'  eval {self.n_evaluations}: fit statistic={chi2:.4f}')
        return chi2 if np.isfinite(chi2) else 1e100

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        self.n_evaluations, self.best_chi2 = 0, np.inf
        x0 = (self._values_to_vector(initial_guess, optimizer_space=True)
              if initial_guess is not None
              else 0.5 * (self._bounds_lo + self._bounds_hi))
        print(f'\nStarting {self.method} additive SED+correlated-flux optimization ...')
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
        result.best_chi2_components = self.best_chi2_components
        result.chi2_red = self.best_chi2 / max(self.n_observations - self.ndim, 1)
        return result

    def set_best_fit_values(self, values):
        self.best_params = self._validate_best_fit(values)
        self.best_chi2_components = self.chi_squared_components(
            self.best_params)
        self.best_chi2 = self.best_chi2_components['fit_statistic']

    def summary(self, include_mass_abundances=True):
        if self.best_params is None:
            print('No fit result yet.')
            return
        components = self.chi_squared_components(self.best_params)
        dof = max(self.n_observations - self.ndim, 1)
        print(f'\nBest fit statistic = {components["fit_statistic"]:.6g} '
              f'(reduced = {components["fit_statistic"] / dof:.6g}, '
              f'dof = {dof})')
        print(f'  SED chi2 = {components["sed"]:.6g}')
        print(f'  Correlated-flux chi2 = {components["correlated_flux"]:.6g}')
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
        print('\nAdditive SED+correlated-flux posterior summary:')
        print(f'{"Parameter":<28} {"median":>14} {"-1sigma":>14} {"+1sigma":>14}')
        print('-' * 73)
        for name in self.param_names:
            value = self.mcmc_param_summary[name]
            print(f'{name:<28} {value["median"]:>14.6g} '
                  f'{value["minus_1sigma"]:>14.6g} '
                  f'{value["plus_1sigma"]:>14.6g}')
        dof = max(self.n_observations - self.ndim, 1)
        print(f'\nBest sampled fit statistic = {self.mcmc_best_chi2:.6g} '
              f'(reduced = {self.mcmc_best_chi2 / dof:.6g}, dof = {dof})')
        self.summary(include_mass_abundances=False)
        print(self.format_component_mass_abundances(), end='')

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run fit/MCMC or restore a chain first.')
        components, models = self.chi_squared_components(
            self.best_params, return_models=True)
        fig, axes = plt.subplots(
            2 if self.fit_sed else 1, 1, figsize=(9, 8 if self.fit_sed else 5),
            squeeze=False)
        axes = axes[:, 0]
        if self.fit_sed:
            order = np.argsort(self.sed_wavelengths_micron)
            axes[0].errorbar(
                self.sed_wavelengths_micron[order],
                self.observed_sed_jy[order],
                yerr=self.sed_error_jy[order], fmt='o', color='black',
                capsize=4, label='SED observations')
            axes[0].plot(
                self.sed_wavelengths_micron[order],
                models['sed_jy'][order], color='black', lw=2,
                label='Total SED model')
            axes[0].set(xscale='log', yscale='log',
                        xlabel='Wavelength [um]', ylabel='Flux [Jy]')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            corr_axis = axes[1]
        else:
            corr_axis = axes[0]

        spatial_frequency = (
            np.hypot(self.correlated_flux['u_m'], self.correlated_flux['v_m'])
            / self.correlated_flux['wavelength_m'] / 1e6)
        order = np.argsort(spatial_frequency)
        corr_axis.errorbar(
            spatial_frequency[order], self.correlated_flux['value'][order],
            yerr=self.correlated_flux['error'][order], fmt='o',
            color='tab:blue', alpha=0.6, capsize=2,
            label='Correlated flux observations')
        corr_axis.plot(
            spatial_frequency[order],
            models['correlated_flux_jy'][order], '.', color='black',
            label='Model')
        corr_axis.set(xlabel='Spatial frequency [Mlambda]',
                      ylabel='Correlated flux [Jy]')
        corr_axis.legend()
        corr_axis.grid(True, alpha=0.3)
        fig.suptitle(
            'Best additive SED+correlated-flux model '
            f'(fit statistic = {components["fit_statistic"]:.3g})')
        fig.tight_layout()
        return fig
