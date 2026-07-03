"""FOV-aware fitter for two rings containing multiple grain materials."""

import numpy as np

from pyGrater.SED_fov import SEDFOV
from pyGraterFit.fitters.multi_component_sed_mcmc import (
    AdditiveSEDMCMCFitter,
)


class FOVAdditiveSEDMCMCFitter(
        AdditiveSEDMCMCFitter):
    """Fit an instrument-transmitted SED from two multi-material rings.

    Physical parameters are shared by all materials within a ring. ``A_norm``
    can be independent for every ring/material component or parameterized as a
    total ring normalization plus composition fractions. Scalars in
    ``ring_params`` remain fixed, two-value sequences are fitted, and callables
    define dependencies such as ``h0=lambda p: 0.05 * p['r0']``. Fitting
    ranges are also used as uniform priors unless separate priors are supplied.

    The inherited API provides ``fit``, ``fit_then_mcmc``, ``run_prior_mcmc``,
    ``restart_mcmc``, ``resume_backend_mcmc``, and all result plots.
    """

    def __init__(
            self, materials, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, instrument_names,
            transmission_by_instrument, fluxes, fluxes_err, ring_params,
            normalization_ranges=(1e25, 1e38),
            prior_ranges_by_component=None, shared_prior_ranges=None,
            best_fit_values=None, method='Nelder-Mead', use_log_params=True,
            N_distances=400, n_azimuth=64, parallel_components='auto',
            max_component_workers=2, spatial_parameter_names=None,
            normalization_mode='independent',
            normalization_total_ranges=None):
        materials = dict(materials)
        ring_params = dict(ring_params)
        if not materials:
            raise ValueError('At least one grain material is required.')
        if len(ring_params) != 2:
            raise ValueError('ring_params must contain exactly two rings.')

        if isinstance(normalization_ranges, dict):
            normalization_by_component = normalization_ranges
        else:
            normalization_by_component = None
            normalization_range = self._normalization_range(
                normalization_ranges, 'normalization_ranges')

        components = {}
        params_by_component = {}
        component_groups = {}
        for ring_name, physical_parameters in ring_params.items():
            if 'A_norm' in physical_parameters:
                raise ValueError(
                    'Put A_norm ranges in normalization_ranges, not ring_params.')
            for material_name, grain in materials.items():
                component_name = f'{ring_name}.{material_name}'
                if normalization_by_component is None:
                    component_normalization = normalization_range
                else:
                    try:
                        component_normalization = self._normalization_range(
                            normalization_by_component[component_name],
                            f'normalization_ranges[{component_name!r}]')
                    except KeyError as exc:
                        raise ValueError(
                            f'Missing normalization range for {component_name}.') \
                            from exc
                components[component_name] = grain
                params_by_component[component_name] = {
                    **physical_parameters,
                    'A_norm': component_normalization,
                }
                component_groups[component_name] = ring_name

        first_ring_parameters = tuple(next(iter(ring_params.values())))
        for ring_name, parameters in ring_params.items():
            if tuple(parameters) != first_ring_parameters:
                raise ValueError(
                    f'{ring_name} parameter names/order differ from the first ring.')

        sed_model_kwargs = {
            'instrument_names': np.asarray(instrument_names, dtype=str),
            'transmission_by_instrument': transmission_by_instrument,
            'n_azimuth': n_azimuth,
            'spatial_parameter_names': spatial_parameter_names,
        }
        if spatial_parameter_names is None:
            sed_model_kwargs.pop('spatial_parameter_names')

        super().__init__(
            components, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, fluxes, fluxes_err,
            params_by_component,
            shared_parameter_names=(),
            prior_ranges_by_component=prior_ranges_by_component,
            shared_prior_ranges=shared_prior_ranges,
            best_fit_values=best_fit_values, method=method,
            use_log_params=use_log_params, N_distances=N_distances,
            parallel_components=parallel_components,
            max_component_workers=max_component_workers,
            component_groups=component_groups,
            group_shared_parameter_names=first_ring_parameters,
            sed_model_class=SEDFOV, sed_model_kwargs=sed_model_kwargs,
            normalization_mode=normalization_mode,
            normalization_groups=component_groups,
            normalization_total_ranges=normalization_total_ranges)

        # Grain composition changes the radiative transfer, not the projected
        # density. Share one bounded spatial cache between materials per ring.
        self.spatial_caches_by_ring = {name: {} for name in ring_params}
        for component_name, sed_object in self.sed_objects.items():
            ring_name = component_groups[component_name]
            sed_object._spatial_cache = self.spatial_caches_by_ring[ring_name]

        self.materials = materials
        self.ring_params = ring_params
        self.instrument_names = np.asarray(instrument_names, dtype=str)
        self.transmission_by_instrument = dict(transmission_by_instrument)

    @staticmethod
    def _normalization_range(value, label):
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ValueError(f'{label} must be a two-value range.')
        lower, upper = float(value[0]), float(value[1])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f'{label} must be finite.')
        if lower <= 0 or lower >= upper:
            raise ValueError(f'Invalid {label}: {(lower, upper)}')
        return lower, upper
