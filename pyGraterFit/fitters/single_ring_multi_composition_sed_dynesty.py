"""Convenience nested fitter for one ring containing multiple materials."""

import numpy as np

from pyGraterFit.fitters.multi_component_sed_dynesty import (
    AdditiveSEDNestedFitter)


class SingleRingMultiCompositionNestedFitter:
    """Fit one shared ring with multiple grain compositions.

    ``ring_params`` uses the standard pyGrater convention: scalars are fixed,
    two-value ranges define priors, and callables define dependencies. Adding
    or removing a material only requires changing the ``materials`` mapping.
    By default, the fit uses one total ring ``A_norm`` plus ``N-1``
    composition-fraction coordinates.
    """

    def __init__(
            self, materials, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, fluxes, fluxes_err,
            ring_params, normalization_ranges=(1e25, 1e50),
            prior_ranges_by_material=None, shared_prior_ranges=None,
            use_log_params=True, N_distances=400,
            parallel_components='auto', max_component_workers=2,
            sed_model_class=None, sed_model_kwargs=None,
            include_likelihood_normalization=True,
            normalization_mode='group_total_fraction',
            normalization_total_range=None):
        self.materials = dict(materials)
        if not self.materials:
            raise ValueError('At least one grain material is required.')
        self.ring_params = dict(ring_params)
        if 'A_norm' in self.ring_params:
            raise ValueError(
                'Put A_norm in normalization_ranges, not ring_params.')

        if isinstance(normalization_ranges, dict):
            missing = set(self.materials).difference(normalization_ranges)
            if missing:
                raise ValueError(
                    f'Missing normalization ranges for {sorted(missing)}.')
            normalization_by_material = {
                name: self._clean_range(normalization_ranges[name], name)
                for name in self.materials}
        else:
            common_range = self._clean_range(
                normalization_ranges, 'normalization_ranges')
            normalization_by_material = {
                name: common_range for name in self.materials}

        params_by_material = {
            name: {
                **self.ring_params,
                'A_norm': normalization_by_material[name],
            }
            for name in self.materials}
        mass_groups = {name: 'ring' for name in self.materials}

        arguments = {}
        if sed_model_class is not None:
            arguments['sed_model_class'] = sed_model_class
        self._nested_fitter = AdditiveSEDNestedFitter(
            self.materials, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, fluxes, fluxes_err,
            params_by_material,
            shared_parameter_names=tuple(self.ring_params),
            prior_ranges_by_component=prior_ranges_by_material,
            shared_prior_ranges=shared_prior_ranges,
            use_log_params=use_log_params, N_distances=N_distances,
            parallel_components=parallel_components,
            max_component_workers=max_component_workers,
            sed_model_kwargs=sed_model_kwargs,
            mass_abundance_groups=mass_groups,
            include_likelihood_normalization=include_likelihood_normalization,
            normalization_mode=normalization_mode,
            normalization_groups=mass_groups,
            normalization_total_ranges=(
                None if normalization_total_range is None
                else {'ring': normalization_total_range}),
            **arguments)

    @staticmethod
    def _clean_range(value, label):
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ValueError(f'{label} must be a two-value range.')
        low, high = float(value[0]), float(value[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f'Invalid range for {label}: {(low, high)}')
        return low, high

    def __getattr__(self, name):
        """Expose the general nested fitter API on this simpler wrapper."""
        return getattr(self._nested_fitter, name)

    def load_results(self, filename):
        self._nested_fitter.load_results(filename)
        return self
