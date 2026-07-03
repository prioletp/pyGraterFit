"""Optimized fitter for sums of arbitrary SED components."""

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize

from pyGrater import CachedSED, SharedSEDCache
from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.utils.mcmc_backend import write_parameter_names
from pyGraterFit.utils.parameter_handling import resolve_parameters


LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}
# Finite numerical rejection ceiling used for models that overflow or produce
# non-finite residuals.  This is intentionally much smaller than the largest
# representable float: dynesty's evidence bookkeeping can become pathological
# if rejected points are assigned log-likelihoods such as -1e99.
FINITE_CHI2_CEILING = 2.0e6
MAX_SAFE_RESIDUAL = np.sqrt(FINITE_CHI2_CEILING)


def _components_from_ring_materials(
        materials, ring_params, normalization_range=(1e25, 1e38),
        normalization_total_ranges=None, mass_abundance_by_ring=True):
    """Expand a friendly ``rings × materials`` specification.

    This keeps the public constructor simple while preserving the lower-level
    additive-component representation internally.  Each ring/material pair
    remains a distinct SED component, so each material keeps its own grain
    temperatures, sublimation behaviour, and density weighting.
    """
    materials = dict(materials)
    ring_params = dict(ring_params)
    if not materials:
        raise ValueError('At least one material is required.')
    if not ring_params:
        raise ValueError('At least one ring is required.')

    first_parameter_names = tuple(next(iter(ring_params.values())))
    for ring_name, parameters in ring_params.items():
        if tuple(parameters) != first_parameter_names:
            raise ValueError(
                f'{ring_name} parameter names/order differ from the '
                'first ring.')
        if 'A_norm' in parameters:
            raise ValueError(
                'Put A_norm in normalization_range, not ring_params.')

    components = {}
    params_by_component = {}
    component_groups = {}
    for ring_name, physical_parameters in ring_params.items():
        for material_name, grain in materials.items():
            component_name = f'{ring_name}.{material_name}'
            components[component_name] = grain
            params_by_component[component_name] = {
                **physical_parameters,
                'A_norm': normalization_range,
            }
            component_groups[component_name] = ring_name

    if normalization_total_ranges is None:
        normalization_total_ranges = {
            ring_name: normalization_range for ring_name in ring_params}
    mass_abundance_groups = (
        component_groups if mass_abundance_by_ring else None)

    return {
        'components': components,
        'params_by_component': params_by_component,
        'component_groups': component_groups,
        'group_shared_parameter_names': first_parameter_names,
        'mass_abundance_groups': mass_abundance_groups,
        'normalization_groups': component_groups,
        'normalization_total_ranges': normalization_total_ranges,
    }


def _numba_threading_layer():
    """Return the active Numba threading layer, if Numba is available."""
    try:
        import numba

        # This initializes the backend before worker threads can race to do it.
        numba.get_num_threads()
        return numba.threading_layer()
    except (ImportError, RuntimeError, ValueError):
        return None


class AdditiveSEDMCMCFitter:
    """Fit the sum of N rings and/or N grain compositions.

    ``components`` and ``params_by_component`` are ordered dictionaries with
    matching labels. Parameters in ``shared_parameter_names`` are copied to
    every component. Parameters in ``group_shared_parameter_names`` are copied
    only between components with the same ``component_groups`` label. All
    remaining parameters are independent and component-qualified.

    Scalars are fixed, two-value ranges are fitted, and callables are
    dependent parameters, e.g. ``h0=lambda p: 0.05 * p['r0']``.
    """

    _FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

    def __init__(self, components=None, star=None, density_distribution=None,
                 size_distribution=None, scattering_phase_function=None,
                 wavelengths=None, fluxes=None, fluxes_err=None,
                 params_by_component=None,
                 shared_parameter_names=(), prior_ranges_by_component=None,
                 shared_prior_ranges=None, best_fit_values=None,
                 method='Nelder-Mead', use_log_params=True,
                 N_distances=400, parallel_components='auto',
                 max_component_workers=2, component_groups=None,
                 group_shared_parameter_names=(),
                 sed_model_class=CachedSED, sed_model_kwargs=None,
                 mass_abundance_groups=None, share_spatial_grid=True,
                 normalization_mode='independent',
                 normalization_groups=None,
                 normalization_total_ranges=None,
                 materials=None, ring_params=None,
                 normalization_range=(1e25, 1e38),
                 mass_abundance_by_ring=True):
        if components is None:
            if materials is None or ring_params is None:
                raise ValueError(
                    'Provide either components/params_by_component or '
                    'materials/ring_params.')
            expanded = _components_from_ring_materials(
                materials, ring_params,
                normalization_range=normalization_range,
                normalization_total_ranges=normalization_total_ranges,
                mass_abundance_by_ring=mass_abundance_by_ring)
            components = expanded['components']
            params_by_component = expanded['params_by_component']
            if component_groups is None:
                component_groups = expanded['component_groups']
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
        if params_by_component is None:
            raise ValueError(
                'params_by_component is required when components are given.')
        self.components = dict(components)
        if not self.components:
            raise ValueError('At least one SED component is required.')
        self.component_names = list(self.components)
        if set(params_by_component) != set(self.component_names):
            raise ValueError('params_by_component labels must match components.')
        self.star = star
        self.density_distribution = density_distribution
        self.size_distribution = size_distribution
        self.scattering_phase_function = scattering_phase_function
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.obs = np.asarray(fluxes, dtype=np.float64)
        self.obs_err = np.asarray(fluxes_err, dtype=np.float64)
        if not (self.wavelengths.shape == self.obs.shape == self.obs_err.shape):
            raise ValueError('wavelengths, fluxes, and fluxes_err must match.')
        if np.any(~np.isfinite(self.obs_err)) or np.any(self.obs_err <= 0):
            raise ValueError('All observational errors must be finite and positive.')
        self._inv_obs_err = 1.0 / self.obs_err
        self.method = method
        self.use_log_params = bool(use_log_params)

        sed_model_kwargs = dict(sed_model_kwargs or {})
        sed_model_kwargs.setdefault('N_distances', N_distances)
        self.shared_sed_cache = None
        if isinstance(sed_model_class, type) and issubclass(
                sed_model_class, CachedSED):
            self.shared_sed_cache = SharedSEDCache(
                max_entries=max(4, 2 * len(self.component_names)))
            sed_model_kwargs.setdefault(
                'shared_cache', self.shared_sed_cache)
        self.sed_objects = {
            name: sed_model_class(
                grain, star, density_distribution, size_distribution,
                self.wavelengths, **sed_model_kwargs)
            for name, grain in self.components.items()}

        self.params_by_component = {
            name: dict(params_by_component[name]) for name in self.component_names}
        self.shared_parameter_names = tuple(shared_parameter_names)
        self.group_shared_parameter_names = tuple(group_shared_parameter_names)
        if set(self.shared_parameter_names) & set(
                self.group_shared_parameter_names):
            raise ValueError(
                'A parameter cannot be both globally and group shared.')
        component_groups_were_supplied = component_groups is not None
        if component_groups is None:
            component_groups = {name: name for name in self.component_names}
        if set(component_groups) != set(self.component_names):
            raise ValueError('component_groups labels must match components.')
        self.component_groups = dict(component_groups)
        if mass_abundance_groups is not None:
            if set(mass_abundance_groups) != set(self.component_names):
                raise ValueError(
                    'mass_abundance_groups labels must match components.')
            self.mass_abundance_groups = dict(mass_abundance_groups)
        else:
            self.mass_abundance_groups = (
                self.component_groups
                if component_groups_were_supplied
                else ({name: 'ring' for name in self.component_names}
                      if self.shared_parameter_names else self.component_groups))
        self.group_names = list(dict.fromkeys(self.component_groups.values()))
        self.components_by_group = {
            group: [name for name in self.component_names
                    if self.component_groups[name] == group]
            for group in self.group_names}
        self.normalization_mode = normalization_mode
        if self.normalization_mode not in {
                'independent', 'group_total_fraction'}:
            raise ValueError(
                "normalization_mode must be 'independent' or "
                "'group_total_fraction'.")
        if normalization_groups is None:
            normalization_groups = (
                self.component_groups
                if self.normalization_mode == 'group_total_fraction'
                else {name: name for name in self.component_names})
        if set(normalization_groups) != set(self.component_names):
            raise ValueError('normalization_groups labels must match components.')
        self.normalization_groups = dict(normalization_groups)
        self.normalization_group_names = list(
            dict.fromkeys(self.normalization_groups.values()))
        self.components_by_normalization_group = {
            group: [name for name in self.component_names
                    if self.normalization_groups[name] == group]
            for group in self.normalization_group_names}
        self.normalization_total_ranges = (
            {} if normalization_total_ranges is None
            else dict(normalization_total_ranges))
        if self.shared_sed_cache is not None:
            self.shared_sed_cache.max_entries = max(
                2, 2 * len(self.group_names))
        if self.shared_sed_cache is not None and share_spatial_grid:
            for group, names in self.components_by_group.items():
                members = [self.sed_objects[name] for name in names]
                for name in names:
                    self.sed_objects[name].cache_namespace = group
                    self.sed_objects[name].spatial_group_members = members
                    self.sed_objects[name].spatial_member_key = name
        self._validate_shared_specs()
        self.prior_ranges_by_component = (
            {} if prior_ranges_by_component is None
            else {k: dict(v) for k, v in prior_ranges_by_component.items()})
        self.shared_prior_ranges = (
            {} if shared_prior_ranges is None else dict(shared_prior_ranges))

        self._entries = []
        self._normalization_group_specs = {}
        self.fixed_params_by_component = {name: {} for name in self.component_names}
        self.dependent_params_by_component = {
            name: {} for name in self.component_names}
        self._build_parameter_entries()
        self.param_names = [entry['label'] for entry in self._entries]
        self.ndim = len(self.param_names)
        self.log_params = {
            entry['label'] for entry in self._entries if entry['log']}
        self.prior_ranges = {
            entry['label']: entry['prior'] for entry in self._entries}
        self._bounds = [entry['fit_bounds'] for entry in self._entries]
        self._bounds_lo = np.asarray([bound[0] for bound in self._bounds])
        self._bounds_hi = np.asarray([bound[1] for bound in self._bounds])

        if parallel_components == 'auto':
            same_grain = len({id(grain) for grain in self.components.values()}) == 1
            # Large same-grain components invoke the same bandwidth-heavy
            # parallel Numba kernel; running those kernels together is slower.
            parallel_components = (
                len(self.component_names) > 1
                and not (same_grain and len(self.wavelengths) >= 80))
        requested_parallel_components = bool(parallel_components)
        self.numba_threading_layer = _numba_threading_layer()
        if (requested_parallel_components
                and self.numba_threading_layer == 'workqueue'):
            print(
                'Parallel component evaluations disabled: Numba selected '
                "the non-thread-safe 'workqueue' backend. Numba kernels "
                'remain internally parallel.')
            parallel_components = False
        self.parallel_components = bool(parallel_components)
        self.max_component_workers = max(
            1, min(int(max_component_workers), len(self.component_names)))
        self._executor = (
            ThreadPoolExecutor(max_workers=self.max_component_workers)
            if self.parallel_components else None)

        self.input_best_fit_values = (
            None if best_fit_values is None
            else self._validate_best_fit(best_fit_values))
        self.n_evaluations = 0
        self.n_mcmc_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None
        self.mcmc_sampler = None
        self.mcmc_chain = None
        self.mcmc_log_prob = None
        self.flat_samples = None
        self.mcmc_best_params = None
        self.mcmc_best_log_prob = None
        self.mcmc_best_chi2 = None
        self.mcmc_param_summary = None
        self.component_mass_abundances = None
        self.initial_walker_summary = None
        self.initial_walker_positions = None
        self.loaded_chain = None
        self.loaded_log_prob = None
        self.loaded_last_positions = None

        print(f'Components: {self.component_names}')
        print(f'Shared parameters: {self.shared_parameter_names}')
        if self.group_shared_parameter_names:
            print(f'Component groups: {self.components_by_group}')
            print(f'Group-shared parameters: '
                  f'{self.group_shared_parameter_names}')
        print(f'Free parameters ({self.ndim}): {self.param_names}')
        dependent_labels = [
            f'{name}.{parameter}'
            for name in self.component_names
            for parameter in self.dependent_params_by_component[name]]
        print(f'Dependent parameters: {dependent_labels}')
        print(f'Log-space parameters: {self.log_params}')
        print(f'Normalization mode: {self.normalization_mode}')
        print(f'Parallel component evaluations: {self.parallel_components} '
              f'({self.max_component_workers} workers)')
        if self.numba_threading_layer is not None:
            print(f'Numba threading layer: {self.numba_threading_layer}')

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # ThreadPoolExecutor owns thread locks that cannot be pickled by
        # dynesty checkpoints. It is a runtime acceleration only; restored
        # samplers can recreate it or safely evaluate components serially.
        state['_executor'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._executor = None
        if getattr(self, 'parallel_components', False):
            self._executor = ThreadPoolExecutor(
                max_workers=getattr(self, 'max_component_workers', 1))

    def __del__(self):
        executor = getattr(self, '_executor', None)
        if executor is not None:
            executor.shutdown(wait=False)

    @staticmethod
    def _is_range(value):
        return isinstance(value, (tuple, list, np.ndarray))

    def _validate_shared_specs(self):
        first = self.params_by_component[self.component_names[0]]
        for parameter in self.shared_parameter_names:
            if parameter == 'A_norm':
                raise ValueError(
                    'A_norm cannot be shared; each additive component needs '
                    'its own normalization.')
            if parameter not in first:
                raise ValueError(f'Missing shared parameter {parameter}.')
            reference = first[parameter]
            for name in self.component_names[1:]:
                if parameter not in self.params_by_component[name]:
                    raise ValueError(f'{name} is missing shared {parameter}.')
                candidate = self.params_by_component[name][parameter]
                if callable(reference) or callable(candidate):
                    if not (callable(reference) and candidate is reference):
                        raise ValueError(
                            f'Shared dependent parameter {parameter} must use '
                            'the same named function for every component.')
                    continue
                if self._is_range(reference) != self._is_range(candidate):
                    raise ValueError(f'Shared {parameter} differs in {name}.')
                if not np.array_equal(np.asarray(reference), np.asarray(candidate)):
                    raise ValueError(
                        f'Shared parameter specification {parameter} must be '
                        'identical for every component.')

        for parameter in self.group_shared_parameter_names:
            if parameter == 'A_norm':
                raise ValueError(
                    'A_norm cannot be group shared; each additive component '
                    'needs its own normalization.')
            for group, names in self.components_by_group.items():
                first = self.params_by_component[names[0]]
                if parameter not in first:
                    raise ValueError(
                        f'Missing group-shared parameter {group}.{parameter}.')
                reference = first[parameter]
                for name in names[1:]:
                    if parameter not in self.params_by_component[name]:
                        raise ValueError(f'{name} is missing shared {parameter}.')
                    candidate = self.params_by_component[name][parameter]
                    if callable(reference) or callable(candidate):
                        if not (callable(reference) and candidate is reference):
                            raise ValueError(
                                f'Group-shared dependent parameter {parameter} '
                                'must use the same named function for every '
                                'component in the group.')
                        continue
                    if self._is_range(reference) != self._is_range(candidate):
                        raise ValueError(
                            f'Group-shared {parameter} differs in {name}.')
                    if not np.array_equal(
                            np.asarray(reference), np.asarray(candidate)):
                        raise ValueError(
                            f'Group {group} parameter specification '
                            f'{parameter} must be identical.')

    @staticmethod
    def _clean_range(value, label):
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ValueError(f'{label} must be a two-value range.')
        lo, hi = float(value[0]), float(value[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f'Invalid range for {label}: {(lo, hi)}')
        return lo, hi

    def _prior_for(self, component, parameter, fit_range, shared):
        if shared and parameter in self.shared_prior_ranges:
            value = self.shared_prior_ranges[parameter]
        elif component in self.prior_ranges_by_component and parameter in (
                self.prior_ranges_by_component[component]):
            value = self.prior_ranges_by_component[component][parameter]
        else:
            value = fit_range
        return self._clean_range(value, f'prior for {component}.{parameter}')

    def _add_entry(self, label, parameter, component, fit_range, shared,
                   targets=None, role=None, group=None):
        lo, hi = self._clean_range(fit_range, label)
        is_log = self.use_log_params and parameter in LOG_SPACE_PARAMS
        if is_log and lo <= 0:
            raise ValueError(f'Log-space parameter {label} must be positive.')
        prior = self._prior_for(component, parameter, (lo, hi), shared)
        if is_log and prior[0] <= 0:
            raise ValueError(f'Log-space prior {label} must be positive.')
        fit_bounds = (
            (np.log10(lo), np.log10(hi)) if is_log else (lo, hi))
        self._entries.append({
            'label': label, 'parameter': parameter, 'component': component,
            'shared': shared, 'log': is_log, 'fit_bounds': fit_bounds,
            'prior': prior,
            'targets': tuple(targets if targets is not None else [component]),
            'role': role,
            'group': group})

    def _uses_group_total_normalization(self, component_name):
        if self.normalization_mode != 'group_total_fraction':
            return False
        group = self.normalization_groups[component_name]
        return len(self.components_by_normalization_group[group]) > 1

    def _add_group_total_normalization_entries(self):
        if self.normalization_mode != 'group_total_fraction':
            return
        for group, names in self.components_by_normalization_group.items():
            if len(names) <= 1:
                continue
            ranges = []
            for name in names:
                if 'A_norm' not in self.params_by_component[name]:
                    raise ValueError(
                        f'{name} is missing A_norm for grouped normalization.')
                value = self.params_by_component[name]['A_norm']
                if not self._is_range(value):
                    raise ValueError(
                        f'{name}.A_norm must be a range for grouped '
                        'normalization.')
                ranges.append(self._clean_range(value, f'{name}.A_norm'))
            if group in self.normalization_total_ranges:
                total_range = self._clean_range(
                    self.normalization_total_ranges[group],
                    f'{group}.A_norm_total')
            else:
                reference = ranges[0]
                if any(rng != reference for rng in ranges[1:]):
                    raise ValueError(
                        f'Normalization group {group} has different component '
                        'A_norm ranges. Provide normalization_total_ranges.')
                total_range = reference
            self._normalization_group_specs[group] = {
                'components': tuple(names),
                'total_range': total_range,
            }
            self._add_entry(
                f'{group}.A_norm_total', 'A_norm', names[0], total_range,
                False, targets=(), role='normalization_total', group=group)
            for name in names[:-1]:
                self._add_entry(
                    f'{name}.fraction_stick', 'fraction_stick', name,
                    (0.0, 1.0), False, targets=(),
                    role='normalization_fraction_stick', group=group)

    @staticmethod
    def _fractions_from_sticks(sticks):
        sticks = np.clip(np.asarray(sticks, dtype=np.float64), 0.0, 1.0)
        n_components = len(sticks) + 1
        fractions = []
        remaining = 1.0
        for index, unit_value in enumerate(sticks):
            beta_power = n_components - index - 1
            break_fraction = 1.0 - (1.0 - unit_value) ** (
                1.0 / beta_power)
            fraction = remaining * break_fraction
            fractions.append(fraction)
            remaining -= fraction
        fractions.append(remaining)
        return np.asarray(fractions, dtype=np.float64)

    @staticmethod
    def _sticks_from_fractions(fractions):
        fractions = np.asarray(fractions, dtype=np.float64)
        if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError('Grouped A_norm fractions must be positive and sum to 1.')
        sticks = []
        remaining = 1.0
        n_components = len(fractions)
        for index, fraction in enumerate(fractions[:-1]):
            if remaining <= 0:
                sticks.append(0.0)
                continue
            break_fraction = np.clip(fraction / remaining, 0.0, 1.0)
            beta_power = n_components - index - 1
            sticks.append(1.0 - (1.0 - break_fraction) ** beta_power)
            remaining -= fraction
        return np.asarray(sticks, dtype=np.float64)

    def _apply_group_total_normalizations(self, values, vector_values):
        if self.normalization_mode != 'group_total_fraction':
            return
        by_group = {}
        for entry, value in zip(self._entries, vector_values):
            role = entry.get('role')
            if role is None:
                continue
            by_group.setdefault(entry['group'], {}).setdefault(role, []).append(
                float(value))
        for group, spec in self._normalization_group_specs.items():
            total_values = by_group.get(group, {}).get(
                'normalization_total', [])
            if len(total_values) != 1:
                raise ValueError(f'Missing {group}.A_norm_total.')
            sticks = by_group.get(group, {}).get(
                'normalization_fraction_stick', [])
            names = spec['components']
            if len(sticks) != len(names) - 1:
                raise ValueError(f'Missing composition fractions for {group}.')
            fractions = self._fractions_from_sticks(sticks)
            for name, fraction in zip(names, fractions):
                values[name]['A_norm'] = total_values[0] * fraction

    def _build_parameter_entries(self):
        self._add_group_total_normalization_entries()
        first_name = self.component_names[0]
        first = self.params_by_component[first_name]
        for parameter in self.shared_parameter_names:
            value = first[parameter]
            if callable(value):
                for name in self.component_names:
                    self.dependent_params_by_component[name][parameter] = value
            elif self._is_range(value):
                self._add_entry(
                    f'shared.{parameter}', parameter, first_name, value, True,
                    self.component_names)
            else:
                fixed = (int(value) if isinstance(value, (int, np.integer))
                         else float(value))
                for name in self.component_names:
                    self.fixed_params_by_component[name][parameter] = fixed

        shared = set(self.shared_parameter_names)
        group_shared = set(self.group_shared_parameter_names)
        for group, names in self.components_by_group.items():
            first_name = names[0]
            first = self.params_by_component[first_name]
            for parameter in self.group_shared_parameter_names:
                value = first[parameter]
                if callable(value):
                    for name in names:
                        self.dependent_params_by_component[name][parameter] = value
                elif self._is_range(value):
                    self._add_entry(
                        f'{group}.{parameter}', parameter, first_name, value,
                        False, names)
                else:
                    fixed = (int(value) if isinstance(value, (int, np.integer))
                             else float(value))
                    for name in names:
                        self.fixed_params_by_component[name][parameter] = fixed

        for name in self.component_names:
            for parameter, value in self.params_by_component[name].items():
                if (parameter == 'A_norm'
                        and self._uses_group_total_normalization(name)):
                    continue
                if parameter in shared or parameter in group_shared:
                    continue
                if callable(value):
                    self.dependent_params_by_component[name][parameter] = value
                elif self._is_range(value):
                    self._add_entry(
                        f'{name}.{parameter}', parameter, name, value, False)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    self.fixed_params_by_component[name][parameter] = (
                        int(value) if isinstance(value, (int, np.integer))
                        else float(value))
                else:
                    raise ValueError(
                        f'Invalid parameter {name}.{parameter}: {type(value)}')

    def _empty_values(self):
        return {name: {} for name in self.component_names}

    def _vector_to_values(self, vector, optimizer_space=False):
        values = self._empty_values()
        physical_vector = []
        for i, entry in enumerate(self._entries):
            value = float(vector[i])
            if optimizer_space and entry['log']:
                value = 10**value
            physical_vector.append(value)
            if entry.get('role') is not None:
                continue
            for name in entry['targets']:
                values[name][entry['parameter']] = value
        self._apply_group_total_normalizations(values, physical_vector)
        return values

    def _values_to_vector(self, values, optimizer_space=False):
        values = self._validate_best_fit(values)
        vector = []
        for entry in self._entries:
            role = entry.get('role')
            if role == 'normalization_total':
                component_names = self._normalization_group_specs[
                    entry['group']]['components']
                value = sum(values[name]['A_norm'] for name in component_names)
                vector.append(
                    np.log10(value) if optimizer_space and entry['log']
                    else value)
                continue
            if role == 'normalization_fraction_stick':
                component_names = self._normalization_group_specs[
                    entry['group']]['components']
                total = sum(values[name]['A_norm'] for name in component_names)
                fractions = [values[name]['A_norm'] / total
                             for name in component_names]
                sticks = self._sticks_from_fractions(fractions)
                stick_index = component_names.index(entry['component'])
                vector.append(float(sticks[stick_index]))
                continue
            name = self.component_names[0] if entry['shared'] else entry['component']
            value = values[name][entry['parameter']]
            vector.append(np.log10(value) if optimizer_space and entry['log'] else value)
        return np.asarray(vector, dtype=np.float64)

    def _validate_best_fit(self, values):
        if set(values) != set(self.component_names):
            raise ValueError(
                f'best_fit_values needs component keys {self.component_names}.')
        clean = {name: {} for name in self.component_names}
        for entry in self._entries:
            if entry.get('role') is not None:
                continue
            targets = entry['targets']
            found = []
            for name in targets:
                if entry['parameter'] not in values[name]:
                    raise ValueError(f'Missing best fit {name}.{entry["parameter"]}.')
                found.append(float(values[name][entry['parameter']]))
            if len(targets) > 1 and not np.all(np.asarray(found) == found[0]):
                raise ValueError(
                    f'Shared best-fit parameter {entry["parameter"]} differs '
                    'between components.')
            for name, value in zip(targets, found):
                clean[name][entry['parameter']] = value
        for spec in self._normalization_group_specs.values():
            for name in spec['components']:
                if 'A_norm' not in values[name]:
                    raise ValueError(f'Missing best fit {name}.A_norm.')
                clean[name]['A_norm'] = float(values[name]['A_norm'])
        return clean

    def _full_component_params(self, values, name):
        return resolve_parameters(
            values[name], self.fixed_params_by_component[name],
            self.dependent_params_by_component[name], context=name)

    def _evaluate_one(self, item, keep_separate_fluxes):
        name, params = item
        return self.sed_objects[name].get_SED(
            keep_separate_fluxes=keep_separate_fluxes, **params)

    def component_seds(self, values, keep_separate_fluxes=False):
        params = [(name, self._full_component_params(values, name))
                  for name in self.component_names]
        if self._executor is not None:
            return list(self._executor.map(
                lambda item: self._evaluate_one(item, keep_separate_fluxes),
                params))
        return [self._evaluate_one(item, keep_separate_fluxes) for item in params]

    def model(self, values, keep_separate_fluxes=False):
        outputs = self.component_seds(values, keep_separate_fluxes)
        if keep_separate_fluxes:
            dust_thermal_sed = np.real(outputs[0][0]).copy()
            scattered_starlight_sed = np.real(outputs[0][1]).copy()
            for (component_dust_thermal_sed,
                 component_scattered_starlight_sed) in outputs[1:]:
                dust_thermal_sed += np.real(component_dust_thermal_sed)
                scattered_starlight_sed += np.real(
                    component_scattered_starlight_sed)
            return dust_thermal_sed, scattered_starlight_sed
        total = np.real(outputs[0]).copy()
        for output in outputs[1:]:
            total += np.real(output)
        return total

    def chi_squared_physical(self, values):
        try:
            model = self.model(values)
            if not np.all(np.isfinite(model)):
                return FINITE_CHI2_CEILING
            residual = (self.obs - model) * self._inv_obs_err
            if not np.all(np.isfinite(residual)):
                return FINITE_CHI2_CEILING
            if np.max(np.abs(residual)) > MAX_SAFE_RESIDUAL:
                return FINITE_CHI2_CEILING
            chi2 = float(np.dot(residual, residual))
            if not np.isfinite(chi2):
                return FINITE_CHI2_CEILING
            return min(chi2, FINITE_CHI2_CEILING)
        except Exception as exc:
            print(f'[ERROR] additive SED evaluation failed: {exc}')
            return FINITE_CHI2_CEILING

    def chi_squared(self, vector):
        vector = np.clip(vector, self._bounds_lo, self._bounds_hi)
        values = self._vector_to_values(vector, optimizer_space=True)
        chi2 = self.chi_squared_physical(values)
        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2, self.best_params = chi2, values
        if self.n_evaluations % 20 == 0:
            print(f'  eval {self.n_evaluations}: chi2={chi2:.4f}')
        return chi2 if np.isfinite(chi2) else 1e100

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        self.n_evaluations, self.best_chi2 = 0, np.inf
        x0 = (self._values_to_vector(initial_guess, optimizer_space=True)
              if initial_guess is not None
              else 0.5 * (self._bounds_lo + self._bounds_hi))
        print(f'\nStarting {self.method} additive SED optimization ...')
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
        result.chi2_red = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        return result

    def log_prior(self, theta):
        for i, entry in enumerate(self._entries):
            lo, hi = entry['prior']
            if not np.isfinite(theta[i]) or not lo <= theta[i] <= hi:
                return -np.inf
        return 0.0

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        self.n_mcmc_evaluations += 1
        chi2 = self.chi_squared_physical(self._vector_to_values(theta))
        return lp - 0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def set_best_fit_values(self, values):
        self.best_params = self._validate_best_fit(values)
        self.best_chi2 = self.chi_squared_physical(self.best_params)
        return self.best_params

    @staticmethod
    def _draw_truncated_normal(rng, center, low, high, sigma, count):
        if sigma <= 0:
            raise ValueError('Walker FWHM must be positive.')
        result = np.empty(count)
        filled = 0
        while filled < count:
            draws = rng.normal(center, sigma, max(64, 4 * (count - filled)))
            draws = draws[(draws >= low) & (draws <= high)]
            take = min(len(draws), count - filled)
            result[filled:filled + take] = draws[:take]
            filled += take
        return result

    def _record_walker_summary(self, positions, label):
        self.initial_walker_positions = positions.copy()
        self.initial_walker_summary = {}
        print(f'\nInitial walker spread ({label}):')
        for i, entry in enumerate(self._entries):
            values = positions[:, i]
            summary = {'min': float(values.min()), 'max': float(values.max()),
                       'std': float(values.std())}
            if entry['log']:
                summary['log10_std'] = float(np.log10(values).std())
                spread = f'log10_std={summary["log10_std"]:.6g}'
            else:
                spread = f'std={summary["std"]:.6g}'
            print(f'  {entry["label"]:<28} min={summary["min"]:.6g} '
                  f'max={summary["max"]:.6g} {spread}')
            self.initial_walker_summary[entry['label']] = summary

    def _prior_walker_positions(self, nwalkers, seed):
        rng = np.random.default_rng(seed)
        positions = np.empty((nwalkers, self.ndim))
        for i, entry in enumerate(self._entries):
            lo, hi = entry['prior']
            if entry['log']:
                positions[:, i] = 10**rng.uniform(
                    np.log10(lo), np.log10(hi), nwalkers)
            else:
                positions[:, i] = rng.uniform(lo, hi, nwalkers)
        self._record_walker_summary(positions, 'prior')
        return positions

    def _best_walker_positions(self, center, nwalkers, fwhm_frac, seed):
        rng = np.random.default_rng(seed)
        center = self._values_to_vector(center)
        positions = np.empty((nwalkers, self.ndim))
        for i, entry in enumerate(self._entries):
            lo, hi = entry['prior']
            value = center[i]
            if not lo <= value <= hi:
                raise ValueError(f'{entry["label"]}={value} is outside its prior.')
            if entry['log']:
                low, high, middle = np.log10([lo, hi, value])
                sigma = fwhm_frac * (high - low) * self._FWHM_TO_SIGMA
                positions[:, i] = 10**self._draw_truncated_normal(
                    rng, middle, low, high, sigma, nwalkers)
            else:
                sigma = fwhm_frac * (hi - lo) * self._FWHM_TO_SIGMA
                positions[:, i] = self._draw_truncated_normal(
                    rng, value, lo, hi, sigma, nwalkers)
        self._record_walker_summary(positions, 'best-fit truncated Gaussian')
        return positions

    def _store_results(self, chain, log_prob, burn_in=0, thin=1):
        chain = np.asarray(chain, dtype=np.float64)
        log_prob = np.asarray(log_prob, dtype=np.float64)
        if chain.ndim != 3 or chain.shape[-1] != self.ndim:
            raise ValueError(f'Invalid chain shape: {chain.shape}')
        if log_prob.shape != chain.shape[:2]:
            raise ValueError('log_prob shape does not match chain.')
        if burn_in < 0 or burn_in >= chain.shape[0] or thin < 1:
            raise ValueError('Invalid burn-in or thinning.')
        self.mcmc_chain, self.mcmc_log_prob = chain, log_prob
        self.flat_samples = chain[burn_in::thin].reshape(-1, self.ndim)
        flat_log_prob = log_prob[burn_in::thin].reshape(-1)
        if not np.any(np.isfinite(flat_log_prob)):
            raise ValueError('No finite posterior samples remain.')
        index = int(np.nanargmax(flat_log_prob))
        self.mcmc_best_log_prob = float(flat_log_prob[index])
        self.mcmc_best_chi2 = -2 * self.mcmc_best_log_prob
        self.mcmc_best_params = self._vector_to_values(self.flat_samples[index])
        if self.best_params is None or self.mcmc_best_chi2 < self.best_chi2:
            self.best_chi2 = self.mcmc_best_chi2
            self.best_params = self.mcmc_best_params
        self.mcmc_param_summary = self._summarise_samples(self.flat_samples)

    def run_mcmc(self, best_fit_values=None, nwalkers=None, nsteps=1000,
                 burn_in=0, thin=1, best_fit_fwhm_frac=0.02,
                 scatter_frac=None, seed=42, init='best_fit',
                 restart_from=None, initial_state=None, save_path=None,
                 backend_path=None, resume_backend=False,
                 reset_backend=True, progress=True, **run_kwargs):
        import emcee
        if scatter_frac is not None:
            best_fit_fwhm_frac = scatter_frac
        nwalkers = nwalkers or max(32, 2 * self.ndim + 2)
        if nwalkers < 2 * self.ndim:
            raise ValueError('nwalkers must be at least 2 * ndim.')
        backend, last = None, None
        if backend_path is not None:
            backend = emcee.backends.HDFBackend(str(backend_path))
            if resume_backend:
                if backend.iteration <= 0:
                    raise ValueError(f'Cannot resume empty backend: {backend_path}')
                last = backend.get_last_sample()
                if last.coords.shape[1] != self.ndim:
                    raise ValueError('Backend ndim differs from fitter ndim.')
                nwalkers = last.coords.shape[0]
                print(f'Resuming {backend_path} at step {backend.iteration}.')
            elif reset_backend:
                backend.reset(nwalkers, self.ndim)
                write_parameter_names(backend_path, self.param_names)
        if last is not None:
            positions = last
        elif restart_from is not None:
            positions = self.load_chain(restart_from)
        elif initial_state is not None:
            positions = np.asarray(initial_state, dtype=np.float64)
        elif init == 'prior':
            positions = self._prior_walker_positions(nwalkers, seed)
        elif init == 'best_fit':
            if best_fit_values is not None:
                center = self.set_best_fit_values(best_fit_values)
            elif self.best_params is not None:
                center = self.best_params
            elif self.input_best_fit_values is not None:
                center = self.set_best_fit_values(self.input_best_fit_values)
            else:
                raise RuntimeError('Run fit(), provide best values, or use prior init.')
            positions = self._best_walker_positions(
                center, nwalkers, best_fit_fwhm_frac, seed)
        else:
            raise ValueError('init must be "best_fit" or "prior".')
        coords = np.asarray(last.coords if last is not None else positions)
        if coords.ndim != 2 or coords.shape[1] != self.ndim:
            raise ValueError(f'Walker positions need shape (walkers, {self.ndim}).')
        if coords.shape[0] < 2 * self.ndim:
            raise ValueError('Initial state has too few walkers.')
        if not np.all(np.isfinite([self.log_prior(row) for row in coords])):
            raise ValueError('Initial walkers must lie inside all priors.')
        print(f'\nStarting additive MCMC: {coords.shape[0]} walkers, '
              f'{nsteps} steps, {self.ndim} dimensions.')
        self.n_mcmc_evaluations = 0
        sampler = emcee.EnsembleSampler(
            coords.shape[0], self.ndim, self.log_probability, backend=backend)
        sampler.run_mcmc(positions, nsteps, progress=progress, **run_kwargs)
        self.mcmc_sampler = sampler
        self._store_results(
            sampler.get_chain(), sampler.get_log_prob(), burn_in, thin)
        if save_path is not None:
            self.save_chain(save_path)
        return sampler

    def save_chain(self, filename):
        if self.mcmc_chain is None:
            raise RuntimeError('Run or load MCMC before saving.')
        np.savez(
            filename, chain=self.mcmc_chain, log_prob=self.mcmc_log_prob,
            last_positions=self.mcmc_chain[-1],
            param_names=np.asarray(self.param_names, dtype=str),
            component_names=np.asarray(self.component_names, dtype=str),
            prior_lows=np.asarray([self.prior_ranges[k][0] for k in self.param_names]),
            prior_highs=np.asarray([self.prior_ranges[k][1] for k in self.param_names]))
        return filename

    def load_chain(self, filename, burn_in=0, thin=1, restore_results=False):
        with np.load(filename) as data:
            names = list(data['param_names'].astype(str))
            if names != self.param_names:
                raise ValueError(f'Chain parameter order differs: {names}')
            chain = np.array(data['chain'], dtype=np.float64)
            log_prob = np.array(data['log_prob'], dtype=np.float64)
            last = np.array(data['last_positions'], dtype=np.float64)
        self.loaded_chain, self.loaded_log_prob = chain, log_prob
        self.loaded_last_positions = last
        if restore_results:
            self._store_results(chain, log_prob, burn_in, thin)
        return last

    def load_backend(self, filename, burn_in=0, thin=1):
        import emcee
        backend = emcee.backends.HDFBackend(str(filename), read_only=True)
        if backend.iteration <= 0:
            raise ValueError(f'Backend contains no samples: {filename}')
        self._store_results(
            backend.get_chain(), backend.get_log_prob(), burn_in, thin)
        return self.mcmc_chain[-1]

    def fit_then_mcmc(self, best_fit_values=None, initial_guess=None,
                      maxiter=1000, fit_verbose=True, **mcmc_kwargs):
        if best_fit_values is not None:
            self.set_best_fit_values(best_fit_values)
            result = None
        elif self.input_best_fit_values is not None:
            self.set_best_fit_values(self.input_best_fit_values)
            result = None
        else:
            result = self.fit(initial_guess, maxiter, fit_verbose)
        return result, self.run_mcmc(**mcmc_kwargs)

    def run_prior_mcmc(self, **kwargs):
        return self.run_mcmc(init='prior', **kwargs)

    def restart_mcmc(self, filename, **kwargs):
        return self.run_mcmc(restart_from=filename, **kwargs)

    def resume_backend_mcmc(self, backend_path, **kwargs):
        return self.run_mcmc(
            backend_path=backend_path, resume_backend=True, **kwargs)

    def _summarise_samples(self, samples):
        q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
        return {name: {
            'median': float(q50[i]), 'minus_1sigma': float(q50[i] - q16[i]),
            'plus_1sigma': float(q84[i] - q50[i]),
            'q16': float(q16[i]), 'q84': float(q84[i])}
            for i, name in enumerate(self.param_names)}

    def mcmc_diagnostics(self):
        if self.mcmc_sampler is None:
            raise RuntimeError('Diagnostics require run_mcmc().')
        acceptance = self.mcmc_sampler.acceptance_fraction
        result = {
            'acceptance_fraction_mean': float(acceptance.mean()),
            'acceptance_fraction_min': float(acceptance.min()),
            'acceptance_fraction_max': float(acceptance.max())}
        try:
            tau = self.mcmc_sampler.get_autocorr_time(tol=0)
            result['autocorr_time'] = dict(zip(self.param_names, map(float, tau)))
        except Exception as exc:
            result['autocorr_time_error'] = str(exc)
        return result

    def summary(self, include_mass_abundances=True):
        if self.best_params is None:
            print('No fit result yet.')
            return
        dof = max(len(self.obs) - self.ndim, 1)
        print(f'\nBest chi2 = {self.best_chi2:.6g} '
              f'(reduced chi2 = {self.best_chi2 / dof:.6g}, dof = {dof})')
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
        print('\nAdditive SED posterior summary:')
        print(f'{"Parameter":<28} {"median":>14} {"-1sigma":>14} {"+1sigma":>14}')
        print('-' * 73)
        for name in self.param_names:
            value = self.mcmc_param_summary[name]
            print(f'{name:<28} {value["median"]:>14.6g} '
                  f'{value["minus_1sigma"]:>14.6g} '
                  f'{value["plus_1sigma"]:>14.6g}')
        dof = max(len(self.obs) - self.ndim, 1)
        print(f'\nBest sampled chi2 = {self.mcmc_best_chi2:.6g} '
              f'(reduced chi2 = {self.mcmc_best_chi2 / dof:.6g}, dof = {dof})')
        self.summary(include_mass_abundances=False)
        print(self.format_component_mass_abundances(), end='')

    def get_component_masses(self, values=None):
        values = self.best_params if values is None else self._validate_best_fit(values)
        if values is None:
            raise RuntimeError('No fitted parameters are available.')
        items = [
            (name, self._full_component_params(values, name))
            for name in self.component_names]

        def calculate(item):
            name, params = item
            return name, self.sed_objects[name].get_total_mass(**params)

        if self._executor is not None:
            return dict(self._executor.map(calculate, items))
        return dict(map(calculate, items))

    def get_component_mass_abundances(self, values=None, masses=None):
        """Return masses and composition fractions within each ring.

        ``fraction_of_ring`` is the requested composition abundance. The
        additional ``fraction_of_total`` reports the component's contribution
        to all fitted rings combined.
        """
        if masses is None:
            masses = self.get_component_masses(values)
        else:
            masses = {name: float(masses[name]) for name in self.component_names}
        scale = max([abs(mass) for mass in masses.values()] + [1e-300])
        tolerance = 64.0 * np.finfo(float).eps * scale
        for name, mass in list(masses.items()):
            if mass < -tolerance:
                raise ValueError(
                    f'Component {name} has an unphysical negative mass: {mass}')
            if mass < 0.0:
                masses[name] = 0.0

        total_mass = float(sum(masses.values()))
        group_names = list(dict.fromkeys(self.mass_abundance_groups.values()))
        groups = {}
        for group_name in group_names:
            component_names = [
                name for name in self.component_names
                if self.mass_abundance_groups[name] == group_name]
            group_mass = float(sum(masses[name] for name in component_names))
            components = {}
            for name in component_names:
                composition_name = (
                    name[len(group_name) + 1:]
                    if name.startswith(f'{group_name}.') else name)
                fraction_of_group = (
                    masses[name] / group_mass if group_mass > 0.0 else 0.0)
                fraction_of_total = (
                    masses[name] / total_mass if total_mass > 0.0 else 0.0)
                components[name] = {
                    'composition': composition_name,
                    'mass_earth': masses[name],
                    'fraction_of_ring': fraction_of_group,
                    'percent_of_ring': 100.0 * fraction_of_group,
                    'fraction_of_total': fraction_of_total,
                    'percent_of_total': 100.0 * fraction_of_total,
                }
            groups[group_name] = {
                'mass_earth': group_mass,
                'fraction_of_total': (
                    group_mass / total_mass if total_mass > 0.0 else 0.0),
                'percent_of_total': (
                    100.0 * group_mass / total_mass
                    if total_mass > 0.0 else 0.0),
                'components': components,
            }
        self.component_mass_abundances = {
            'total_mass_earth': total_mass, 'rings': groups}
        return self.component_mass_abundances

    def format_component_mass_abundances(self, values=None, masses=None):
        """Return a readable mass-abundance table for summaries and files."""
        results = self.get_component_mass_abundances(values, masses)
        lines = ['\nDust mass composition at the best-fit point:']
        for ring_name, ring in results['rings'].items():
            lines.append(
                f'{ring_name}: {ring["mass_earth"]:.8g} Earth masses '
                f'({ring["percent_of_total"]:.4f}% of all rings)')
            for component in ring['components'].values():
                lines.append(
                    f'  {component["composition"]:<28} '
                    f'{component["mass_earth"]:>14.8g} Earth masses  '
                    f'{component["percent_of_ring"]:>10.6f}% of ring  '
                    f'{component["percent_of_total"]:>10.6f}% of total')
        lines.append(
            f'All rings total: {results["total_mass_earth"]:.8g} Earth masses')
        return '\n'.join(lines) + '\n'

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run fit/MCMC or restore a chain first.')
        outputs = self.component_seds(self.best_params, False)
        order = np.argsort(self.wavelengths)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(self.wavelengths[order], self.obs[order],
                    yerr=self.obs_err[order], fmt='o', color='black',
                    capsize=4, zorder=5, label='Observations')
        total = np.real(outputs[0]).copy()
        for name, output in zip(self.component_names, outputs):
            ax.plot(self.wavelengths[order], np.real(output)[order], '--',
                    lw=1.4, label=name)
        for output in outputs[1:]:
            total += np.real(output)
        ax.plot(self.wavelengths[order], total[order], color='black', lw=2,
                label='Combined total')
        ax.set(xscale='log', yscale='log', xlabel='Wavelength [um]',
               ylabel='Flux [Jy]')
        dof = max(len(self.obs) - self.ndim, 1)
        ax.set_title(f'Best additive SED (reduced chi2 = '
                     f'{self.best_chi2 / dof:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def mcmc_walkers_plot(self, discard=0, thin=1, max_walkers=None,
                          alpha=0.35, **plot_kwargs):
        if self.mcmc_chain is None:
            raise RuntimeError('Run MCMC or restore a chain first.')
        chain = self.mcmc_chain[discard::thin]
        if chain.size == 0:
            raise ValueError('No samples remain after discard/thin.')
        nsteps, nwalkers, _ = chain.shape
        walkers = np.arange(nwalkers)
        if max_walkers is not None and nwalkers > max_walkers:
            walkers = np.linspace(0, nwalkers - 1, max_walkers, dtype=int)
        fig, axes = plt.subplots(
            self.ndim, 1, figsize=(10, max(2 * self.ndim, 3)),
            sharex=True, squeeze=False)
        x = np.arange(discard, discard + nsteps * thin, thin)
        style = {'color': 'black', 'lw': 0.6, 'alpha': alpha, **plot_kwargs}
        for i, entry in enumerate(self._entries):
            ax = axes[i, 0]
            for walker in walkers:
                ax.plot(x, chain[:, walker, i], **style)
            ax.set_ylabel(entry['label'])
            lo, hi = entry['prior']
            ax.axhline(lo, color='tab:red', lw=0.7, alpha=0.5)
            ax.axhline(hi, color='tab:red', lw=0.7, alpha=0.5)
            if entry['log'] and np.all(chain[:, walkers, i] > 0):
                ax.set_yscale('log')
            ax.grid(True, alpha=0.25)
        axes[-1, 0].set_xlabel('MCMC step')
        fig.suptitle('Additive SED MCMC walker traces')
        fig.tight_layout()
        return fig

    def mcmc_corner_plot(self, max_samples=None, seed=42, **corner_kwargs):
        if self.flat_samples is None:
            raise RuntimeError('Run MCMC or restore a chain first.')
        samples = self.flat_samples
        if max_samples is not None and len(samples) > max_samples:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(samples), max_samples, replace=False))
            samples = samples[indices]
        defaults = {'title_kwargs': {'fontsize': 10}}
        if self.best_params is not None:
            defaults['truths'] = self._values_to_vector(self.best_params)
        defaults.update(corner_kwargs)
        return make_corner_plot(
            samples, self.param_names, self.log_params, **defaults)
