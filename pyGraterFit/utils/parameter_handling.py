"""Shared parsing and resolution of fitted, fixed, and dependent parameters.

Commented examples
------------------
Use the same syntax in the ``params`` dictionary of every fitter::

    params = {
        'r0': 1.0,                         # fixed
        'h0': lambda p: 0.05 * p['r0'],   # dependent on fixed r0
        'a_min': (0.1, 10.0),              # fitted
        'a_max': lambda p: 100 * p['a_min'],  # dependent on fitted a_min
    }

Dependencies may be chained and dictionary order does not matter::

    params = {
        'h_outer': lambda p: 2.0 * p['h0'],
        'h0': lambda p: 0.05 * p['r0'],
        'r0': (0.5, 5.0),
    }

For component fitters, put the callable in each component dictionary. Shared
dependent parameters should use the same named function object::

    def scale_height(p):
        return 0.05 * p['r0']

    params_by_component = {
        'ring1_olivine': {'r0': (0.5, 2.0), 'h0': scale_height},
        'ring1_astroSi': {'r0': (0.5, 2.0), 'h0': scale_height},
    }
    shared_parameter_names = ('r0', 'h0')
"""

from collections.abc import Mapping

import numpy as np


def is_parameter_range(value):
    """Return whether ``value`` is a two-element fitted-parameter range."""
    return isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2


def split_parameter_specifications(parameters, context='parameters'):
    """Split one parameter dictionary into free, fixed, and dependent parts.

    A callable defines a dependent parameter and receives a read-only mapping
    containing all free, fixed, and lazily resolved dependent parameters.
    """
    free = {}
    fixed = {}
    dependent = {}
    for name, specification in parameters.items():
        if callable(specification):
            dependent[name] = specification
        elif is_parameter_range(specification):
            low, high = map(float, specification)
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(
                    f'Invalid fitted range for {context}.{name}: '
                    f'{specification!r}')
            free[name] = (low, high)
        elif isinstance(specification,
                        (int, float, np.integer, np.floating)):
            fixed[name] = (
                int(specification)
                if isinstance(specification, (int, np.integer))
                else float(specification))
        else:
            raise ValueError(
                f'Invalid parameter specification for {context}.{name}: '
                f'{type(specification).__name__}. Use a scalar, a two-value '
                'range, or a callable dependency.')
    return free, fixed, dependent


class _LazyParameterMapping(Mapping):
    def __init__(self, independent, dependent, context):
        self._values = dict(independent)
        self._dependent = dependent
        self._context = context
        self._resolving = []

    def __getitem__(self, name):
        if name in self._values:
            return self._values[name]
        if name not in self._dependent:
            requested_by = self._resolving[-1] if self._resolving else None
            prefix = (f'Dependent parameter {self._context}.{requested_by}'
                      if requested_by else self._context)
            raise KeyError(f'{prefix} references unknown parameter {name!r}.')
        if name in self._resolving:
            start = self._resolving.index(name)
            cycle = self._resolving[start:] + [name]
            raise ValueError(
                f'Dependent-parameter cycle in {self._context}: '
                + ' -> '.join(cycle))

        self._resolving.append(name)
        try:
            value = self._dependent[name](self)
        finally:
            self._resolving.pop()
        if (not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(
                f'Dependent parameter {self._context}.{name} must return one '
                f'finite scalar, received {value!r}.')
        self._values[name] = (
            int(value) if isinstance(value, (int, np.integer))
            else float(value))
        return self._values[name]

    def __iter__(self):
        return iter(dict.fromkeys((*self._values, *self._dependent)))

    def __len__(self):
        return len(set(self._values) | set(self._dependent))

    def resolve_all(self):
        for name in self._dependent:
            self[name]
        return dict(self._values)


def resolve_parameters(free_values, fixed_values, dependent_functions,
                       context='parameters'):
    """Return a complete physical parameter dictionary."""
    independent = {**free_values, **fixed_values}
    if not dependent_functions:
        return independent
    overlap = set(independent) & set(dependent_functions)
    if overlap:
        raise ValueError(
            f'{context} parameters are both independent and dependent: '
            f'{sorted(overlap)}')
    return _LazyParameterMapping(
        independent, dependent_functions, context).resolve_all()
