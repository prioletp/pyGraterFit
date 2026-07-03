"""Consistent corner-plot formatting shared by all pyGrater fitters."""

import numpy as np


DEFAULT_LOG_SPACE_PARAMETER_NAMES = {
    'M_tot', 'A_norm', 'a_min', 'r0',
}


def infer_log_space_parameters(parameter_names):
    """Infer qualified log parameters such as ``ring1.olivine.A_norm``."""
    return {
        name for name in parameter_names
        if name.rsplit('.', 1)[-1] in DEFAULT_LOG_SPACE_PARAMETER_NAMES
    }


def make_corner_plot(samples, parameter_names, log_parameter_names=(),
                     **corner_arguments):
    """Create a corner plot with scientific titles and native log axes."""
    import corner

    samples = np.asarray(samples, dtype=np.float64)
    parameter_names = list(parameter_names)
    log_parameter_names = set(log_parameter_names)
    unknown = log_parameter_names.difference(parameter_names)
    if unknown:
        raise ValueError(f'Unknown log-space parameters: {sorted(unknown)}')
    for index, name in enumerate(parameter_names):
        if name in log_parameter_names and np.any(samples[:, index] <= 0):
            raise ValueError(
                f'Cannot use a logarithmic corner axis for non-positive {name}.')

    defaults = {
        'labels': parameter_names,
        'show_titles': True,
        'quantiles': [0.16, 0.5, 0.84],
        'title_fmt': '.3e',
        'use_math_text': True,
        'axes_scale': [
            'log' if name in log_parameter_names else 'linear'
            for name in parameter_names],
    }
    defaults.update(corner_arguments)
    return corner.corner(samples, **defaults)
