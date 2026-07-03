"""Plot saved MCMC results from every pyGrater optimized fitter.

This module understands both NPZ naming schemes used by the SED and
SED+interferometry fitters, as well as emcee HDF backends. Generic posterior
plots do not require rebuilding the model. Pass an initialized fitter to
``plot_fitter_results`` to additionally plot the best physical model.
"""

import argparse
import contextlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault(
    'MPLCONFIGDIR',
    str(Path(tempfile.gettempdir()) / 'pygrater_matplotlib_cache'))

import matplotlib.pyplot as plt

from pyGraterFit.utils.corner_plotting import (
    infer_log_space_parameters,
    make_corner_plot,
)


@dataclass
class SavedMCMCResults:
    chain: np.ndarray
    log_probability: np.ndarray
    parameter_names: list
    acceptance_fraction: np.ndarray | None = None
    prior_lower: np.ndarray | None = None
    prior_upper: np.ndarray | None = None
    source: Path | None = None

    @property
    def n_steps(self):
        return self.chain.shape[0]

    @property
    def n_walkers(self):
        return self.chain.shape[1]

    @property
    def n_parameters(self):
        return self.chain.shape[2]


def _names_or_defaults(names, n_parameters):
    if names is None:
        return [f'parameter_{index}' for index in range(n_parameters)]
    names = [str(name) for name in names]
    if len(names) != n_parameters:
        raise ValueError(
            f'Got {len(names)} names for {n_parameters} parameters.')
    return names


def _validate_results(results):
    if results.chain.ndim != 3:
        raise ValueError('Chain must have shape (steps, walkers, parameters).')
    if results.log_probability.shape != results.chain.shape[:2]:
        raise ValueError('Log-probability shape does not match the chain.')
    if results.n_steps == 0:
        raise ValueError('The saved MCMC contains no committed samples.')
    if not np.all(np.isfinite(results.chain)):
        raise ValueError('The committed chain contains non-finite coordinates.')
    results.parameter_names = _names_or_defaults(
        results.parameter_names, results.n_parameters)
    return results


def load_saved_mcmc(path, parameter_names=None, hdf_group='mcmc'):
    """Load an NPZ chain or HDF backend, ignoring uncommitted HDF rows."""
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path) as saved:
            chain = np.asarray(saved['chain'], dtype=np.float64)
            log_key = ('log_prob' if 'log_prob' in saved.files
                       else 'log_probability')
            log_probability = np.asarray(saved[log_key], dtype=np.float64)
            name_key = ('param_names' if 'param_names' in saved.files
                        else ('parameter_names'
                              if 'parameter_names' in saved.files else None))
            saved_names = None if name_key is None else saved[name_key].astype(str)
            prior_lower = (np.asarray(saved['prior_lows'], dtype=np.float64)
                           if 'prior_lows' in saved.files else None)
            prior_upper = (np.asarray(saved['prior_highs'], dtype=np.float64)
                           if 'prior_highs' in saved.files else None)
        names = parameter_names if parameter_names is not None else saved_names
        # The fraction of transitions that moved is a useful NPZ-only proxy.
        acceptance = None
        if chain.shape[0] > 1:
            acceptance = np.mean(
                np.any(chain[1:] != chain[:-1], axis=2), axis=0)
        return _validate_results(SavedMCMCResults(
            chain, log_probability, names, acceptance,
            prior_lower, prior_upper, path))

    with h5py.File(path, 'r') as backend:
        group = backend[hdf_group]
        committed = int(group.attrs['iteration'])
        embedded_names = group.attrs.get('parameter_names')
        chain = np.asarray(group['chain'][:committed], dtype=np.float64)
        log_probability = np.asarray(
            group['log_prob'][:committed], dtype=np.float64)
        acceptance = np.asarray(group['accepted'], dtype=np.float64) / max(
            committed, 1)
    if parameter_names is None and embedded_names is not None:
        parameter_names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in embedded_names]
    if parameter_names is None and '_backend' in path.stem:
        chain_stem = path.stem.split('_backend', 1)[0] + '_chain.npz'
        neighboring_chain = path.with_name(chain_stem)
        if neighboring_chain.exists():
            with np.load(neighboring_chain) as saved:
                name_key = ('param_names' if 'param_names' in saved.files
                            else ('parameter_names'
                                  if 'parameter_names' in saved.files else None))
                if name_key is not None:
                    parameter_names = saved[name_key].astype(str)
    return _validate_results(SavedMCMCResults(
        chain, log_probability, parameter_names, acceptance, source=path))


def posterior_samples(results, burn_in=0, thin=1):
    if burn_in < 0 or burn_in >= results.n_steps:
        raise ValueError(f'burn_in must be between 0 and {results.n_steps - 1}.')
    if thin < 1:
        raise ValueError('thin must be at least one.')
    chain = results.chain[burn_in::thin]
    log_probability = results.log_probability[burn_in::thin]
    samples = chain.reshape(-1, results.n_parameters)
    flat_log_probability = log_probability.reshape(-1)
    finite = np.isfinite(flat_log_probability)
    if not np.any(finite):
        raise ValueError('No finite posterior samples remain.')
    return samples[finite], flat_log_probability[finite]


def best_sample(results, burn_in=0, thin=1):
    samples, log_probability = posterior_samples(results, burn_in, thin)
    index = int(np.argmax(log_probability))
    return samples[index], float(log_probability[index])


def plot_walkers(results, discard=0, thin=1, max_walkers=None):
    chain = results.chain[discard::thin]
    walker_indices = np.arange(results.n_walkers)
    if max_walkers is not None and len(walker_indices) > max_walkers:
        walker_indices = np.linspace(
            0, results.n_walkers - 1, max_walkers, dtype=int)
    figure, axes = plt.subplots(
        results.n_parameters, 1,
        figsize=(11, max(2.0 * results.n_parameters, 3.5)),
        sharex=True, squeeze=False)
    steps = np.arange(discard, discard + len(chain) * thin, thin)
    for parameter_index, name in enumerate(results.parameter_names):
        axis = axes[parameter_index, 0]
        axis.plot(
            steps, chain[:, walker_indices, parameter_index],
            color='black', alpha=0.22, linewidth=0.55)
        if results.prior_lower is not None:
            axis.axhline(results.prior_lower[parameter_index], color='tab:red',
                         linewidth=0.7, alpha=0.6)
            axis.axhline(results.prior_upper[parameter_index], color='tab:red',
                         linewidth=0.7, alpha=0.6)
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel('MCMC step')
    figure.suptitle('Walker traces')
    figure.tight_layout()
    return figure


def _corner_ranges(results, range_fraction, parameter_ranges):
    if range_fraction is not None and not 0.0 < range_fraction <= 1.0:
        raise ValueError('corner range fraction must be in (0, 1].')
    ranges = [range_fraction for _ in results.parameter_names]
    if parameter_ranges == 'prior':
        if results.prior_lower is None or results.prior_upper is None:
            raise ValueError('This saved chain does not contain prior bounds.')
        return list(zip(results.prior_lower, results.prior_upper))
    if parameter_ranges is None:
        return ranges
    unknown = set(parameter_ranges).difference(results.parameter_names)
    if unknown:
        raise ValueError(f'Unknown corner-range parameters: {sorted(unknown)}')
    for name, limits in parameter_ranges.items():
        lower, upper = map(float, limits)
        if not np.isfinite(lower + upper) or lower >= upper:
            raise ValueError(f'Invalid corner range for {name}: {limits}')
        ranges[results.parameter_names.index(name)] = (lower, upper)
    return ranges


def plot_corner(results, burn_in=0, thin=1, max_samples=50000, seed=42,
                range_fraction=0.997, parameter_ranges=None,
                log_parameter_names=None,
                **corner_arguments):
    samples, _ = posterior_samples(results, burn_in, thin)
    if max_samples is not None and len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        samples = samples[rng.choice(len(samples), max_samples, replace=False)]
    defaults = {
        'title_kwargs': {'fontsize': 9},
        'range': _corner_ranges(
            results, range_fraction, parameter_ranges),
    }
    defaults.update(corner_arguments)
    if log_parameter_names is None:
        log_parameter_names = infer_log_space_parameters(
            results.parameter_names)
    return make_corner_plot(
        samples, results.parameter_names, log_parameter_names, **defaults)


def plot_log_probability(results):
    figure, (trace_axis, best_axis) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    steps = np.arange(results.n_steps)
    finite_log_probability = np.where(
        np.isfinite(results.log_probability), results.log_probability, np.nan)
    trace_axis.plot(
        steps, finite_log_probability, color='black', alpha=0.12,
        linewidth=0.45)
    trace_axis.set_ylabel('Log probability')
    trace_axis.grid(alpha=0.2)
    best_axis.plot(
        steps, np.nanmax(finite_log_probability, axis=1),
        color='tab:blue', label='Best walker')
    best_axis.plot(
        steps, np.nanmedian(finite_log_probability, axis=1),
        color='tab:orange', label='Walker median')
    best_axis.set(xlabel='MCMC step', ylabel='Log probability')
    best_axis.legend()
    best_axis.grid(alpha=0.2)
    return figure


def plot_acceptance(results):
    if results.acceptance_fraction is None:
        return None
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    walker = np.arange(results.n_walkers)
    axis.bar(walker, results.acceptance_fraction, color='tab:blue')
    axis.axhline(np.mean(results.acceptance_fraction), color='black',
                 linestyle='--', label='Mean')
    axis.set(xlabel='Walker', ylabel='Acceptance fraction', ylim=(0, 1))
    axis.legend()
    axis.grid(axis='y', alpha=0.2)
    return figure


def posterior_summary_text(results, burn_in=0, thin=1,
                           number_of_observations=None):
    samples, log_probability = posterior_samples(results, burn_in, thin)
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    best_index = int(np.argmax(log_probability))
    best_log_probability = float(log_probability[best_index])
    best_chi_squared = -2.0 * best_log_probability
    lines = [
        f'Source: {results.source}',
        f'Steps: {results.n_steps}',
        f'Walkers: {results.n_walkers}',
        f'Parameters: {results.n_parameters}',
        f'Burn-in: {burn_in}',
        f'Thin: {thin}',
        f'Finite posterior samples: {len(samples)}',
        f'Best log probability: {best_log_probability:.12g}',
        f'Best chi2: {best_chi_squared:.12g}',
    ]
    if number_of_observations is not None:
        degrees_of_freedom = max(
            int(number_of_observations) - results.n_parameters, 1)
        lines.append(f'Degrees of freedom: {degrees_of_freedom}')
        lines.append(
            f'Reduced chi2: {best_chi_squared / degrees_of_freedom:.12g}')
    if results.acceptance_fraction is not None:
        lines.append(
            f'Mean acceptance fraction: '
            f'{np.mean(results.acceptance_fraction):.8g}')
    lines.extend(['', 'Posterior percentiles:',
                  f'{"Parameter":<36} {"q16":>14} {"median":>14} {"q84":>14} {"best":>14}'])
    for index, name in enumerate(results.parameter_names):
        lines.append(
            f'{name:<36} {q16[index]:>14.7g} {q50[index]:>14.7g} '
            f'{q84[index]:>14.7g} {samples[best_index, index]:>14.7g}')
    return '\n'.join(lines) + '\n'


def _fitter_parameter_names(fitter):
    names = getattr(fitter, 'param_names', None)
    if names is None:
        names = getattr(fitter, 'parameter_names', None)
    if names is None:
        raise TypeError('The fitter does not expose parameter names.')
    return list(names)


def _best_vector_to_fitter_parameters(fitter, vector):
    if hasattr(fitter, '_mcmc_vec_to_dict'):
        return fitter._mcmc_vec_to_dict(vector)
    if hasattr(fitter, '_mcmc_vec_to_dicts'):
        return fitter._mcmc_vec_to_dicts(vector)
    if hasattr(fitter, '_vector_to_values'):
        return fitter._vector_to_values(vector)
    if hasattr(fitter, 'parameter_vector_to_dictionary'):
        return fitter.parameter_vector_to_dictionary(vector)
    return dict(zip(_fitter_parameter_names(fitter), map(float, vector)))


def restore_results_in_fitter(fitter, results, burn_in=0, thin=1):
    """Populate any optimized MCMC fitter from normalized saved results."""
    expected_names = _fitter_parameter_names(fitter)
    if results.parameter_names != expected_names:
        if all(name.startswith('parameter_') for name in results.parameter_names):
            results.parameter_names = expected_names
        else:
            raise ValueError(
                f'Parameter order differs: {results.parameter_names} != '
                f'{expected_names}')
    samples, log_probability = posterior_samples(results, burn_in, thin)
    best_index = int(np.argmax(log_probability))
    vector = samples[best_index]
    best_parameters = _best_vector_to_fitter_parameters(fitter, vector)
    summary = np.percentile(samples, [16, 50, 84], axis=0)
    summary_dictionary = {
        name: {'q16': float(summary[0, index]),
               'median': float(summary[1, index]),
               'q84': float(summary[2, index]),
               'minus_1sigma': float(summary[1, index] - summary[0, index]),
               'plus_1sigma': float(summary[2, index] - summary[1, index])}
        for index, name in enumerate(expected_names)}
    best_chi_squared = float(-2.0 * log_probability[best_index])

    fitter.mcmc_chain = results.chain
    fitter.flat_samples = samples
    if hasattr(fitter, 'mcmc_log_prob'):
        fitter.mcmc_log_prob = results.log_probability
        fitter.mcmc_best_params = best_parameters
        fitter.mcmc_best_chi2 = best_chi_squared
        fitter.mcmc_best_log_prob = float(log_probability[best_index])
        fitter.mcmc_param_summary = summary_dictionary
        fitter.best_params = best_parameters
        fitter.best_chi2 = best_chi_squared
    else:
        fitter.mcmc_log_probability = results.log_probability
        fitter.mcmc_best_parameters = best_parameters
        fitter.mcmc_best_chi_squared = best_chi_squared
        fitter.posterior_summary = summary_dictionary
        fitter.best_parameters = best_parameters
        fitter.best_chi_squared = best_chi_squared
        if hasattr(fitter, 'evaluate_physical_parameters'):
            _, components, _, _, _ = fitter.evaluate_physical_parameters(
                best_parameters)
            fitter.best_chi_squared_components = components
    return best_parameters


def _plot_interferometry_best_fit(fitter):
    _, components, models, _, _ = fitter.evaluate_physical_parameters(
        fitter.best_parameters)
    figure, axes = plt.subplots(
        1, 3, figsize=(16, 4.7), constrained_layout=True)
    sed_order = np.argsort(fitter.sed_wavelengths_micron)
    axes[0].errorbar(
        fitter.sed_wavelengths_micron[sed_order],
        fitter.observed_sed_jy[sed_order],
        yerr=fitter.sed_error_jy[sed_order], fmt='o', color='black',
        label='Observed')
    axes[0].plot(
        fitter.sed_wavelengths_micron[sed_order],
        models['sed_jy'][sed_order], color='tab:red', label='Model')
    axes[0].set(xscale='log', yscale='log', xlabel='Wavelength [um]',
                ylabel='Flux [Jy]', title=f'SED chi2={components["sed"]:.3g}')
    axes[0].legend()

    vis2 = fitter.observations['vis2']
    if vis2 is not None:
        spatial_frequency = np.hypot(vis2['u_m'], vis2['v_m']) / (
            vis2['wavelength_m'] * 1e6)
        order = np.argsort(spatial_frequency)
        axes[1].errorbar(
            spatial_frequency[order], vis2['value'][order],
            yerr=vis2['error'][order], fmt='.', color='black', label='Observed')
        axes[1].plot(
            spatial_frequency[order], models['vis2'][order], '.',
            color='tab:red', label='Model')
        axes[1].set(xlabel='Spatial frequency [Mlambda]', ylabel='Squared visibility',
                    title=f'VIS2 chi2={components["vis2"]:.3g}')
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    closure = fitter.observations['closure_phase']
    if closure is not None:
        spatial_frequency = np.hypot(
            closure['u1_m'], closure['v1_m']) / (
                closure['wavelength_m'] * 1e6)
        order = np.argsort(spatial_frequency)
        axes[2].errorbar(
            spatial_frequency[order], closure['value_degrees'][order],
            yerr=closure['error_degrees'][order], fmt='.', color='black',
            label='Observed')
        axes[2].plot(
            spatial_frequency[order],
            models['closure_phase_degrees'][order], '.', color='tab:red',
            label='Model')
        axes[2].set(xlabel='First-baseline spatial frequency [Mlambda]',
                    ylabel='Closure phase [deg]',
                    title=f'Closure chi2={components["closure_phase"]:.3g}')
        axes[2].legend()
    else:
        axes[2].set_visible(False)
    return figure


def save_generic_plots(results, output_directory, burn_in=0, thin=1,
                       corner_max_samples=50000, walkers_max=None, dpi=160,
                       number_of_observations=None,
                       corner_range_fraction=0.997, corner_ranges=None,
                       log_parameter_names=None):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = {
        'walkers.png': plot_walkers(results, max_walkers=walkers_max),
        'corner.png': plot_corner(
            results, burn_in, thin, corner_max_samples,
            range_fraction=corner_range_fraction,
            parameter_ranges=corner_ranges,
            log_parameter_names=log_parameter_names),
        'log_probability.png': plot_log_probability(results),
        'acceptance_fraction.png': plot_acceptance(results),
    }
    for filename, figure in figures.items():
        if figure is not None:
            figure.savefig(output_directory / filename, dpi=dpi)
            plt.close(figure)
    summary = posterior_summary_text(
        results, burn_in, thin, number_of_observations)
    (output_directory / 'posterior_summary.txt').write_text(summary)
    return output_directory


def plot_fitter_results(fitter, saved_path, output_directory, burn_in=0,
                        thin=1, corner_max_samples=50000,
                        walkers_max=None, dpi=160,
                        corner_range_fraction=0.997,
                        corner_ranges=None):
    """Create generic plots plus the best physical model for any optimized fitter."""
    names = _fitter_parameter_names(fitter)
    results = load_saved_mcmc(saved_path, names)
    restore_results_in_fitter(fitter, results, burn_in, thin)
    observation_count = getattr(fitter, 'n_observations', None)
    if observation_count is None:
        observations = getattr(fitter, 'obs', None)
        observation_count = None if observations is None else len(observations)
    output_directory = save_generic_plots(
        results, output_directory, burn_in, thin, corner_max_samples,
        walkers_max, dpi, observation_count, corner_range_fraction,
        corner_ranges,
        getattr(fitter, 'log_params',
                getattr(fitter, 'log_space_parameters', None)))

    if hasattr(fitter, 'evaluate_physical_parameters'):
        best_figure = _plot_interferometry_best_fit(fitter)
    elif hasattr(fitter, 'plot_best_fit'):
        best_figure = fitter.plot_best_fit()
    else:
        best_figure = None
    if best_figure is not None:
        best_figure.savefig(
            Path(output_directory) / 'best_fit_model.png', dpi=dpi)
        plt.close(best_figure)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if hasattr(fitter, 'mcmc_summary'):
            fitter.mcmc_summary()
        else:
            fitter.summary()
    model_summary = buffer.getvalue()
    print(model_summary, end='')
    (Path(output_directory) / 'model_summary.txt').write_text(model_summary)
    return results


def plot_scipy_fitter_results(fitter, output_directory, dpi=160):
    """Save the current best model and summary from any SciPy optimized fitter."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_parameters = getattr(fitter, 'best_params', None)
    if best_parameters is None:
        best_parameters = getattr(fitter, 'best_parameters', None)
    if best_parameters is None:
        raise RuntimeError('Run fitter.fit() before plotting its result.')

    if hasattr(fitter, 'evaluate_physical_parameters'):
        figure = _plot_interferometry_best_fit(fitter)
    elif hasattr(fitter, 'plot_best_fit'):
        figure = fitter.plot_best_fit()
    else:
        raise TypeError('This fitter does not expose a best-model plot.')
    figure.savefig(output_directory / 'best_fit_model.png', dpi=dpi)
    plt.close(figure)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fitter.summary()
    summary = buffer.getvalue()
    print(summary, end='')
    (output_directory / 'model_summary.txt').write_text(summary)
    return output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_mcmc', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('mcmc_plots'))
    parser.add_argument('--parameter-names', nargs='*')
    parser.add_argument('--burn-in', type=int, default=0)
    parser.add_argument('--thin', type=int, default=1)
    parser.add_argument('--corner-max-samples', type=int, default=50000)
    parser.add_argument(
        '--corner-range-fraction', type=float, default=0.997,
        help='Central posterior fraction displayed per parameter (default: 0.997).')
    parser.add_argument(
        '--corner-full-range', action='store_true',
        help='Display sample extrema instead of a robust credible range.')
    parser.add_argument(
        '--corner-prior-range', action='store_true',
        help='Use saved prior bounds (available in pyGrater NPZ chains).')
    parser.add_argument(
        '--corner-range', action='append', default=[], metavar='NAME:LOW:HIGH',
        help='Override one parameter range; may be supplied repeatedly.')
    parser.add_argument(
        '--log-parameter', action='append', default=None, metavar='NAME',
        help=('Declare a log-scaled parameter; repeat for every log parameter. '
              'By default standard pyGrater scale parameters are inferred.'))
    parser.add_argument('--walkers-max', type=int)
    parser.add_argument('--number-of-observations', type=int)
    parser.add_argument('--dpi', type=int, default=160)
    arguments = parser.parse_args()
    results = load_saved_mcmc(
        arguments.saved_mcmc, arguments.parameter_names)
    explicit_corner_ranges = {}
    for specification in arguments.corner_range:
        try:
            name, lower, upper = specification.rsplit(':', 2)
            explicit_corner_ranges[name] = (float(lower), float(upper))
        except ValueError as error:
            raise ValueError(
                f'Invalid --corner-range {specification!r}; expected '
                'NAME:LOW:HIGH.') from error
    if arguments.corner_prior_range and explicit_corner_ranges:
        raise ValueError(
            '--corner-prior-range and --corner-range cannot be combined.')
    corner_ranges = ('prior' if arguments.corner_prior_range else
                     (explicit_corner_ranges or None))
    corner_range_fraction = (
        1.0 if arguments.corner_full_range
        else arguments.corner_range_fraction)
    save_generic_plots(
        results, arguments.output_dir, arguments.burn_in, arguments.thin,
        arguments.corner_max_samples, arguments.walkers_max, arguments.dpi,
        arguments.number_of_observations, corner_range_fraction,
        corner_ranges, arguments.log_parameter)
    print(posterior_summary_text(
        results, arguments.burn_in, arguments.thin,
        arguments.number_of_observations))
    print(f'Plots saved in {arguments.output_dir.resolve()}')


if __name__ == '__main__':
    main()
