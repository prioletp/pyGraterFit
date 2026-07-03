"""Batch-plot every saved pyGrater optimized MCMC run in a directory."""

import argparse
from pathlib import Path

from pyGraterFit.utils.mcmc_plotting import (
    load_saved_mcmc,
    save_generic_plots,
)


def _run_name(path):
    stem = path.stem
    for marker in ('_backend', '_chain'):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def discover_saved_runs(directory, include_broken=False):
    """Return one preferred saved file per run, choosing HDF over NPZ."""
    directory = Path(directory)
    candidates = list(directory.glob('*_backend*.h5'))
    candidates += list(directory.glob('*_chain.npz'))
    if not include_broken:
        candidates = [path for path in candidates if '_broken' not in path.stem]
    selected = {}
    for path in sorted(candidates):
        name = _run_name(path)
        current = selected.get(name)
        if current is None or (path.suffix == '.h5' and current.suffix != '.h5'):
            selected[name] = path
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path.cwd())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--burn-in', type=int, default=0)
    parser.add_argument('--thin', type=int, default=1)
    parser.add_argument('--corner-max-samples', type=int, default=50000)
    parser.add_argument('--corner-range-fraction', type=float, default=0.997)
    parser.add_argument('--walkers-max', type=int)
    parser.add_argument('--log-parameter', action='append', default=None)
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--include-broken', action='store_true')
    arguments = parser.parse_args()

    output_root = (arguments.output_dir
                   if arguments.output_dir is not None
                   else arguments.directory / 'all_mcmc_plots')
    runs = discover_saved_runs(arguments.directory, arguments.include_broken)
    if not runs:
        raise FileNotFoundError(
            f'No pyGrater backend or NPZ chains found in {arguments.directory}.')

    failures = []
    for name, path in runs.items():
        print(f'Plotting {name}: {path.name}')
        try:
            results = load_saved_mcmc(path)
            save_generic_plots(
                results, output_root / name,
                burn_in=arguments.burn_in, thin=arguments.thin,
                corner_max_samples=arguments.corner_max_samples,
                walkers_max=arguments.walkers_max, dpi=arguments.dpi,
                corner_range_fraction=arguments.corner_range_fraction,
                log_parameter_names=arguments.log_parameter)
        except Exception as error:
            failures.append((name, error))
            print(f'  FAILED: {error}')
        else:
            print(f'  Saved in {output_root / name}')

    if failures:
        failed_names = ', '.join(name for name, _ in failures)
        raise RuntimeError(f'Could not plot: {failed_names}')
    print(f'All plots saved under {output_root.resolve()}')


if __name__ == '__main__':
    main()
