"""Plot and inspect completed or checkpointed dynesty runs."""

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.utils.dynesty_backend import normalized_weights


@dataclass
class NestedResults:
    samples: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    param_names: list
    log_params: set
    log_evidence: float
    log_evidence_error: float
    n_likelihood_calls: Optional[int] = None
    source: Optional[Path] = None


def from_fitter(fitter):
    """Build a plotting view from any pyGraterFit nested fitter."""
    if fitter.samples is None:
        raise RuntimeError("Run or load nested-sampling results first.")
    return NestedResults(
        samples=np.asarray(fitter.samples),
        weights=np.asarray(fitter.weights),
        log_likelihood=np.asarray(fitter.log_likelihood_values),
        param_names=list(fitter.param_names),
        log_params=set(fitter.log_params),
        log_evidence=float(fitter.log_evidence),
        log_evidence_error=float(fitter.log_evidence_error),
        n_likelihood_calls=int(fitter.n_likelihood_calls),
    )


def load_results(filename):
    """Load the portable NPZ written by a nested fitter."""
    filename = Path(filename)
    with np.load(filename, allow_pickle=False) as saved:
        return NestedResults(
            samples=np.asarray(saved["samples"], dtype=np.float64),
            weights=np.asarray(saved["weights"], dtype=np.float64),
            log_likelihood=np.asarray(
                saved["log_likelihood"], dtype=np.float64),
            param_names=list(saved["param_names"].astype(str)),
            log_params=set(saved["log_params"].astype(str)),
            log_evidence=float(saved["log_evidence"]),
            log_evidence_error=float(saved["log_evidence_error"]),
            n_likelihood_calls=(
                int(saved["n_likelihood_calls"])
                if "n_likelihood_calls" in saved else None),
            source=filename,
        )


def load_checkpoint(filename, param_names=None, log_params=()):
    """Load live or completed results directly from a dynesty checkpoint."""
    import dynesty

    filename = Path(filename)
    errors = []
    sampler = None
    for sampler_class in (
            dynesty.DynamicNestedSampler, dynesty.NestedSampler):
        try:
            sampler = sampler_class.restore(str(filename))
            break
        except Exception as exc:
            errors.append(exc)
    if sampler is None:
        raise RuntimeError(
            f"Could not restore dynesty checkpoint {filename}: {errors[-1]}")
    results = sampler.results
    samples = np.asarray(results.samples, dtype=np.float64)
    if param_names is None:
        param_names = [f"parameter_{index}" for index in range(samples.shape[1])]
    if len(param_names) != samples.shape[1]:
        raise ValueError("param_names length does not match checkpoint ndim.")
    calls = np.asarray(results.ncall)
    return NestedResults(
        samples=samples,
        weights=normalized_weights(results),
        log_likelihood=np.asarray(results.logl, dtype=np.float64),
        param_names=list(param_names),
        log_params=set(log_params),
        log_evidence=float(np.asarray(results.logz)[-1]),
        log_evidence_error=float(np.asarray(results.logzerr)[-1]),
        n_likelihood_calls=int(calls.sum()) if calls.ndim else int(calls),
        source=filename,
    )


def plot_trace(results):
    """Plot parameter values and cumulative posterior weight versus sample."""
    count = len(results.param_names)
    figure, axes = plt.subplots(
        count, 1, figsize=(10, max(2.2 * count, 4)), squeeze=False)
    cumulative_weight = np.cumsum(results.weights)
    for index, (name, axis) in enumerate(
            zip(results.param_names, axes[:, 0])):
        values = results.samples[:, index]
        if name in results.log_params:
            values = np.log10(values)
            label = f"log10({name})"
        else:
            label = name
        points = axis.scatter(
            np.arange(values.size), values, c=cumulative_weight,
            s=4, cmap="viridis", rasterized=True)
        axis.set_ylabel(label)
    axes[-1, 0].set_xlabel("Saved nested sample")
    figure.colorbar(
        points, ax=axes[:, 0].tolist(), label="Cumulative posterior weight")
    figure.suptitle("Nested-sampling parameter trace")
    return figure


def plot_likelihood(results):
    """Plot likelihood progression and posterior weights."""
    index = np.arange(results.log_likelihood.size)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(index, results.log_likelihood, linewidth=0.8)
    axes[0].set_ylabel("log likelihood")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(index, np.maximum(results.weights, np.finfo(float).tiny))
    axes[1].set(xlabel="Saved nested sample", ylabel="Posterior weight")
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        f"log(Z) = {results.log_evidence:.6g} "
        f"+/- {results.log_evidence_error:.3g}")
    figure.tight_layout()
    return figure


def plot_corner(results, max_samples=50000, seed=8, **kwargs):
    """Create the same corner style used by the fitter classes."""
    rng = np.random.default_rng(seed)
    count = min(int(max_samples), len(results.samples))
    indices = rng.choice(
        len(results.samples), size=count, replace=True, p=results.weights)
    return make_corner_plot(
        results.samples[indices], results.param_names,
        results.log_params, **kwargs)


def plot_nested_results(results, output_directory, prefix="nested",
                        max_corner_samples=50000, seed=8):
    """Write the standard nested-sampling diagnostic plot bundle."""
    if not isinstance(results, NestedResults):
        results = from_fitter(results)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = {
        f"{prefix}_trace.png": plot_trace(results),
        f"{prefix}_likelihood_weights.png": plot_likelihood(results),
        f"{prefix}_corner.png": plot_corner(
            results, max_samples=max_corner_samples, seed=seed),
    }
    for name, figure in figures.items():
        figure.savefig(output_directory / name, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return [output_directory / name for name in figures]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot pyGraterFit nested-sampling results.")
    parser.add_argument("results", help="Portable nested-results NPZ file")
    parser.add_argument("-o", "--output-directory", default="nested_plots")
    parser.add_argument("--prefix", default="nested")
    args = parser.parse_args(argv)
    paths = plot_nested_results(
        load_results(args.results), args.output_directory, args.prefix)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
