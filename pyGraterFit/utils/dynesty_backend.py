"""Shared dynesty runner and result utilities for every nested fitter."""

from pathlib import Path
import time

import numpy as np

DEFAULT_LOG_LIKELIHOOD_FLOOR = -1.0e6


def _dynesty():
    try:
        import dynesty
        from dynesty import utils as dyfunc
    except ImportError as exc:
        raise ImportError(
            "Nested sampling requires dynesty. Install "
            "pyGraterFit with its current dependencies."
        ) from exc
    return dynesty, dyfunc


def normalized_weights(results):
    """Return normalized posterior weights from dynesty results."""
    if hasattr(results, "importance_weights"):
        weights = np.asarray(results.importance_weights(), dtype=np.float64)
        total = weights.sum()
        if np.isfinite(total) and total > 0 and np.all(np.isfinite(weights)):
            return weights / total

    log_weights = np.asarray(results.logwt, dtype=np.float64)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("dynesty returned no finite posterior weights.")
    log_norm = np.logaddexp.reduce(log_weights[finite])
    weights = np.zeros_like(log_weights)
    weights[finite] = np.exp(log_weights[finite] - log_norm)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("dynesty returned invalid posterior weights.")
    return weights / total


class FiniteLogLikelihood:
    """Pickle-safe likelihood adapter that never returns NaN or +/-inf."""

    def __init__(self, log_likelihood, floor=DEFAULT_LOG_LIKELIHOOD_FLOOR):
        self.log_likelihood = log_likelihood
        self.floor = float(floor)

    def __call__(self, values):
        try:
            value = float(self.log_likelihood(values))
        except Exception:
            return self.floor
        if not np.isfinite(value):
            return self.floor
        return max(value, self.floor)


def finite_log_likelihood(log_likelihood, floor=DEFAULT_LOG_LIKELIHOOD_FLOOR):
    """Wrap a likelihood so dynesty never receives NaN or +/-inf values."""
    return FiniteLogLikelihood(log_likelihood, floor=floor)


def likelihood_call_count(results):
    """Return dynesty's total likelihood-call count."""
    calls = np.asarray(results.ncall)
    return int(calls.sum()) if calls.ndim else int(calls)


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(1.0 / np.sum(weights * weights))


def resample_equal(samples, weights, seed=8):
    """Draw equal-weight posterior samples with a reproducible generator."""
    _, dyfunc = _dynesty()
    return dyfunc.resample_equal(
        np.asarray(samples), np.asarray(weights),
        rstate=np.random.default_rng(seed))


def run_dynesty(
        log_likelihood, prior_transform, ndim, *,
        npoints=500, bound="multi", sample="rslice", dynamic=True,
        dlogz=0.1, maxiter=None, maxcall=None, seed=8,
        checkpoint_file=None, checkpoint_every=300, resume=False,
        progress=True, update_interval=None, walks=None, slices=None,
        n_effective=None, maxbatch=None):
    """Run or resume static/dynamic dynesty with consistent diagnostics."""
    dynesty, _ = _dynesty()
    sampler_class = (
        dynesty.DynamicNestedSampler if dynamic else dynesty.NestedSampler)
    safe_log_likelihood = finite_log_likelihood(log_likelihood)
    checkpoint_path = (
        None if checkpoint_file is None else Path(checkpoint_file))
    restoring = bool(
        resume and checkpoint_path is not None and checkpoint_path.exists())

    if restoring:
        sampler = sampler_class.restore(str(checkpoint_path))
    else:
        sampler_kwargs = {
            "nlive": int(npoints),
            "bound": bound,
            "sample": sample,
            "rstate": np.random.default_rng(seed),
        }
        if update_interval is not None:
            sampler_kwargs["update_interval"] = update_interval
        if walks is not None:
            sampler_kwargs["walks"] = int(walks)
        if slices is not None:
            sampler_kwargs["slices"] = int(slices)
        sampler = sampler_class(
            safe_log_likelihood, prior_transform, int(ndim), **sampler_kwargs)

    run_kwargs = {
        "print_progress": bool(progress),
        "resume": restoring,
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        run_kwargs["checkpoint_file"] = str(checkpoint_path)
        run_kwargs["checkpoint_every"] = float(checkpoint_every)

    if dynamic:
        run_kwargs.update({
            "nlive_init": int(npoints),
            "dlogz_init": dlogz,
            "maxiter": maxiter,
            "maxcall": maxcall,
        })
        if n_effective is not None:
            run_kwargs["n_effective"] = int(n_effective)
        if maxbatch is not None:
            run_kwargs["maxbatch"] = int(maxbatch)
    else:
        run_kwargs.update({
            "dlogz": dlogz,
            "maxiter": maxiter,
            "maxcall": maxcall,
        })

    start = time.perf_counter()
    sampler.run_nested(**run_kwargs)
    elapsed = time.perf_counter() - start
    results = sampler.results
    weights = normalized_weights(results)
    calls = likelihood_call_count(results)
    diagnostics = {
        "engine": "dynesty",
        "dynamic": bool(dynamic),
        "bound": bound,
        "sample": sample,
        "elapsed_seconds": elapsed,
        "n_iterations": int(len(results.samples)),
        "n_likelihood_calls": calls,
        "calls_per_iteration": calls / max(len(results.samples), 1),
        "effective_sample_size": effective_sample_size(weights),
        "sampling_efficiency_percent": (
            100.0 * len(results.samples) / max(calls, 1)),
        "checkpoint_file": (
            None if checkpoint_path is None else str(checkpoint_path)),
        "resumed": restoring,
    }
    return sampler, results, weights, diagnostics
