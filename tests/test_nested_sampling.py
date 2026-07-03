"""Fast backend tests shared by all dynesty-based fitters."""

import numpy as np

from pyGraterFit.utils.dynesty_backend import (
    finite_log_likelihood, normalized_weights, resample_equal, run_dynesty)
from pyGraterFit.utils.nested_plotting import (
    NestedResults, load_results, plot_nested_results)


def _log_likelihood(values):
    return -0.5 * np.dot(values, values)


def _prior_transform(unit_cube):
    return -5.0 + 10.0 * unit_cube


def test_static_dynesty_results_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "static.save"
    _, results, weights, diagnostics = run_dynesty(
        _log_likelihood, _prior_transform, 2,
        npoints=30, dynamic=False, dlogz=5.0, maxcall=160,
        progress=False, checkpoint_file=checkpoint,
        checkpoint_every=0.01)
    assert checkpoint.exists()
    np.testing.assert_allclose(weights.sum(), 1.0)
    np.testing.assert_allclose(normalized_weights(results), weights)
    assert diagnostics["engine"] == "dynesty"
    assert diagnostics["n_likelihood_calls"] > 0


def test_dynamic_dynesty_and_equal_weight_samples():
    _, results, weights, diagnostics = run_dynesty(
        _log_likelihood, _prior_transform, 2,
        npoints=30, dynamic=True, dlogz=5.0, maxcall=160,
        progress=False)
    equal = resample_equal(results.samples, weights, seed=2)
    assert equal.shape == results.samples.shape
    assert diagnostics["dynamic"]


def test_dynesty_backend_never_passes_nonfinite_loglikelihood():
    wrapped = finite_log_likelihood(lambda values: -np.inf)
    assert np.isfinite(wrapped(np.zeros(2)))

    _, results, weights, _ = run_dynesty(
        lambda values: -np.inf, _prior_transform, 2,
        npoints=30, dynamic=False, dlogz=5.0, maxcall=120,
        progress=False)
    assert np.all(np.isfinite(results.logl))
    assert np.isfinite(results.logz[-1])
    assert np.isfinite(results.logzerr[-1])
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_nested_result_plot_bundle_and_reload(tmp_path):
    results_path = tmp_path / "results.npz"
    samples = np.column_stack((
        np.linspace(1.0, 2.0, 30),
        np.linspace(-1.0, 1.0, 30)))
    weights = np.full(30, 1.0 / 30.0)
    np.savez_compressed(
        results_path,
        samples=samples,
        weights=weights,
        log_likelihood=np.linspace(-20.0, -1.0, 30),
        param_names=np.array(["A_norm", "slope"]),
        log_params=np.array(["A_norm"]),
        log_evidence=-3.0,
        log_evidence_error=0.2,
        n_likelihood_calls=100,
    )
    loaded = load_results(results_path)
    assert isinstance(loaded, NestedResults)
    paths = plot_nested_results(
        loaded, tmp_path / "plots", prefix="test", max_corner_samples=50)
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
