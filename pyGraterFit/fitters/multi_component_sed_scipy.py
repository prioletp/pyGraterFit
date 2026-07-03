"""SciPy-only additive SED fitter.

This is the deterministic optimization interface for sums of pyGrater SED
components.  It intentionally reuses the optimized component/cache machinery
from :mod:`multi_component_sed_mcmc` so the objective function and numerical
results are identical to calling ``AdditiveSEDMCMCFitter.fit(...)`` without
running MCMC.
"""

from pyGraterFit.fitters.multi_component_sed_mcmc import (
    AdditiveSEDMCMCFitter,
)


class AdditiveSEDScipyFitter(AdditiveSEDMCMCFitter):
    """Fit multiple additive pyGrater SED components with SciPy optimizers.

    The class supports the same component dictionaries, ring/material compact
    constructor, shared parameters, grouped total-normalization, spatial-cache
    sharing, and optional parallel component evaluation as the additive MCMC
    fitter.  It only exposes the deterministic SciPy workflow as the intended
    public use case.

    Common usage::

        fitter = AdditiveSEDScipyFitter(materials=..., ring_params=...)
        result = fitter.fit(maxiter=1000)
        fitter.summary()

    Available ``method`` values are the same as the base class:
    ``'Nelder-Mead'``, ``'L-BFGS-B'``, ``'Powell'``,
    ``'differential_evolution'``, and ``'dual_annealing'``.
    """

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        """Run the configured SciPy optimizer and return its result object."""
        return super().fit(
            initial_guess=initial_guess, maxiter=maxiter, verbose=verbose)


MultiComponentSEDScipyFitter = AdditiveSEDScipyFitter


__all__ = [
    'AdditiveSEDScipyFitter',
    'MultiComponentSEDScipyFitter',
]
