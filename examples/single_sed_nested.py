"""Single-component SED fit with dynesty nested sampling.

Nested sampling estimates both posterior samples and the Bayesian evidence
`logz`. It is useful when you care about model comparison or want a sampler
that can handle multimodal posteriors.
"""

from pyGraterFit import SEDNestedFitter


# ---------------------------------------------------------------------------
# 1. Build the fitter
# ---------------------------------------------------------------------------

# Replace the ellipsis with the same model/data arguments used in the SciPy
# example. The nested fitter uses the same pyGrater physics, but dynesty
# explores the prior volume instead of starting from a local optimizer.
fitter = SEDNestedFitter(...)


# ---------------------------------------------------------------------------
# 2. Nested-sampling controls
# ---------------------------------------------------------------------------

# "fresh": start a new dynesty run.
# "resume": continue from the checkpoint.
# "load": skip sampling and read the saved portable .npz results.
RUN_MODE = "fresh"

# dynesty live points. More live points are safer but more expensive.
N_LIVE_POINTS = 300

# Evidence stopping criterion. Use larger values for quick tests and smaller
# values for production evidence estimates.
DLOGZ = 0.1

# Bounding method. "multi" is a robust default for curved/multimodal problems.
METHOD = "multi"

# Proposal method. "rslice" is random-direction slice sampling and is usually
# robust for moderately high-dimensional correlated parameters.
SAMPLE = "rslice"

# Dynamic nested sampling adaptively allocates live points. Set False for
# simpler static nested sampling during debugging.
DYNAMIC = True

CHECKPOINT = "single_sed_nested.checkpoint"
RESULTS = "single_sed_nested_results.npz"


# ---------------------------------------------------------------------------
# 3. Run, resume, or load
# ---------------------------------------------------------------------------

if RUN_MODE == "load":
    fitter.load_results(RESULTS)
elif RUN_MODE == "resume":
    fitter.resume_backend_nested(
        CHECKPOINT,
        npoints=N_LIVE_POINTS,
        method=METHOD,
        sample=SAMPLE,
        dynamic=DYNAMIC,
        dlogz=DLOGZ,
        checkpoint_every=300,
    )
    fitter.save_results(RESULTS)
else:
    fitter.run(
        npoints=N_LIVE_POINTS,
        method=METHOD,
        sample=SAMPLE,
        dynamic=DYNAMIC,
        dlogz=DLOGZ,
        checkpoint_file=CHECKPOINT,
        checkpoint_every=300,
        resume=False,
    )
    fitter.save_results(RESULTS)


# ---------------------------------------------------------------------------
# 4. Diagnostics and plots
# ---------------------------------------------------------------------------

fitter.summary()
fitter.plot_nested_diagnostics(
    "single_sed_nested_plots",
    prefix="single_sed_nested",
)
fitter.plot_best_fit().savefig("single_sed_nested_best_fit.png", dpi=150)
fitter.corner_plot(max_samples=5000).savefig(
    "single_sed_nested_corner.png", dpi=150)
