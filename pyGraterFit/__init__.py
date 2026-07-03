"""Optimized fitting tools for pyGrater models."""

from importlib import import_module


_PUBLIC_CLASSES = {
    "AdditiveSEDCorrelatedFluxFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_mcmc"),
    "AdditiveSEDCorrelatedFluxNestedFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_dynesty"),
    "AdditiveSEDMCMCFitter": (
        "pyGraterFit.fitters.multi_component_sed_mcmc"),
    "AdditiveSEDNestedFitter": (
        "pyGraterFit.fitters.multi_component_sed_dynesty"),
    "AdditiveSEDScipyFitter": (
        "pyGraterFit.fitters.multi_component_sed_scipy"),
    "MultiRingSEDScipyFitter": (
        "pyGraterFit.fitters"),
    "MultiRingSEDMCMCFitter": (
        "pyGraterFit.fitters"),
    "MultiRingSEDNestedFitter": (
        "pyGraterFit.fitters"),
    "MultiRingSEDCorrelatedFluxFitter": (
        "pyGraterFit.fitters"),
    "MultiRingSEDCorrelatedFluxNestedFitter": (
        "pyGraterFit.fitters"),
    "FOVAdditiveSEDMCMCFitter": (
        "pyGraterFit.fitters.multi_component_fov_sed_mcmc"),
    "MultiComponentSEDMCMCFitter": (
        "pyGraterFit.fitters"),
    "MultiComponentSEDDynestyFitter": (
        "pyGraterFit.fitters"),
    "MultiComponentSEDScipyFitter": (
        "pyGraterFit.fitters"),
    "MultiComponentSEDCorrelatedFluxFitter": (
        "pyGraterFit.fitters"),
    "MultiComponentSEDCorrelatedFluxDynestyFitter": (
        "pyGraterFit.fitters"),
    "SEDCorrelatedFluxNestedFitter": (
        "pyGraterFit.fitters.single_ring_sed_correlated_flux_dynesty"),
    "SEDFitter": "pyGraterFit.fitters.single_ring_sed_scipy",
    "SEDInterferometryFitter": (
        "pyGraterFit.fitters.single_ring_sed_interferometry_scipy"),
    "SEDInterferometryMCMCFitter": (
        "pyGraterFit.fitters.single_ring_sed_interferometry_mcmc"),
    "SEDMCMCFitter": "pyGraterFit.fitters.single_ring_sed_mcmc",
    "SEDNestedFitter": "pyGraterFit.fitters.single_ring_sed_dynesty",
    "SEDVisibilityMCMCFitter": (
        "pyGraterFit.fitters.multi_component_sed_visibility_mcmc"),
    "SingleRingMultiCompositionNestedFitter": (
        "pyGraterFit.fitters.single_ring_multi_composition_sed_dynesty"),
    "SingleRingSEDFitter": "pyGraterFit.fitters",
    "SingleRingSEDScipyFitter": "pyGraterFit.fitters",
    "SingleRingSEDMCMCFitter": "pyGraterFit.fitters",
    "SingleRingSEDNestedFitter": "pyGraterFit.fitters",
    "SingleRingSEDDynestyFitter": "pyGraterFit.fitters",
    "correlated_flux_from_vlti_loader": (
        "pyGraterFit.fitters.single_ring_sed_correlated_flux_dynesty"),
    "vis2_from_vlti_loader": (
        "pyGraterFit.fitters.multi_component_sed_visibility_mcmc"),
    "MILLIARCSECONDS_PER_RADIAN": "pyGraterFit.utils.interferometry",
}

__all__ = list(_PUBLIC_CLASSES)


def __getattr__(name):
    """Import a public fitter only when it is first requested."""
    try:
        module_name = _PUBLIC_CLASSES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
