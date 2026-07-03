"""Model-fitting implementations, organized by scope and algorithm.

The subpackage uses lazy public exports so an optional dependency required by
one fitter family cannot make unrelated fitter imports fail.
"""

from importlib import import_module


_PUBLIC_CLASSES = {
    "AdditiveSEDCorrelatedFluxFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_mcmc",
        "AdditiveSEDCorrelatedFluxFitter",
    ),
    "AdditiveSEDCorrelatedFluxNestedFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_dynesty",
        "AdditiveSEDCorrelatedFluxNestedFitter",
    ),
    "AdditiveSEDMCMCFitter": (
        "pyGraterFit.fitters.multi_component_sed_mcmc",
        "AdditiveSEDMCMCFitter",
    ),
    "AdditiveSEDNestedFitter": (
        "pyGraterFit.fitters.multi_component_sed_dynesty",
        "AdditiveSEDNestedFitter",
    ),
    "AdditiveSEDScipyFitter": (
        "pyGraterFit.fitters.multi_component_sed_scipy",
        "AdditiveSEDScipyFitter",
    ),
    "FOVAdditiveSEDMCMCFitter": (
        "pyGraterFit.fitters.multi_component_fov_sed_mcmc",
        "FOVAdditiveSEDMCMCFitter",
    ),
    "MultiComponentSEDCorrelatedFluxDynestyFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_dynesty",
        "AdditiveSEDCorrelatedFluxNestedFitter",
    ),
    "MultiComponentSEDCorrelatedFluxFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_mcmc",
        "AdditiveSEDCorrelatedFluxFitter",
    ),
    "MultiComponentSEDMCMCFitter": (
        "pyGraterFit.fitters.multi_component_sed_mcmc",
        "AdditiveSEDMCMCFitter",
    ),
    "MultiComponentSEDDynestyFitter": (
        "pyGraterFit.fitters.multi_component_sed_dynesty",
        "AdditiveSEDNestedFitter",
    ),
    "MultiComponentSEDScipyFitter": (
        "pyGraterFit.fitters.multi_component_sed_scipy",
        "AdditiveSEDScipyFitter",
    ),
    "MultiRingSEDCorrelatedFluxFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_mcmc",
        "AdditiveSEDCorrelatedFluxFitter",
    ),
    "MultiRingSEDCorrelatedFluxNestedFitter": (
        "pyGraterFit.fitters.multi_component_sed_correlated_flux_dynesty",
        "AdditiveSEDCorrelatedFluxNestedFitter",
    ),
    "MultiRingSEDMCMCFitter": (
        "pyGraterFit.fitters.multi_component_sed_mcmc",
        "AdditiveSEDMCMCFitter",
    ),
    "MultiRingSEDNestedFitter": (
        "pyGraterFit.fitters.multi_component_sed_dynesty",
        "AdditiveSEDNestedFitter",
    ),
    "MultiRingSEDScipyFitter": (
        "pyGraterFit.fitters.multi_component_sed_scipy",
        "AdditiveSEDScipyFitter",
    ),
    "SEDCorrelatedFluxNestedFitter": (
        "pyGraterFit.fitters.single_ring_sed_correlated_flux_dynesty",
        "SEDCorrelatedFluxNestedFitter",
    ),
    "SEDFitter": (
        "pyGraterFit.fitters.single_ring_sed_scipy",
        "SEDFitter",
    ),
    "SEDInterferometryFitter": (
        "pyGraterFit.fitters.single_ring_sed_interferometry_scipy",
        "SEDInterferometryFitter",
    ),
    "SEDInterferometryMCMCFitter": (
        "pyGraterFit.fitters.single_ring_sed_interferometry_mcmc",
        "SEDInterferometryMCMCFitter",
    ),
    "SEDMCMCFitter": (
        "pyGraterFit.fitters.single_ring_sed_mcmc",
        "SEDMCMCFitter",
    ),
    "SEDNestedFitter": (
        "pyGraterFit.fitters.single_ring_sed_dynesty",
        "SEDNestedFitter",
    ),
    "SEDVisibilityMCMCFitter": (
        "pyGraterFit.fitters.multi_component_sed_visibility_mcmc",
        "SEDVisibilityMCMCFitter",
    ),
    "SingleRingMultiCompositionNestedFitter": (
        "pyGraterFit.fitters.single_ring_multi_composition_sed_dynesty",
        "SingleRingMultiCompositionNestedFitter",
    ),
    "SingleRingSEDFitter": (
        "pyGraterFit.fitters.single_ring_sed_scipy",
        "SEDFitter",
    ),
    "SingleRingSEDScipyFitter": (
        "pyGraterFit.fitters.single_ring_sed_scipy",
        "SEDFitter",
    ),
    "SingleRingSEDMCMCFitter": (
        "pyGraterFit.fitters.single_ring_sed_mcmc",
        "SEDMCMCFitter",
    ),
    "SingleRingSEDNestedFitter": (
        "pyGraterFit.fitters.single_ring_sed_dynesty",
        "SEDNestedFitter",
    ),
    "SingleRingSEDDynestyFitter": (
        "pyGraterFit.fitters.single_ring_sed_dynesty",
        "SEDNestedFitter",
    ),
}

__all__ = list(_PUBLIC_CLASSES)


def __getattr__(name):
    try:
        module_name, attribute_name = _PUBLIC_CLASSES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
