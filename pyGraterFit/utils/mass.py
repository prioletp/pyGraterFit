"""Shared dust-mass reporting helpers for fitters.

The physical mass calculation belongs to pyGrater's SED object.  These helpers
standardize how fitters choose the parameter point, complete fixed/dependent
parameters, call ``SED.get_total_mass``, and report results.
"""

from __future__ import annotations


def total_mass_from_sed(sed_object, parameters):
    """Return the dust mass in Earth masses for one completed component.

    Parameters
    ----------
    sed_object
        A pyGrater ``SED`` or ``CachedSED``-like object with
        ``get_total_mass(**parameters)``.
    parameters : dict
        Fully completed physical model parameters for one component.

    Returns
    -------
    float
        Surviving dust mass in Earth masses.  If ``M_tot`` is present,
        pyGrater returns it directly; otherwise it converts ``A_norm`` to mass
        using the same density, grain-size, and sublimation treatment as the
        forward model.
    """
    return float(sed_object.get_total_mass(**parameters))


def single_component_total_mass(
        sed_object, values, complete_parameters, *, missing_message=None):
    """Return the best-point mass for a single-component fitter.

    ``values`` should be the fitter's current best free-parameter dictionary.
    ``complete_parameters`` is typically the fitter's ``_complete_parameters``
    method.  Keeping this logic in one place prevents examples from reaching
    into ``fitter.sed_obj`` directly.
    """
    if values is None:
        raise RuntimeError(
            missing_message or 'No fitted parameters are available.')
    return total_mass_from_sed(sed_object, complete_parameters(values))


def format_total_mass(mass_earth, label='Total dust mass'):
    """Return a one-line human-readable mass summary."""
    return f'{label}: {float(mass_earth):.8g} Earth masses'
