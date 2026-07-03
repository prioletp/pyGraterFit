"""Readable analytical visibility models for centered debris disks.

Conventions
-----------
``u_m`` is the projected baseline toward East and ``v_m`` toward North.
Position angle is measured east of North. Inclination is zero for a face-on
disk. Wavelengths are in metres and physical disk dimensions are in au.

The Gaussian ring implemented here is a geometrically thin circular ring
convolved with a circular Gaussian in the disk plane. Its normalized
visibility has the exact analytical form

    J0(2 pi angular_radius spatial_frequency)
    * exp(-2 pi^2 angular_sigma^2 spatial_frequency^2).

The visibility is real for a centered, point-symmetric source. Its sign is
retained because negative visibility lobes can cancel the stellar coherent
flux before the observed correlated-flux amplitude is taken.
"""

import numpy as np
from scipy.special import j0

from pyGraterFit.utils.interferometry import ARCSECONDS_PER_RADIAN


def au_to_radian(distance_au, distance_pc):
    """Convert a physical dimension in au to an angular dimension in radians."""
    distance_pc = float(distance_pc)
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError('Stellar distance must be finite and positive.')
    return np.asarray(distance_au, dtype=np.float64) / (
        distance_pc * ARCSECONDS_PER_RADIAN)


def projected_disk_spatial_frequency(
        u_m, v_m, wavelengths_m, inclination_degrees, position_angle_degrees):
    """Return disk-plane spatial frequency in cycles per radian.

    The baseline is rotated into axes parallel and perpendicular to the
    projected disk major axis. Projection foreshortens the disk minor axis by
    ``cos(inclination)``, so the corresponding Fourier coordinate is multiplied
    by that same factor.
    """
    u_m, v_m, wavelengths_m = np.broadcast_arrays(
        np.asarray(u_m, dtype=np.float64),
        np.asarray(v_m, dtype=np.float64),
        np.asarray(wavelengths_m, dtype=np.float64))
    if np.any(~np.isfinite(wavelengths_m)) or np.any(wavelengths_m <= 0):
        raise ValueError('Interferometric wavelengths must be finite and positive.')

    inclination = np.radians(float(inclination_degrees))
    if not np.isfinite(inclination) or inclination < 0 or inclination > np.pi / 2:
        raise ValueError('Disk inclination must be between 0 and 90 degrees.')
    position_angle = np.radians(float(position_angle_degrees))
    if not np.isfinite(position_angle):
        raise ValueError('Disk position angle must be finite.')

    # PA=0: major axis North-South, so v is the major-axis baseline.
    baseline_along_major_m = (
        u_m * np.sin(position_angle) + v_m * np.cos(position_angle))
    baseline_along_minor_m = (
        u_m * np.cos(position_angle) - v_m * np.sin(position_angle))
    projected_baseline_m = np.hypot(
        baseline_along_major_m,
        np.cos(inclination) * baseline_along_minor_m)
    return projected_baseline_m / wavelengths_m


def gaussian_ring_visibility(
        u_m, v_m, wavelengths_m, distance_pc, ring_radius_au,
        ring_fwhm_au, inclination_degrees=0.0, position_angle_degrees=0.0):
    """Return normalized visibility of a Gaussian-broadened thin ring.

    ``ring_radius_au`` is the radius of the thin ring before convolution.
    ``ring_fwhm_au`` is the FWHM of the convolving Gaussian in the disk plane;
    setting it to zero gives an infinitesimally thin ring.
    """
    radius_au = float(ring_radius_au)
    fwhm_au = float(ring_fwhm_au)
    if not np.isfinite(radius_au) or radius_au < 0:
        raise ValueError('Ring radius must be finite and non-negative.')
    if not np.isfinite(fwhm_au) or fwhm_au < 0:
        raise ValueError('Ring FWHM must be finite and non-negative.')

    spatial_frequency = projected_disk_spatial_frequency(
        u_m, v_m, wavelengths_m,
        inclination_degrees, position_angle_degrees)
    angular_radius_radian = au_to_radian(radius_au, distance_pc)
    angular_sigma_radian = (
        au_to_radian(fwhm_au, distance_pc) / np.sqrt(8.0 * np.log(2.0)))

    thin_ring_visibility = j0(
        2.0 * np.pi * angular_radius_radian * spatial_frequency)
    gaussian_visibility = np.exp(
        -2.0 * np.pi**2 * angular_sigma_radian**2
        * spatial_frequency**2)
    return thin_ring_visibility * gaussian_visibility


ANALYTICAL_DISK_VISIBILITY_MODELS = {
    'gaussian_ring': gaussian_ring_visibility,
}


def analytical_disk_visibility(model_name, **model_arguments):
    """Evaluate a named analytical model from the extensible model registry."""
    try:
        model = ANALYTICAL_DISK_VISIBILITY_MODELS[model_name]
    except KeyError as exc:
        available = ', '.join(sorted(ANALYTICAL_DISK_VISIBILITY_MODELS))
        raise ValueError(
            f'Unknown analytical visibility model {model_name!r}. '
            f'Available models: {available}.') from exc
    return model(**model_arguments)


def correlated_flux_from_components(
        disk_flux_jy, disk_visibility, stellar_flux_jy, stellar_visibility):
    """Return the observed correlated-flux amplitude in Jy.

    Coherent fluxes are added with their signs before taking the amplitude.
    This is required beyond a visibility null, where either centered component
    may have a phase of pi and therefore a negative real visibility.
    """
    disk_flux_jy, disk_visibility, stellar_flux_jy, stellar_visibility = (
        np.broadcast_arrays(
            np.asarray(disk_flux_jy, dtype=np.float64),
            np.asarray(disk_visibility, dtype=np.float64),
            np.asarray(stellar_flux_jy, dtype=np.float64),
            np.asarray(stellar_visibility, dtype=np.float64)))
    coherent_flux_jy = (
        disk_flux_jy * disk_visibility
        + stellar_flux_jy * stellar_visibility)
    return np.abs(coherent_flux_jy)
