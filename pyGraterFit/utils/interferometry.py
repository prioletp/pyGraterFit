"""Interferometric observables from pyGrater model images.

Sky and Fourier conventions
---------------------------
``Image`` images have North at row 0 and East toward the left, so array
columns increase toward West.
Position angle is measured in degrees east of North: PA=0 places the disk
major axis North-South and PA=90 places it East-West.

OIFITS uses ``u`` toward East and ``v`` toward North.  NumPy array rows
increase from North toward South, so an OIFITS sample ``(u, v)`` is evaluated
at FFT coordinates ``(-u / wavelength, -v / wavelength)``.  The complex
Fourier transform uses ``exp[-2 pi i (u East + v North)]``.
"""

from pathlib import Path
import time

import numpy as np
from astropy.io import fits
from scipy.special import j1


ARCSECONDS_PER_RADIAN = 206264.80624709636
MILLIARCSECONDS_PER_RADIAN = ARCSECONDS_PER_RADIAN * 1000.0


def uniform_disk_argument_per_mas(u_m, v_m, wavelengths_m):
    """Return the invariant uniform-disk Bessel argument per mas."""
    u_m, v_m, wavelengths_m = np.broadcast_arrays(
        np.asarray(u_m, dtype=np.float64),
        np.asarray(v_m, dtype=np.float64),
        np.asarray(wavelengths_m, dtype=np.float64))
    if np.any(wavelengths_m <= 0):
        raise ValueError('Interferometric wavelengths must be positive.')
    return (np.pi * np.hypot(u_m, v_m)
            / (wavelengths_m * MILLIARCSECONDS_PER_RADIAN))


def uniform_disk_visibility_from_argument(argument):
    """Evaluate ``2 J1(x) / x`` while preserving its exact x=0 limit."""
    argument = np.asarray(argument, dtype=np.float64)
    visibility = np.ones_like(argument)
    np.divide(2.0 * j1(argument), argument, out=visibility,
              where=argument != 0.0)
    return visibility


def uniform_disk_visibility(u_m, v_m, wavelengths_m,
                            angular_diameter_mas):
    """Return the analytic complex visibility of a centered uniform disk.

    ``angular_diameter_mas`` is the full angular diameter. The result is real,
    but it is not forced positive beyond visibility nulls because its sign is
    physically required when it is combined with another complex component.
    """
    diameter_mas = np.asarray(angular_diameter_mas, dtype=np.float64)
    if np.any(~np.isfinite(diameter_mas)) or np.any(diameter_mas < 0):
        raise ValueError('Uniform-disk angular diameter must be non-negative.')
    argument = (uniform_disk_argument_per_mas(u_m, v_m, wavelengths_m)
                * diameter_mas)
    return uniform_disk_visibility_from_argument(argument)


def wrap_phase_degrees(angle_degrees):
    """Wrap an angle or phase residual to the interval [-180, 180)."""
    return (np.asarray(angle_degrees) + 180.0) % 360.0 - 180.0


def pixel_scale_au_to_radian(pixel_scale_au, distance_pc):
    """Convert a physical pixel scale at ``distance_pc`` to radians."""
    return float(pixel_scale_au) / (
        float(distance_pc) * ARCSECONDS_PER_RADIAN)


def _next_fast_padded_size(image_size, padding_factor):
    requested_size = int(np.ceil(image_size * padding_factor))
    return int(2 ** np.ceil(np.log2(max(requested_size, image_size))))


def _center_image_in_array(image, output_shape):
    """Place an image at the geometric center of a larger zero-filled array."""
    output_rows, output_columns = output_shape
    image_rows, image_columns = image.shape
    row_start = (output_rows - image_rows) // 2
    column_start = (output_columns - image_columns) // 2
    padded_image = np.zeros(output_shape, dtype=np.float64)
    padded_image[
        row_start:row_start + image_rows,
        column_start:column_start + image_columns] = image
    return padded_image


def _centered_complex_fourier_transform(image, pixel_scale_radian,
                                        padding_factor):
    """Return a centered complex FFT and its West/South frequency axes."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError('A model image must be a two-dimensional array.')

    n_rows = _next_fast_padded_size(image.shape[0], padding_factor)
    n_columns = _next_fast_padded_size(image.shape[1], padding_factor)
    padded_image = _center_image_in_array(image, (n_rows, n_columns))

    fourier_transform = np.fft.fftshift(np.fft.fft2(padded_image))
    south_frequency = np.fft.fftshift(
        np.fft.fftfreq(n_rows, d=pixel_scale_radian))
    west_frequency = np.fft.fftshift(
        np.fft.fftfreq(n_columns, d=pixel_scale_radian))

    # np.fft places the coordinate origin at array index zero.  Remove the
    # resulting linear phase so the origin is the geometric image center,
    # including for even-sized images whose center lies between four pixels.
    # The embedded image center can differ by half a pixel from the padded
    # array center when their dimensions have different parity.
    center_row = ((n_rows - image.shape[0]) // 2
                  + (image.shape[0] - 1.0) / 2.0)
    center_column = ((n_columns - image.shape[1]) // 2
                     + (image.shape[1] - 1.0) / 2.0)
    row_phase = np.exp(
        2j * np.pi * south_frequency * center_row * pixel_scale_radian)
    column_phase = np.exp(
        2j * np.pi * west_frequency * center_column * pixel_scale_radian)
    fourier_transform *= row_phase[:, None] * column_phase[None, :]
    return fourier_transform, south_frequency, west_frequency


def _bilinear_complex_samples(values, row_grid, column_grid,
                              row_coordinates, column_coordinates):
    """Bilinearly sample a complex array, returning zero outside its grid."""
    row_coordinates = np.asarray(row_coordinates, dtype=np.float64)
    column_coordinates = np.asarray(column_coordinates, dtype=np.float64)
    if row_coordinates.shape != column_coordinates.shape:
        raise ValueError('Fourier row and column coordinates must match.')

    row_position = (row_coordinates - row_grid[0]) / (
        row_grid[1] - row_grid[0])
    column_position = (column_coordinates - column_grid[0]) / (
        column_grid[1] - column_grid[0])
    row_lower = np.floor(row_position).astype(np.int64)
    column_lower = np.floor(column_position).astype(np.int64)
    inside = (
        (row_lower >= 0) & (row_lower < len(row_grid) - 1)
        & (column_lower >= 0) & (column_lower < len(column_grid) - 1))

    samples = np.zeros(row_coordinates.shape, dtype=np.complex128)
    if not np.any(inside):
        return samples

    rows = row_lower[inside]
    columns = column_lower[inside]
    row_fraction = row_position[inside] - rows
    column_fraction = column_position[inside] - columns
    samples[inside] = (
        values[rows, columns]
        * (1.0 - row_fraction) * (1.0 - column_fraction)
        + values[rows + 1, columns]
        * row_fraction * (1.0 - column_fraction)
        + values[rows, columns + 1]
        * (1.0 - row_fraction) * column_fraction
        + values[rows + 1, columns + 1]
        * row_fraction * column_fraction)
    return samples


def complex_visibilities_from_image(
        image, pixel_scale_au, distance_pc, u_m, v_m, wavelengths_m,
        unresolved_flux_jy=0.0, padding_factor=4,
        stellar_angular_diameter_mas=0.0):
    """Sample normalized complex visibility at OIFITS baselines.

    ``image`` contains resolved-disk flux per pixel.  ``unresolved_flux_jy``
    optionally adds a star at the image center before normalization.
    """
    u_m, v_m, wavelengths_m = np.broadcast_arrays(
        np.asarray(u_m, dtype=np.float64),
        np.asarray(v_m, dtype=np.float64),
        np.asarray(wavelengths_m, dtype=np.float64))
    if np.any(wavelengths_m <= 0):
        raise ValueError('Interferometric wavelengths must be positive.')

    disk_flux_jy = float(np.sum(image, dtype=np.float64))
    stellar_flux_jy = np.broadcast_to(
        np.asarray(unresolved_flux_jy, dtype=np.float64), u_m.shape)
    total_flux_jy = disk_flux_jy + stellar_flux_jy
    if np.any(total_flux_jy <= 0):
        raise ValueError('Disk plus unresolved flux must be positive.')

    pixel_scale_radian = pixel_scale_au_to_radian(
        pixel_scale_au, distance_pc)
    transform, south_frequency, west_frequency = (
        _centered_complex_fourier_transform(
            image, pixel_scale_radian, padding_factor))
    return _sample_normalized_visibility(
        transform, south_frequency, west_frequency,
        disk_flux_jy, stellar_flux_jy, u_m, v_m, wavelengths_m,
        stellar_angular_diameter_mas)


def _sample_normalized_visibility(
        transform, south_frequency, west_frequency, disk_flux_jy,
        stellar_flux_jy, u_m, v_m, wavelengths_m,
        stellar_angular_diameter_mas=0.0, stellar_visibility=None):
    """Sample a disk transform and add a point or uniform-disk star."""
    u_m, v_m, wavelengths_m = np.broadcast_arrays(
        np.asarray(u_m, dtype=np.float64),
        np.asarray(v_m, dtype=np.float64),
        np.asarray(wavelengths_m, dtype=np.float64))
    stellar_flux_jy = np.broadcast_to(
        np.asarray(stellar_flux_jy, dtype=np.float64), u_m.shape)
    total_flux_jy = disk_flux_jy + stellar_flux_jy
    disk_transform = _bilinear_complex_samples(
        transform, south_frequency, west_frequency,
        -v_m / wavelengths_m, -u_m / wavelengths_m)
    if stellar_visibility is None:
        stellar_visibility = uniform_disk_visibility(
            u_m, v_m, wavelengths_m, stellar_angular_diameter_mas)
    else:
        stellar_visibility = np.broadcast_to(
            np.asarray(stellar_visibility, dtype=np.float64), u_m.shape)
    return (disk_transform
            + stellar_flux_jy * stellar_visibility) / total_flux_jy


def observables_from_image(
        image, pixel_scale_au, distance_pc,
        vis2_u_m=None, vis2_v_m=None, vis2_wavelength_m=None,
        closure_u1_m=None, closure_v1_m=None,
        closure_u2_m=None, closure_v2_m=None,
        closure_wavelength_m=None, unresolved_flux_jy=0.0,
        closure_unresolved_flux_jy=None,
        padding_factor=4,
        stellar_angular_diameter_mas=0.0,
        vis2_stellar_visibility=None):
    """Return squared visibilities and closure phases from one model image."""
    start = time.perf_counter()
    disk_flux_jy = float(np.sum(image, dtype=np.float64))
    if closure_unresolved_flux_jy is None:
        closure_unresolved_flux_jy = unresolved_flux_jy
    if disk_flux_jy + np.min(np.asarray(unresolved_flux_jy)) <= 0:
        raise ValueError('Disk plus unresolved flux must be positive.')
    transform, south_frequency, west_frequency = (
        _centered_complex_fourier_transform(
            image, pixel_scale_au_to_radian(pixel_scale_au, distance_pc),
            padding_factor))

    def sample(u_m, v_m, wavelength_m, central_flux_jy,
               central_visibility=None):
        return _sample_normalized_visibility(
            transform, south_frequency, west_frequency, disk_flux_jy,
            central_flux_jy, u_m, v_m, wavelength_m,
            stellar_angular_diameter_mas, central_visibility)

    squared_visibility = None
    if vis2_u_m is not None:
        complex_visibility = sample(
            vis2_u_m, vis2_v_m, vis2_wavelength_m,
            unresolved_flux_jy, vis2_stellar_visibility)
        squared_visibility = np.abs(complex_visibility)**2

    closure_phase_degrees = None
    if closure_u1_m is not None:
        closure_u3_m = -(np.asarray(closure_u1_m)
                         + np.asarray(closure_u2_m))
        closure_v3_m = -(np.asarray(closure_v1_m)
                         + np.asarray(closure_v2_m))
        visibility_1 = sample(
            closure_u1_m, closure_v1_m, closure_wavelength_m,
            closure_unresolved_flux_jy)
        visibility_2 = sample(
            closure_u2_m, closure_v2_m, closure_wavelength_m,
            closure_unresolved_flux_jy)
        visibility_3 = sample(
            closure_u3_m, closure_v3_m, closure_wavelength_m,
            closure_unresolved_flux_jy)
        closure_phase_degrees = np.degrees(np.angle(
            visibility_1 * visibility_2 * visibility_3))

    return squared_visibility, closure_phase_degrees, {
        'total': time.perf_counter() - start}


def _wavelength_table_by_instrument(hdul):
    tables = {}
    for extension in hdul:
        if extension.header.get('EXTNAME') == 'OI_WAVELENGTH':
            instrument = extension.header.get('INSNAME')
            tables[instrument] = np.asarray(
                extension.data['EFF_WAVE'], dtype=np.float64)
    return tables


def _flatten_oifits_channels(table, wavelengths, value_names):
    """Flatten row-by-channel OIFITS columns while removing flagged values."""
    n_rows = len(table)
    n_channels = len(wavelengths)
    wavelength_grid = np.broadcast_to(wavelengths, (n_rows, n_channels))
    flag = (np.asarray(table['FLAG'], dtype=bool)
            if 'FLAG' in table.names else np.zeros_like(wavelength_grid, bool))
    output = {'wavelength_m': wavelength_grid[~flag]}
    for output_name, column_name in value_names.items():
        values = np.asarray(table[column_name])
        if values.ndim == 1:
            values = np.broadcast_to(values[:, None], wavelength_grid.shape)
        output[output_name] = values[~flag].astype(np.float64)
    return output


def load_oifits_observations(paths):
    """Load the OI_VIS2 and OI_T3 observables needed by the fitters."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    vis2_parts = []
    closure_parts = []
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            wavelength_tables = _wavelength_table_by_instrument(hdul)
            for extension in hdul:
                extension_name = extension.header.get('EXTNAME')
                instrument = extension.header.get('INSNAME')
                wavelengths = wavelength_tables.get(instrument)
                if wavelengths is None or extension.data is None:
                    continue
                if extension_name == 'OI_VIS2':
                    vis2_parts.append(_flatten_oifits_channels(
                        extension.data, wavelengths, {
                            'value': 'VIS2DATA', 'error': 'VIS2ERR',
                            'u_m': 'UCOORD', 'v_m': 'VCOORD'}))
                elif extension_name == 'OI_T3':
                    closure_parts.append(_flatten_oifits_channels(
                        extension.data, wavelengths, {
                            'value_degrees': 'T3PHI',
                            'error_degrees': 'T3PHIERR',
                            'u1_m': 'U1COORD', 'v1_m': 'V1COORD',
                            'u2_m': 'U2COORD', 'v2_m': 'V2COORD'}))

    def concatenate(parts):
        if not parts:
            return None
        return {key: np.concatenate([part[key] for part in parts])
                for key in parts[0]}

    return {'vis2': concatenate(vis2_parts),
            'closure_phase': concatenate(closure_parts)}
