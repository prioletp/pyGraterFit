"""Small shared helpers for pyGrater emcee backends."""

import h5py
import numpy as np


def write_parameter_names(backend_path, parameter_names, group_name='mcmc'):
    """Store parameter order as optional pyGrater metadata in an HDF backend."""
    with h5py.File(backend_path, 'a') as backend:
        backend[group_name].attrs['parameter_names'] = np.asarray(
            parameter_names, dtype=h5py.string_dtype(encoding='utf-8'))
