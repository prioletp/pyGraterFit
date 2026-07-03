"""Recover a partially written emcee HDF backend without altering the source."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def _readable_attributes(group):
    attributes = {}
    unreadable = []
    for name in group.attrs:
        try:
            attributes[name] = group.attrs[name]
        except (OSError, RuntimeError):
            unreadable.append(name)
    return attributes, unreadable


def recover_backend(source_path, destination_path, group_name='mcmc'):
    """Copy all committed samples into a clean, appendable HDF backend.

    emcee updates the ``iteration`` attribute only after a complete sample has
    been stored. Dataset rows at or beyond that value may therefore have been
    allocated but only partly written and are intentionally discarded.
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError('Recovery destination must differ from the source.')
    if destination_path.exists():
        raise FileExistsError(destination_path)

    with h5py.File(source_path, 'r') as source:
        source_group = source[group_name]
        attributes, unreadable = _readable_attributes(source_group)
        committed_iterations = int(attributes['iteration'])
        nwalkers = int(attributes['nwalkers'])
        ndim = int(attributes['ndim'])
        expected_chain_shape = (committed_iterations, nwalkers, ndim)
        expected_log_probability_shape = (committed_iterations, nwalkers)
        if source_group['chain'].shape[0] < committed_iterations:
            raise ValueError('Chain is shorter than the committed iteration count.')
        if source_group['log_prob'].shape[0] < committed_iterations:
            raise ValueError(
                'Log-probability data are shorter than the committed count.')

        final_coordinates = source_group['chain'][committed_iterations - 1]
        final_log_probability = source_group['log_prob'][
            committed_iterations - 1]
        if not np.all(np.isfinite(final_coordinates)):
            raise ValueError('The final committed walker coordinates are invalid.')
        if np.all(final_coordinates == 0) and np.all(final_log_probability == 0):
            raise ValueError('The final committed row appears incomplete.')

        with h5py.File(destination_path, 'x') as destination:
            destination_group = destination.create_group(group_name)
            for name, value in attributes.items():
                destination_group.attrs[name] = value

            # NumPy RandomState stores the generator name separately from its
            # 624-word state. A failed variable-length string write commonly
            # damages only random_state_0 while leaving the numerical state.
            if 'random_state_0' in unreadable:
                required_state = {
                    f'random_state_{index}' for index in range(1, 5)}
                if not required_state.issubset(attributes):
                    raise ValueError(
                        'Random-state metadata is too incomplete to reconstruct.')
                destination_group.attrs['random_state_0'] = 'MT19937'
                unreadable.remove('random_state_0')
            if unreadable:
                raise ValueError(
                    f'Cannot reconstruct unreadable attributes: {unreadable}')

            chain = destination_group.create_dataset(
                'chain', shape=expected_chain_shape,
                maxshape=(None, nwalkers, ndim),
                dtype=source_group['chain'].dtype, chunks=True)
            log_probability = destination_group.create_dataset(
                'log_prob', shape=expected_log_probability_shape,
                maxshape=(None, nwalkers),
                dtype=source_group['log_prob'].dtype, chunks=True)
            chain[...] = source_group['chain'][:committed_iterations]
            log_probability[...] = source_group['log_prob'][:committed_iterations]
            destination_group.create_dataset(
                'accepted', data=source_group['accepted'][...])

            if 'blobs' in source_group:
                source_blobs = source_group['blobs']
                blob_shape = (committed_iterations,) + source_blobs.shape[1:]
                blobs = destination_group.create_dataset(
                    'blobs', shape=blob_shape,
                    maxshape=(None,) + source_blobs.shape[1:],
                    dtype=source_blobs.dtype, chunks=True)
                blobs[...] = source_blobs[:committed_iterations]

    return {
        'source': source_path,
        'destination': destination_path,
        'iterations': committed_iterations,
        'walkers': nwalkers,
        'dimensions': ndim,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--group', default='mcmc')
    arguments = parser.parse_args()
    result = recover_backend(arguments.source, arguments.destination,
                             arguments.group)
    print(f'Recovered {result["iterations"]} iterations, '
          f'{result["walkers"]} walkers, {result["dimensions"]} dimensions')
    print(f'Clean backend: {result["destination"]}')


if __name__ == '__main__':
    main()
