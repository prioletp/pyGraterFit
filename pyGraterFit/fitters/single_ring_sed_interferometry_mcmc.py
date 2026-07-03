"""MCMC extension of the optimized SED and interferometry fitter."""

import numpy as np

from pyGraterFit.utils.corner_plotting import make_corner_plot
from pyGraterFit.fitters.single_ring_sed_interferometry_scipy import (
    SEDInterferometryFitter,
)
from pyGraterFit.utils.mcmc_backend import write_parameter_names


class SEDInterferometryMCMCFitter(SEDInterferometryFitter):
    """Combined SciPy and emcee fitter with restartable chains."""

    FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

    def __init__(self, *args, prior_ranges=None, best_fit_values=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_ranges = self._validate_prior_ranges(prior_ranges)
        self.input_best_fit_values = (
            None if best_fit_values is None
            else self._validate_best_fit_values(best_fit_values))
        self.mcmc_sampler = None
        self.mcmc_chain = None
        self.mcmc_log_probability = None
        self.flat_samples = None
        self.mcmc_best_parameters = None
        self.mcmc_best_chi_squared = None
        self.posterior_summary = None

    def _validate_prior_ranges(self, prior_ranges):
        if prior_ranges is None:
            prior_ranges = self.free_parameter_ranges
        cleaned = {}
        for name in self.parameter_names:
            if name not in prior_ranges:
                raise ValueError(f'Missing prior range for {name}.')
            low, high = map(float, prior_ranges[name])
            if not np.isfinite(low + high) or low >= high:
                raise ValueError(f'Invalid prior range for {name}.')
            cleaned[name] = (low, high)
        return cleaned

    def _validate_best_fit_values(self, values):
        missing = [name for name in self.parameter_names if name not in values]
        if missing:
            raise ValueError('Missing best-fit values: ' + ', '.join(missing))
        return {name: float(values[name]) for name in self.parameter_names}

    def parameter_vector_to_dictionary(self, vector):
        return {name: float(vector[index])
                for index, name in enumerate(self.parameter_names)}

    def dictionary_to_parameter_vector(self, parameters):
        return np.array([parameters[name] for name in self.parameter_names],
                        dtype=np.float64)

    def log_prior(self, parameter_vector):
        for index, name in enumerate(self.parameter_names):
            low, high = self.prior_ranges[name]
            value = parameter_vector[index]
            if not np.isfinite(value) or value < low or value > high:
                return -np.inf
        return 0.0

    def log_probability(self, parameter_vector):
        if not np.isfinite(self.log_prior(parameter_vector)):
            return -np.inf
        parameters = self.parameter_vector_to_dictionary(parameter_vector)
        try:
            chi_squared, _, _, _, _ = self.evaluate_physical_parameters(
                parameters)
        except Exception:
            return -np.inf
        return -0.5 * chi_squared if np.isfinite(chi_squared) else -np.inf

    @staticmethod
    def _truncated_normal(rng, center, low, high, sigma, count):
        values = np.empty(count, dtype=np.float64)
        filled = 0
        while filled < count:
            candidates = rng.normal(
                center, sigma, size=max(4 * (count - filled), 64))
            candidates = candidates[
                (candidates >= low) & (candidates <= high)]
            accepted = min(len(candidates), count - filled)
            values[filled:filled + accepted] = candidates[:accepted]
            filled += accepted
        return values

    def walkers_from_prior(self, n_walkers, seed=42):
        """Draw linear parameters uniformly and scale parameters log-uniformly."""
        rng = np.random.default_rng(seed)
        walkers = np.empty((n_walkers, self.n_dimensions), dtype=np.float64)
        for index, name in enumerate(self.parameter_names):
            low, high = self.prior_ranges[name]
            if name in self.log_space_parameters:
                if low <= 0:
                    raise ValueError(f'{name} requires a positive log-space prior.')
                walkers[:, index] = 10.0**rng.uniform(
                    np.log10(low), np.log10(high), n_walkers)
            else:
                walkers[:, index] = rng.uniform(low, high, n_walkers)
        return walkers

    def walkers_around_best_fit(self, best_fit_values, n_walkers,
                                fwhm_fraction=0.02, seed=42):
        """Draw distinct walkers around a best fit with truncated Gaussians."""
        center = self._validate_best_fit_values(best_fit_values)
        rng = np.random.default_rng(seed)
        walkers = np.empty((n_walkers, self.n_dimensions), dtype=np.float64)
        for index, name in enumerate(self.parameter_names):
            low, high = self.prior_ranges[name]
            value = center[name]
            if not low <= value <= high:
                raise ValueError(f'Best-fit {name} lies outside its prior.')
            if name in self.log_space_parameters:
                log_low, log_high, log_value = np.log10([low, high, value])
                sigma = (fwhm_fraction * (log_high - log_low)
                         * self.FWHM_TO_SIGMA)
                walkers[:, index] = 10.0**self._truncated_normal(
                    rng, log_value, log_low, log_high, sigma, n_walkers)
            else:
                sigma = (fwhm_fraction * (high - low)
                         * self.FWHM_TO_SIGMA)
                walkers[:, index] = self._truncated_normal(
                    rng, value, low, high, sigma, n_walkers)
        if len(np.unique(walkers, axis=0)) != n_walkers:
            raise RuntimeError('Walker initialization produced duplicates.')
        return walkers

    def run_mcmc(
            self, n_steps=1000, n_walkers=None, initialization='best_fit',
            best_fit_values=None, best_fit_fwhm_fraction=0.02, seed=42,
            backend_path=None, resume_backend=False, restart_chain_path=None,
            burn_in=0, thin=1, progress=True, **sampler_arguments):
        """Run or resume emcee and retain the maximum-likelihood sample."""
        try:
            import emcee
        except ImportError as error:
            raise ImportError('Install emcee to run the MCMC fitter.') from error

        if n_walkers is None:
            n_walkers = max(32, 2 * self.n_dimensions + 2)
        if n_walkers < 2 * self.n_dimensions:
            raise ValueError('n_walkers must be at least 2 * n_dimensions.')

        backend = None
        initial_state = None
        if backend_path is not None:
            backend = emcee.backends.HDFBackend(str(backend_path))
            if resume_backend:
                if backend.iteration == 0:
                    raise ValueError('Cannot resume an empty backend.')
                initial_state = backend.get_last_sample()
                n_walkers = initial_state.coords.shape[0]
            else:
                backend.reset(n_walkers, self.n_dimensions)
                write_parameter_names(backend_path, self.parameter_names)

        if initial_state is None and restart_chain_path is not None:
            saved = np.load(restart_chain_path)
            saved_names = list(saved['parameter_names'].astype(str))
            if saved_names != self.parameter_names:
                raise ValueError('Saved parameter order does not match.')
            initial_state = np.asarray(saved['last_positions'])
            n_walkers = initial_state.shape[0]
        elif initial_state is None and initialization == 'prior':
            initial_state = self.walkers_from_prior(n_walkers, seed)
        elif initial_state is None and initialization == 'best_fit':
            center = (best_fit_values or self.best_parameters
                      or self.input_best_fit_values)
            if center is None:
                raise ValueError(
                    'Provide best_fit_values, run fit(), or use '
                    'initialization="prior".')
            initial_state = self.walkers_around_best_fit(
                center, n_walkers, best_fit_fwhm_fraction, seed)
        elif initial_state is None:
            raise ValueError('initialization must be "best_fit" or "prior".')

        sampler = emcee.EnsembleSampler(
            n_walkers, self.n_dimensions, self.log_probability,
            backend=backend, **sampler_arguments)
        sampler.run_mcmc(initial_state, n_steps, progress=progress)
        self.mcmc_sampler = sampler
        self.mcmc_chain = sampler.get_chain()
        self.mcmc_log_probability = sampler.get_log_prob()
        self.flat_samples = sampler.get_chain(
            discard=burn_in, thin=thin, flat=True)
        flat_log_probability = sampler.get_log_prob(
            discard=burn_in, thin=thin, flat=True)
        best_index = int(np.nanargmax(flat_log_probability))
        self.mcmc_best_parameters = self.parameter_vector_to_dictionary(
            self.flat_samples[best_index])
        self.mcmc_best_chi_squared = float(
            -2.0 * flat_log_probability[best_index])
        self.best_parameters = self.mcmc_best_parameters.copy()
        self.best_chi_squared = self.mcmc_best_chi_squared
        (_, self.best_chi_squared_components, _, _, _) = (
            self.evaluate_physical_parameters(self.best_parameters))
        self.posterior_summary = self.summarize_samples(self.flat_samples)
        return sampler

    def save_chain(self, path):
        if self.mcmc_chain is None:
            raise RuntimeError('Run MCMC before saving the chain.')
        np.savez(
            path, chain=self.mcmc_chain,
            log_probability=self.mcmc_log_probability,
            last_positions=self.mcmc_chain[-1],
            parameter_names=np.asarray(self.parameter_names, dtype=str))
        return path

    def summarize_samples(self, samples):
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        return {
            name: {
                'median': float(percentiles[1, index]),
                'minus_1sigma': float(
                    percentiles[1, index] - percentiles[0, index]),
                'plus_1sigma': float(
                    percentiles[2, index] - percentiles[1, index])}
            for index, name in enumerate(self.parameter_names)}

    def mcmc_corner_plot(self, max_samples=None, seed=42, **corner_kwargs):
        """Plot the physical posterior with native logarithmic axes."""
        if self.flat_samples is None:
            raise RuntimeError('Run MCMC or restore a saved chain first.')
        samples = self.flat_samples
        if max_samples is not None and len(samples) > max_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(samples), max_samples, replace=False)
            samples = samples[np.sort(indices)]
        defaults = {'title_kwargs': {'fontsize': 10}}
        if self.best_parameters is not None:
            defaults['truths'] = self.dictionary_to_parameter_vector(
                self.best_parameters)
        defaults.update(corner_kwargs)
        return make_corner_plot(
            samples, self.parameter_names, self.log_space_parameters,
            **defaults)
