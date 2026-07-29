import numpy as np
from types import SimpleNamespace

import pyGraterFit.fitters.multi_component_sed_visibility_mcmc as mcmc_module
from pyGraterFit.fitters.multi_component_sed_visibility_dynesty import (
    SEDVisibilityNestedFitter,
)


class FakeImage:
    def __init__(self, *args, **kwargs):
        self.pixAU = 1.0
        self.radiative_transfer = SimpleNamespace(
            stellar_spectrum_interpolator=lambda waves: np.ones_like(
                np.asarray(waves, dtype=float)))

    def prepare_spatial_disk(self, **kwargs):
        return None

    def get_image(self, **kwargs):
        return np.ones((2, 2, 2), dtype=float)


def fake_additive_init(
        self, components, star, density_distribution, size_distribution,
        scattering_phase_function, wavelengths, fluxes, fluxes_err,
        params_by_component, **kwargs):
    self.components = dict(components)
    self.component_names = list(components)
    self.star = star
    self.density_distribution = density_distribution
    self.size_distribution = size_distribution
    self.scattering_phase_function = scattering_phase_function
    self.wavelengths = np.asarray(wavelengths, dtype=float)
    self.obs = np.asarray(fluxes, dtype=float)
    self.obs_err = np.asarray(fluxes_err, dtype=float)
    self._inv_obs_err = 1.0 / self.obs_err
    self.params_by_component = {k: dict(v) for k, v in params_by_component.items()}
    self.component_groups = kwargs.get('component_groups') or {
        name: name for name in self.component_names}
    self.shared_parameter_names = tuple(kwargs.get('shared_parameter_names', ()))
    self.group_shared_parameter_names = tuple(
        kwargs.get('group_shared_parameter_names', ()))
    self.mass_abundance_groups = self.component_groups
    self.parallel_components = False
    self._executor = None
    self.ndim = 1
    self.param_names = ['component.A_norm']
    self.log_params = {'component.A_norm'}
    self.prior_ranges = {'component.A_norm': (1.0, 10.0)}
    self._entries = [
        {'label': 'component.A_norm', 'prior': (1.0, 10.0), 'log': True}]
    self._bounds = [(0.0, 1.0)]
    self._bounds_lo = np.array([0.0])
    self._bounds_hi = np.array([1.0])
    self.best_params = None
    self.best_chi2 = np.inf


def fake_full_component_params(self, values, name):
    return dict(self.params_by_component[name])


def fake_values_to_vector(self, values, optimizer_space=False):
    del values, optimizer_space
    return np.array([1.0])


def fake_vector_to_values(self, values, optimizer_space=False):
    del values, optimizer_space
    return {'component': {'A_norm': 1.0}}


def fake_model(self, values):
    return np.full_like(self.obs, 1.0, dtype=float)


def fake_observables_from_image(image, pixel_scale_au, distance_pc, **kwargs):
    del image, pixel_scale_au, distance_pc
    return np.ones_like(kwargs['vis2_u_m'], dtype=float), None, None


def build_fitter(monkeypatch, **kwargs):
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '__init__', fake_additive_init)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '_full_component_params',
        fake_full_component_params)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '_values_to_vector',
        fake_values_to_vector)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '_vector_to_values',
        fake_vector_to_values)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, 'model', fake_model)
    monkeypatch.setattr(mcmc_module, 'Image', FakeImage)
    monkeypatch.setattr(
        mcmc_module, 'observables_from_image', fake_observables_from_image)
    spatial_params = {
        'r0': 1.0, 'h0': 0.05, 'alphain': 10.0, 'alphaout': -4.0,
        'beta': 1.0, 'gamma': 2.0, 'itilt': 0.0, 'PA': 0.0,
        'omega': 0.0, 'A_norm': (1.0, 10.0),
    }
    return mcmc_module.SEDVisibilityMCMCFitter(
        components={'component': object()},
        star=SimpleNamespace(distance=100.0),
        density_distribution=object(),
        size_distribution=object(),
        scattering_phase_function=object(),
        vis2={
            'value': np.array([1.0, 1.0]),
            'error': np.array([0.1, 0.1]),
            'u_m': np.array([10.0, 20.0]),
            'v_m': np.array([0.0, 0.0]),
            'wavelength_m': np.array([8e-6, 16e-6]),
        },
        params_by_component={'component': spatial_params},
        include_unresolved_star=False,
        image_settings={'nx': 2, 'ny': 2, 'FOV_AU': 2.0, 'nl': 3},
        **kwargs)


def test_v2_only_uses_nearest_image_wavelength_without_mismatch_error(monkeypatch):
    fitter = build_fitter(
        monkeypatch,
        image_wavelengths=np.array([7.0, 20.0]),
        maximum_wavelength_mismatch=0.0,
    )
    assert fitter.fit_sed is False
    assert fitter.n_sed_points == 0
    assert fitter.n_vis2_points == 2
    assert np.array_equal(fitter._vis2_image_index, np.array([0, 1]))
    assert np.all(fitter._vis2_wavelength_mismatch > 0.0)
    objective, components = fitter.evaluate_physical_parameters(
        {'component': {'A_norm': 1.0}})
    assert objective == components['vis2_chi2_per_point']
    assert components['sed_chi2'] == 0.0


def test_sed_plus_v2_still_adds_both_reduced_terms(monkeypatch):
    fitter = build_fitter(
        monkeypatch,
        sed_wavelengths=np.array([10.0]),
        sed_fluxes=np.array([2.0]),
        sed_flux_errors=np.array([0.5]),
        image_wavelengths=np.array([10.0]),
    )
    assert fitter.fit_sed is True
    objective, components = fitter.evaluate_physical_parameters(
        {'component': {'A_norm': 1.0}})
    assert objective == (
        components['sed_chi2_per_point']
        + components['vis2_chi2_per_point'])
    assert components['sed_chi2'] > 0.0


def test_visibility_nested_wrapper_builds_v2_only(monkeypatch):
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '__init__', fake_additive_init)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '_full_component_params',
        fake_full_component_params)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '_values_to_vector',
        fake_values_to_vector)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, '_vector_to_values',
        fake_vector_to_values)
    monkeypatch.setattr(
        mcmc_module.AdditiveSEDMCMCFitter, 'model', fake_model)
    monkeypatch.setattr(mcmc_module, 'Image', FakeImage)
    monkeypatch.setattr(
        mcmc_module, 'observables_from_image', fake_observables_from_image)
    spatial_params = {
        'r0': 1.0, 'h0': 0.05, 'alphain': 10.0, 'alphaout': -4.0,
        'beta': 1.0, 'gamma': 2.0, 'itilt': 0.0, 'PA': 0.0,
        'omega': 0.0, 'A_norm': (1.0, 10.0),
    }
    fitter = SEDVisibilityNestedFitter(
        components={'component': object()},
        star=SimpleNamespace(distance=100.0),
        density_distribution=object(),
        size_distribution=object(),
        scattering_phase_function=object(),
        vis2={
            'value': np.array([1.0]),
            'error': np.array([0.1]),
            'u_m': np.array([10.0]),
            'v_m': np.array([0.0]),
            'wavelength_m': np.array([8e-6]),
        },
        params_by_component={'component': spatial_params},
        include_unresolved_star=False,
        image_wavelengths=np.array([20.0]),
        image_settings={'nx': 2, 'ny': 2, 'FOV_AU': 2.0, 'nl': 3},
    )
    assert fitter.fit_sed is False
    assert fitter.ndim == 1
    transformed = fitter.prior_transform(np.array([0.5]))
    assert transformed.shape == (1,)
    assert np.isfinite(fitter.log_likelihood(transformed))
