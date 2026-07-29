from pyGraterFit.fitters.multi_component_sed_mcmc import (
    _components_from_ring_materials,
)


def test_ring_material_expansion_builds_expected_backend_structure():
    materials = {
        "olivine": object(),
        "carbon": object(),
    }
    ring_params = {
        "inner": {"r0": (0.1, 1.0), "h0": 0.05},
        "outer": {"r0": (1.0, 10.0), "h0": 0.5},
    }

    expanded = _components_from_ring_materials(
        materials, ring_params, normalization_range=(1e28, 1e38))

    assert list(expanded["components"]) == [
        "inner.olivine",
        "inner.carbon",
        "outer.olivine",
        "outer.carbon",
    ]
    assert expanded["component_groups"] == {
        "inner.olivine": "inner",
        "inner.carbon": "inner",
        "outer.olivine": "outer",
        "outer.carbon": "outer",
    }
    assert expanded["group_shared_parameter_names"] == ("r0", "h0")
    assert expanded["normalization_groups"] == expanded["component_groups"]
    assert expanded["normalization_total_ranges"] == {
        "inner": (1e28, 1e38),
        "outer": (1e28, 1e38),
    }
    assert expanded["params_by_component"]["inner.olivine"]["A_norm"] == (
        1e28,
        1e38,
    )


def test_visibility_fitter_accepts_ring_material_interface(monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    import pyGraterFit.fitters.multi_component_sed_visibility_mcmc as module

    captured = {}

    def fake_additive_init(
            self, components, star, density_distribution, size_distribution,
            scattering_phase_function, wavelengths, fluxes, fluxes_err,
            params_by_component, **kwargs):
        captured['components'] = components
        captured['params_by_component'] = params_by_component
        captured['kwargs'] = kwargs
        self.components = dict(components)
        self.component_names = list(components)
        self.obs = np.asarray(fluxes, dtype=float)
        self.obs_err = np.asarray(fluxes_err, dtype=float)
        self.star = star
        self.density_distribution = density_distribution
        self.size_distribution = size_distribution
        self.scattering_phase_function = scattering_phase_function

    class FakeImage:
        def __init__(self, *args, **kwargs):
            self.radiative_transfer = SimpleNamespace(
                stellar_spectrum_interpolator=lambda waves: np.ones_like(
                    np.asarray(waves, dtype=float)))

    monkeypatch.setattr(module.AdditiveSEDMCMCFitter, '__init__', fake_additive_init)
    monkeypatch.setattr(module, 'Image', FakeImage)

    materials = {'astroSi': object(), 'olivine': object()}
    ring_params = {
        'inner': {
            'r0': (0.2, 1.0), 'h0': 0.05, 'alphain': 8.0,
            'alphaout': -4.0, 'beta': 1.0, 'gamma': 2.0,
            'itilt': 55.0, 'PA': 20.0, 'omega': 0.0,
        },
        'outer': {
            'r0': (1.0, 5.0), 'h0': 0.08, 'alphain': 8.0,
            'alphaout': -4.0, 'beta': 1.0, 'gamma': 2.0,
            'itilt': 55.0, 'PA': 20.0, 'omega': 0.0,
        },
    }
    vis2 = {
        'value': [0.8],
        'error': [0.05],
        'u_m': [20.0],
        'v_m': [0.0],
        'wavelength_m': [10e-6],
    }

    fitter = module.SEDVisibilityMCMCFitter(
        materials=materials,
        ring_params=ring_params,
        star=object(),
        density_distribution=object(),
        size_distribution=object(),
        scattering_phase_function=object(),
        sed_wavelengths=np.array([10.0]),
        sed_fluxes=np.array([1.0]),
        sed_flux_errors=np.array([0.1]),
        vis2=vis2,
        image_settings={'nx': 8, 'ny': 8, 'FOV_AU': 4.0, 'nl': 3},
        stellar_visibility_model='uniform_disk',
        stellar_angular_diameter_mas=0.7,
        normalization_range=(1e25, 1e35),
    )

    assert list(captured['components']) == [
        'inner.astroSi', 'inner.olivine', 'outer.astroSi', 'outer.olivine']
    assert captured['kwargs']['component_groups'] == {
        'inner.astroSi': 'inner',
        'inner.olivine': 'inner',
        'outer.astroSi': 'outer',
        'outer.olivine': 'outer',
    }
    assert captured['kwargs']['normalization_mode'] == 'group_total_fraction'
    assert captured['kwargs']['normalization_groups'] == captured['kwargs'][
        'component_groups']
    assert 'stellar_angular_diameter_mas' in captured['kwargs'][
        'shared_parameter_names']
    assert all(
        params['stellar_angular_diameter_mas'] == 0.7
        for params in captured['params_by_component'].values())
    assert fitter.n_vis2_points == 1
