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
