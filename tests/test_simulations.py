"""Numerical equivalence, connected-component semantics and optional backends."""
import importlib.util

import numpy as np
import pytest
from scipy.ndimage import label
from scipy.stats import ttest_1samp

from dartbrains_tools.simulations import cluster_fwe_p, maximum_cluster_sizes, simulate_max_t


def test_sweep_matches_scipy_and_preserves_rng():
    rng = np.random.default_rng(42)
    expected = np.array([np.abs(ttest_1samp(rng.standard_normal((8, 25)), 0, axis=0).statistic).max() for _ in range(20)])
    actual = simulate_max_t(20, 25, 8, seed=42)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(actual[:, None] >= np.arange(0, 10, .1), expected[:, None] >= np.arange(0, 10, .1))
    np.testing.assert_array_equal(actual, simulate_max_t(20, 25, 8, seed=42))


@pytest.mark.parametrize('backend', ['scipy', 'auto', 'numba'])
def test_cluster_backends_match_scipy(backend):
    if backend == 'numba' and importlib.util.find_spec('numba') is None:
        pytest.skip('optional Numba extra is not installed')
    maps = np.random.default_rng(0).normal(size=(30, 25))
    maps[:3] = [np.zeros(25), np.ones(25), np.eye(5).ravel()]
    expected = []
    for values in maps:
        labelled, count = label((np.abs(values) > .7).reshape(5, 5))
        expected.append(np.bincount(labelled.ravel())[1:].max() if count else 0)
    np.testing.assert_array_equal(maximum_cluster_sizes(maps, 5, .7, backend=backend), expected)
    # Strict threshold, four-connected neighbours, no wraparound across rows.
    np.testing.assert_array_equal(maximum_cluster_sizes([[1, -1, 0, 0], [0, 1, 1, 0]], 2, .5, backend=backend), [2, 1])
    np.testing.assert_array_equal(maximum_cluster_sizes([[1, 1, 1, 1]], 2, 1, backend=backend), [0])


def test_cluster_pvalue_ties_plus_one_and_empty_observed():
    null = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1.]])
    p, size, maxima = cluster_fwe_p([1, 1, 0, 0], null, 2, .5, backend='scipy')
    assert p == 3 / 4 and size == 2
    np.testing.assert_array_equal(maxima, [2, 0, 4])
    assert cluster_fwe_p([0, 0, 0, 0], null, 2, .5)[0:2] == (1., 0)


@pytest.mark.parametrize('kwargs', [{'n_subjects': 1}, {'n_simulations': 0}, {'n_voxels': -1}, {'n_subjects': True}, {'n_voxels': 2.5}])
def test_invalid_simulation_sizes(kwargs):
    with pytest.raises(ValueError):
        simulate_max_t(**kwargs)


@pytest.mark.parametrize('maps,width,forming', [([[1, 2]], 2, .5), ([[np.nan]], 1, .5), ([[np.inf]], 1, .5), ([[1]], 0, .5), ([[1]], True, .5), ([[1]], 1, -1), ([[1]], 1, np.nan)])
def test_invalid_cluster_inputs(maps, width, forming):
    with pytest.raises(ValueError):
        maximum_cluster_sizes(maps, width, forming)


def test_unknown_backend_and_empty_null():
    with pytest.raises(ValueError):
        maximum_cluster_sizes([[1]], 1, .5, backend='typo')
    with pytest.raises(ValueError):
        cluster_fwe_p([1], np.empty((0, 1)), 1, .5)


def test_numba_missing_fallback(monkeypatch):
    import dartbrains_tools.simulations as simulations
    def unavailable():
        raise ImportError('not installed')
    monkeypatch.setattr(simulations, '_numba_kernel', unavailable)
    np.testing.assert_array_equal(maximum_cluster_sizes([[1]], 1, .5), [1])
    with pytest.raises(ImportError, match='simulation'):
        maximum_cluster_sizes([[1]], 1, .5, backend='numba')
