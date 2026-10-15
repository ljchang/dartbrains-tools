"""Small, reproducible simulations for teaching multiple-comparison correction.

These helpers operate on finite synthetic arrays, not whole-brain inference.
The cluster statistic uses four-connected neighbours on the absolute-value map,
so adjacent positive and negative values can join. Production neuroimaging
tools generally form positive and negative clusters separately.

Numba is optional: install ``dartbrains-tools[simulation]`` for accelerated
cluster labeling on native Python. The default works with SciPy alone,
including environments without a JIT. Importing this module never imports Numba.
"""
from functools import lru_cache
from numbers import Integral, Real

import numpy as np
from scipy.ndimage import label


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def simulate_max_t(n_simulations=100, n_voxels=10_000, n_subjects=20, seed=0):
    """Maximum absolute one-sample t for each independent Gaussian experiment.

    Draws match ``default_rng(seed).standard_normal((n_subjects, n_voxels))``
    once per experiment. Reuse the maxima across every candidate cutoff. Only
    one experiment's data is held at a time. The direct NumPy formula avoids
    calculating p-values that the threshold sweep does not use.
    """
    n_simulations = _positive_integer(n_simulations, "n_simulations")
    n_voxels = _positive_integer(n_voxels, "n_voxels")
    n_subjects = _positive_integer(n_subjects, "n_subjects", 2)
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_simulations)
    for i in range(n_simulations):
        noise = rng.standard_normal((n_subjects, n_voxels))
        standard_error = np.sqrt(noise.var(axis=0, ddof=1) / n_subjects)
        maxima[i] = np.max(np.abs(noise.mean(axis=0)) / standard_error)
    return maxima


@lru_cache(maxsize=1)
def _numba_kernel():
    # Lazy import keeps ordinary imports and the SciPy backend lightweight.
    from ._simulation_numba import maximum_cluster_sizes as kernel

    return kernel


def maximum_cluster_sizes(maps, width, forming, *, backend="auto"):
    """Largest four-connected cluster in each thresholded square map.

    Parameters
    ----------
    maps : array, shape (n_maps, width**2)
        Finite statistic maps, usually sign-flipped group means.
    width : int
        Side length of each square map.
    forming : float
        Nonnegative statistic cutoff; voxels survive when ``abs(value) > forming``.
    backend : {"auto", "scipy", "numba"}
        Auto uses Numba if installed, otherwise SciPy. Explicit numba raises a
        helpful ImportError when unavailable. No fastmath or precision change.
    """
    width = _positive_integer(width, "width")
    if (isinstance(forming, (bool, np.bool_)) or not isinstance(forming, Real)
            or not np.isfinite(forming) or forming < 0):
        raise ValueError("forming must be a finite, nonnegative statistic cutoff")
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width**2 or not np.isfinite(values).all():
        raise ValueError("maps must be finite with shape (n_maps, width**2)")
    if backend not in ("auto", "scipy", "numba"):
        raise ValueError("backend must be 'auto', 'scipy', or 'numba'")
    if backend != "scipy":
        try:
            kernel = _numba_kernel()
        except ImportError as error:
            if backend == "numba":
                raise ImportError("Install dartbrains-tools[simulation] to use backend='numba'") from error
        else:
            return kernel(values, width, float(forming))
    maxima = np.zeros(len(values), dtype=np.int64)
    for i, row in enumerate(values):
        labels, count = label((np.abs(row) > forming).reshape(width, width))
        if count:
            maxima[i] = np.bincount(labels.ravel())[1:].max()
    return maxima


def cluster_fwe_p(observed, null, width, forming, *, backend="auto"):
    """Return (corrected p, observed maximum size, null maximum sizes).

    Uses ``(count(null_max >= observed_max) + 1) / (n_permutations + 1)``.
    This includes ties and the observed arrangement. A map without a cluster
    has p=1. Observed and null maps use the same fixed forming threshold.
    """
    null_max = maximum_cluster_sizes(null, width, forming, backend=backend)
    if not len(null_max):
        raise ValueError("null must contain at least one permutation")
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim != 1:
        raise ValueError("observed must be a vector of width**2 voxels")
    size = int(maximum_cluster_sizes(observed[None, :], width, forming, backend=backend)[0])
    p_value = (np.count_nonzero(null_max >= size) + 1) / (len(null_max) + 1) if size else 1.0
    return float(p_value), size, null_max
