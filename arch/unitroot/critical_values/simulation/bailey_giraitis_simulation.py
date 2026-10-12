"""
Quantiles of Bailey and Giraitis' Q unit root test statistic

As for the KPSS statistic, Q is simulated for random walks of 2,000
observations with standard normal steps. Every discrete Fourier coefficient
Q uses, of the differences and of the (detrended) levels, is a fixed linear
function of the steps, so their real and imaginary parts are jointly normal
with covariance B'B, where B is that linear map (Bailey and Giraitis 2016, (11)
and Lemma 4, exact here for normal steps). B is found by applying the
statistic's own operations to the identity matrix, and each replication draws
the coefficients from numpy's ``Generator.multivariate_normal`` rather than
simulating the 2,000 steps. The quantiles are written to
``bailey_giraitis.py`` in the format of ``kpss.py``.
"""

from collections.abc import Sequence

import numpy as np

from arch._typing import Float64Array

FREQUENCIES = ((3, 10, 1, 2), (3, 8, 1, 3), (3, 7, 1, 4))
"""The frequency ranges (l, k, p, q) of Table 1 of Bailey and Giraitis (2016)."""

PERCENTILES = np.concatenate(
    [
        np.arange(0.0, 1.0, 0.1),
        np.arange(1.0, 10.0, 0.5),
        np.arange(10.0, 90.0, 2.0),
        np.arange(90.0, 99.0, 0.5),
        np.arange(99.0, 100.0, 0.05),
        [100.0],
    ]
)


def transform(frequencies: Sequence[int], trend: str, nobs: int) -> Float64Array:
    """
    The linear map from the steps to the Fourier coefficients Q uses.

    Parameters
    ----------
    frequencies : sequence of int
        (l, k, p, q), the numerator's frequencies l to k and the denominator's
        p to q.
    trend : {"c", "ct"}
        The deterministic terms, as in ``BaileyGiraitis``.
    nobs : int
        The number of observations after y[0] = 0.

    Returns
    -------
    ndarray
        nobs by 2 (k - l + 1) + 2 (q - p + 1) array: the real and imaginary
        parts of the differences' coefficients at l to k, then those of the
        levels' coefficients at p to q, for each unit step.
    """
    first, last, low, high = frequencies
    steps = np.eye(nobs)
    y = np.concatenate([np.zeros((nobs, 1)), np.cumsum(steps, axis=1)], axis=1)
    if trend == "ct":
        time = np.arange(1, nobs + 1, dtype=float)
        centered = time - time.mean()
        beta = (y[:, 1:] - y[:, 1:].mean(axis=1, keepdims=True)) @ centered
        beta /= centered @ centered
        y = y - beta[:, None] * np.arange(nobs + 1)
    changes = np.fft.rfft(np.diff(y, axis=1), axis=1)[:, first : last + 1]
    levels = np.fft.rfft(y[:, 1:], axis=1)[:, low : high + 1]
    return np.concatenate(
        [changes.real, changes.imag, levels.real, levels.imag], axis=1
    )


def simulate(
    frequencies: Sequence[int],
    trend: str,
    nobs: int,
    reps: int,
    rng: np.random.Generator,
) -> Float64Array:
    """
    Simulate Q for reps random walks of nobs standard normal steps.

    Parameters
    ----------
    frequencies : sequence of int
        (l, k, p, q).
    trend : {"c", "ct"}
        The deterministic terms.
    nobs : int
        The number of observations after y[0] = 0.
    reps : int
        The number of statistics.
    rng : Generator
        The random number generator.

    Returns
    -------
    ndarray
        The statistics.
    """
    first, last, _, _ = frequencies
    linear = transform(frequencies, trend, nobs)
    coefficients = rng.multivariate_normal(
        np.zeros(linear.shape[1]), linear.T @ linear, size=reps, method="eigh"
    )
    split = 2 * (last - first + 1)
    return (
        nobs**2
        / (2 * np.pi) ** 2
        * (coefficients[:, :split] ** 2).sum(axis=1)
        / (coefficients[:, split:] ** 2).sum(axis=1)
    )


def quantiles(
    frequencies: Sequence[int],
    trend: str,
    nobs: int,
    reps: int,
    block: int,
    seed: int,
) -> Float64Array:
    """
    The quantiles at PERCENTILES of reps statistics, simulated in blocks.

    Parameters
    ----------
    frequencies : sequence of int
        (l, k, p, q).
    trend : {"c", "ct"}
        The deterministic terms.
    nobs : int
        The number of observations after y[0] = 0.
    reps : int
        The number of statistics.
    block : int
        The number of statistics drawn at once.
    seed : int
        The generator's seed.

    Returns
    -------
    ndarray
        The quantiles.
    """
    rng = np.random.default_rng(seed)
    draws = np.concatenate(
        [
            simulate(frequencies, trend, nobs, min(block, reps - start), rng)
            for start in range(0, reps, block)
        ]
    )
    return np.percentile(draws, PERCENTILES)


if __name__ == "__main__":
    import sys

    nobs, reps, block = (int(arg) for arg in sys.argv[1:])
    print("from numpy import asarray")
    print()
    print("bailey_giraitis_critical_values = {}")
    for seed, (trend, frequencies) in enumerate(
        (trend, frequencies) for trend in ("c", "ct") for frequencies in FREQUENCIES
    ):
        values = quantiles(frequencies, trend, nobs, reps, block, seed)
        rows = ",\n    ".join(
            f"({percent:.3f}, {value:#.6g})"
            for percent, value in zip(100 - PERCENTILES, values, strict=True)
        )
        print(f"c = (\n    {rows},\n)")
        print(
            f"bailey_giraitis_critical_values[{trend!r}, {tuple(frequencies)!r}] "
            "= asarray(c)"
        )
