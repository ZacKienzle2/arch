from numpy import arange, asarray, log2, squeeze
from numpy.fft import irfft, rfft
from scipy.fft import next_fast_len

from arch._typing import Float64Array


def cov_nw(
    y: Float64Array, lags: int = 0, demean: bool = True, axis: int = 0, ddof: int = 0
) -> Float64Array | float:
    """
    Computes Newey-West covariance for 1-d and 2-d arrays

    Parameters
    ----------
    y : ndarray
        Values to use when computing the Newey-West covariance estimator, either
        1-d or 2-d. When y is 2d, default behavior is to treat columns as variables
        and rows as observations.
    lags : int
        Number of lags to include in the Newey-West covariance estimator
    demean : bool
        Indicates whether to subtract the mean.  Default is True
    axis : int
        The axis to use when y is 2d
    ddof : int
        Degree of freedom correction for compatibility with simple covariance
        estimators.  Default is 0.

    Returns
    -------
    ndarray
        The estimated covariance

    Notes
    -----
    Summing one product over the sample for each lag costs ``n * lags``. For
    1-d data, the autocovariances of every lag come instead from the inverse
    FFT of the periodogram of the data zero-padded to at least ``n + lags``,
    so the circular correlation does not wrap at the lags used, which costs
    in the order of ``m log2(m)`` for padded length ``m``. The cheaper of
    the two is used.
    """
    z = y
    is_1d = False
    if axis > z.ndim:
        raise ValueError("axis must be less than the dimension of y")
    if z.ndim == 1:
        is_1d = True
        z = asarray(z)[:, None]
    if axis == 1:
        z = z.T
    n = z.shape[0]
    if ddof > n:
        raise ValueError(
            "ddof must be strictly smaller than the number of observations"
        )
    if lags > n:
        error = "lags must be weakly smaller than the number of observations"
        raise ValueError(error)

    if demean:
        z = z - z.mean(0)
    size = next_fast_len(n + lags)
    if is_1d and lags * n > size * log2(size):
        spectrum = rfft(z[:, 0], size)
        gammas = irfft(spectrum.real**2 + spectrum.imag**2, size)[: lags + 1]
        weights = 1 - arange(lags + 1) / (lags + 1)
        return float((2 * weights @ gammas - gammas[0]) / (n - ddof))
    cov = z.T.dot(z)
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        gamma = z[j:].T.dot(z[:-j])
        cov += w * (gamma + gamma.T)
    cov = cov / (n - ddof)
    if is_1d:
        return float(squeeze(cov))
    return asarray(cov)
