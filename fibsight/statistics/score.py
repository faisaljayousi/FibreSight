from typing import Optional, Tuple, Union

import numpy as np

from .statistic import Statistic
from .statistics_utils import _circfuncs_common


class AlignScore(Statistic):
    def __call__(self, samples, weights, axis=None):
        return align_score(
            samples, high=np.pi, low=0, weights=weights, axis=axis
        )


def align_score(
    samples: np.ndarray,
    high: float = 2 * np.pi,
    low: float = 0,
    weights: Optional[np.ndarray] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> float:
    """Compute the alignment score for samples assumed to be in a range.

    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for the sample range. Default is ``2*pi``.
    low : float or int, optional
        Low boundary for the sample range. Default is 0.

    Returns
    -------
    float
        Alignment score

    See Also
    --------
    circmean : Circular mean.
    circstd : Circular standard deviation.

    Notes
    -----
    This uses the following definition of circular variance: ``1-R``, where
    ``R`` is the mean resultant vector. The
    returned value is in the range [0, 1], 0 standing for no variance, and 1
    for a large variance. In the limit of small angles, this value is similar
    to half the 'linear' variance.

    References
    ----------
    .. [1] Fisher, N.I. *Statistical analysis of circular data*. Cambridge
          University Press, 1993.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import circvar
    >>> import matplotlib.pyplot as plt
    >>> samples_1 = np.array([0.072, -0.158, 0.077, 0.108, 0.286,
    ...                       0.133, -0.473, -0.001, -0.348, 0.131])
    >>> samples_2 = np.array([0.111, -0.879, 0.078, 0.733, 0.421,
    ...                       0.104, -0.136, -0.867,  0.012,  0.105])
    >>> circvar_1 = circvar(samples_1)
    >>> circvar_2 = circvar(samples_2)

    Plot the samples.

    >>> fig, (left, right) = plt.subplots(ncols=2)
    >>> for image in (left, right):
    ...     image.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...                np.sin(np.linspace(0, 2*np.pi, 500)),
    ...                c='k')
    ...     image.axis('equal')
    ...     image.axis('off')
    >>> left.scatter(np.cos(samples_1), np.sin(samples_1), c='k', s=15)
    >>> left.set_title(f"circular variance: {np.round(circvar_1, 2)!r}")
    >>> right.scatter(np.cos(samples_2), np.sin(samples_2), c='k', s=15)
    >>> right.set_title(f"circular variance: {np.round(circvar_2, 2)!r}")
    >>> plt.show()

    """
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    if weights is None:
        sin_mean = sin_samp.mean(axis)
        cos_mean = cos_samp.mean(axis)
    else:
        sin_mean = np.average(sin_samp, axis=axis, weights=weights)
        cos_mean = np.average(cos_samp, axis=axis, weights=weights)
    # hypot can go slightly above 1 due to rounding errors
    with np.errstate(invalid="ignore"):
        R = np.minimum(1, np.hypot(sin_mean, cos_mean))

    return 1.0 - R
