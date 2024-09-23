import numpy as np
from numpy.exceptions import DTypePromotionError


def _circfuncs_common(samples, high, low):
    # Ensure samples are array-like and size is not zero
    if samples.size == 0:
        NaN = _get_nan(samples)
        return NaN, NaN, NaN

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low) * 2.0 * np.pi / (high - low))
    cos_samp = np.cos((samples - low) * 2.0 * np.pi / (high - low))

    return samples, sin_samp, cos_samp


def _get_nan(*data):
    # Get NaN of appropriate dtype for data
    data = [np.asarray(item) for item in data]
    try:
        dtype = np.result_type(*data, np.half)  # must be a float16 at least
    except DTypePromotionError:
        # fallback to float64
        return np.array(np.nan, dtype=np.float64)[()]
    return np.array(np.nan, dtype=dtype)[()]
