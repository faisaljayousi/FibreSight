import numpy as np
from skimage.filters import gaussian


def normalise_pm(x: np.ndarray) -> np.ndarray:
    r""" """
    positive_part, negative_part = x > 0, x < 0
    out = np.where(
        positive_part,
        x - x[positive_part].mean(),
        np.where(negative_part, x - x[negative_part].mean(), x),
    )
    return out


def normalise_gaussian(img, sig):
    r""" """
    local_avg = gaussian(img, sigma=(sig, sig), preserve_range=True)
    img_c = img - local_avg
    # local_var = gaussian(img_c ** 2, sigma=(sig, sig), preserve_range=True)
    return img_c  # / np.sqrt(local_var)


def normalise_wout_zeros(mat):
    r"""Mean is calculated using only non-zero components of `mat`.
    Normalisation is then calculated by subtracting the mean and dividing
    by the squared norm of vec(`mat`).

    This code assumes that the input matrix mat contains at least
    one nonzero value.
    """
    eps = 1e-15

    # create a masked array with zeros masked
    mat_nonzero = np.ma.masked_equal(mat, 0)
    mu_nonzero = np.ma.mean(mat_nonzero)  # calculate mean of non-zero values

    centered_mat = mat - mu_nonzero  # subtract mean from all values
    centered_mat[(centered_mat < eps) & (centered_mat > -eps)] = (
        0  # numerical correction
    )

    norm_squared = np.sum(centered_mat**2)

    return centered_mat / norm_squared


def theta_lamb_comb(
    theta_vals: np.ndarray, lambd_vals: np.ndarray
) -> np.ndarray:
    r"""Returns list of all possible (theta, lambd) combinations"""
    return np.array(np.meshgrid(theta_vals, lambd_vals)).T.reshape(-1, 2)
