import cmath
from typing import Dict

import cv2
import numpy as np

from .texture_utils import normalise_pm, theta_lamb_comb

# @dataclass
# class GaborKernelParams:
#     size: Tuple[int, int]
#     gamma: float
#     psi: float
#     angle: Union[List, np.ndarray]
#     wavelength: Union[List, np.ndarray]
#     b: float


# def _create_gabor_kernel(
#     kernel_params: GaborKernelParams,
#     angle: float,
#     wavelength: float,
#     sigma: float,
# ) -> np.ndarray:
#     kernel = cv2.getGaborKernel(
#         kernel_params.size,
#         sigma,
#         angle,
#         wavelength,
#         kernel_params.gamma,
#         kernel_params.psi,
#     )
#     return normalise_pm(kernel)


def classify_points_relative_to_line(
    point_matrix: np.ndarray, line_start: tuple, line_end: tuple
) -> np.ndarray:
    """
    Classifies points as being to the left or right of a line segment.

    Given a line defined by two points (`line_start` and `line_end`) and a
    matrix of points (`point_matrix`), this function determines whether each
    point in the matrix is to the 'left' or 'right' of the line segment. The line
    is represented by the vector from `line_start` to `line_end`. Points to the
    'left' of the line are classified as `-1`, and points to the 'right' are
    classified as `1`.

    Parameters:
    - point_matrix (np.ndarray): An array of shape (..., 2) representing the
      coordinates of points in 2D space.
    - line_start (tuple): The starting point of the line segment (x1, y1).
    - line_end (tuple): The ending point of the line segment (x2, y2).

    Returns:
    - np.ndarray: An array of the same shape as `point_matrix`, with values
      `-1` for points to the left of the line and `1` for points to the right.
    """
    assert point_matrix.ndim >= 2

    x1, y1 = line_start
    x2, y2 = line_end
    x_coords, y_coords = point_matrix[..., 0], point_matrix[..., 1]

    # Calculate the determinant to classify points relative to the line
    determinant = (x2 - x1) * (y_coords - y1) - (y2 - y1) * (x_coords - x1)

    return np.where(determinant >= 0, 1, -1)


def left_right_kernel(
    kernel: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a Gabor filter into two overlapping parts around the Gaussian
    enveloppe.

    Parameters:
    - kernel (np.ndarray): The input kernel to be split.
    - theta (float): The orientation angle (in radians) to determine the line
      that separates the left and right sides.

    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple containing:
        - The kernel for the right side of the line.
        - The kernel for the left side of the line.
    """
    h, w = kernel.shape
    coords = np.stack(np.mgrid[:h, :w], axis=2)

    # Filter centre
    centre_x, centre_y = ((w - 1) / 2, (h - 1) / 2)

    pt = cmath.rect(1, theta - np.pi / 2)
    pt2 = cmath.rect(1, theta + np.pi / 2)

    x1, y1 = centre_x + pt.real, centre_y + pt.imag
    x2, y2 = centre_x + pt2.real, centre_y + pt2.imag

    kernel_left, kernel_right = np.zeros_like(kernel), np.zeros_like(kernel)

    test_left_right = classify_points_relative_to_line(
        coords, np.array([x1, y1]), np.array([x2, y2])
    )

    kernel_left = np.where(test_left_right >= 0, kernel, 0)
    kernel_right = np.where(test_left_right < 0, kernel, 0)

    # add missing parts
    pos_right = np.where(kernel_right > 0.05, kernel_right, 0)
    pos_left = np.where(kernel_left > 0.05, kernel_left, 0)

    return kernel_right + pos_left, kernel_left + pos_right


def _max_gabor_full(
    img: np.ndarray,
    /,
    theta_vals: np.ndarray,
    lambd_vals: np.ndarray,
    kernel_params: dict,
    b: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the maximum Gabor filter responses for a given image.

    Applies a set of Gabor filters to an image, each defined by combinations
    of angles and wavelengths. Returns the angles, wavelengths, and the
    corresponding filter responses that produce the maximum response for each
    pixel in the image.

    Parameters:
    - img (np.ndarray): The input image to be filtered.
    - theta_vals (np.ndarray): Array of angles (theta values) for Gabor filters.
    - lambd_vals (np.ndarray): Array of wavelengths (lambda values) for Gabor
    filters.
    - kernel_params (dict): Dictionary containing parameters for the Gabor kernel:
        - "size": Size of the Gabor kernel.
        - "gamma": Spatial aspect ratio of the Gabor kernel.
        - "psi": Phase offset of the Gabor kernel.
    - b (float, optional): Bandwidth parameter for the Gabor filters.
    Default is 1.6.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - An array with the angles that produce the maximum response for
        each pixel.
        - An array with the wavelengths that produce the maximum response for
        each pixel.
        - An array with the maximum responses for each pixel.


    Notes
    -----
    Only supports optimisation over theta and lambda
    """
    # Input validation
    if img.ndim != 2:
        raise ValueError("Input image `img` must be a 2D array.")
    if not theta_vals.size:
        raise ValueError("`theta_vals` must not be empty.")
    if not lambd_vals.size:
        raise ValueError("`lambd_vals` must not be empty.")

    # Initialise arrays to store results
    max_responses = np.zeros_like(img, dtype=np.float64)
    optimal_angles = np.zeros_like(img, dtype=np.float64)
    optimal_wavelengths = np.zeros_like(img, dtype=np.float64)

    for angle, wavelength in theta_lamb_comb(theta_vals, lambd_vals):
        sigma = lambda_to_sigma(wavelength, b)
        kernel = cv2.getGaborKernel(
            kernel_params["size"],
            sigma,
            angle,
            wavelength,
            kernel_params["gamma"],
            kernel_params["psi"],
        )

        # Normalise kernel
        kernel = normalise_pm(kernel)

        # Convolve filter with image
        filtered_image = cv2.filter2D(img, -1, kernel)

        # Update optimal values where new response is higher
        mask = filtered_image > max_responses

        np.copyto(max_responses, filtered_image, where=mask)
        np.copyto(optimal_angles, angle, where=mask)
        np.copyto(optimal_wavelengths, wavelength, where=mask)

    return optimal_angles, optimal_wavelengths, max_responses


def gabor_lr(
    img: np.ndarray,
    theta: float,
    lambd: float,
    kernel_params: dict,
    b: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r""" """

    # img = normalise_gaussian(img, sig)  # local normalisation

    thetas_left, thetas_right = np.zeros_like(
        img, dtype=np.float64
    ), np.zeros_like(img, dtype=np.float64)
    lambds_left, lambds_right = np.zeros_like(
        img, dtype=np.float64
    ), np.zeros_like(img, dtype=np.float64)
    response_left, response_right = np.zeros_like(
        img, dtype=np.float64
    ), np.zeros_like(img, dtype=np.float64)

    kernel = cv2.getGaborKernel(
        kernel_params["size"],
        lambda_to_sigma(lambd, b),
        theta,
        lambd,
        kernel_params["gamma"],
        kernel_params["psi"],
    )

    kernel_right, kernel_left = left_right_kernel(kernel, theta)

    kernel_right = normalise_pm(kernel_right)
    kernel_left = normalise_pm(kernel_left)

    left_filtered_FN = cv2.filter2D(img, -1, kernel_left)
    right_filtered_FN = cv2.filter2D(img, -1, kernel_right)

    mask_left = left_filtered_FN > response_left
    np.copyto(response_left, left_filtered_FN, where=mask_left)
    np.copyto(thetas_left, theta, where=mask_left)
    np.copyto(lambds_left, lambd, where=mask_left)

    mask_right = right_filtered_FN > response_right
    np.copyto(response_right, right_filtered_FN, where=mask_right)
    np.copyto(thetas_right, theta, where=mask_right)
    np.copyto(lambds_right, lambd, where=mask_right)

    thetas = np.minimum(thetas_left, thetas_right)  # point-wise min
    lambds = np.minimum(lambds_left, lambds_right)
    response = np.minimum(response_left, response_right)

    return thetas, lambds, response


def _max_gabor_lr(
    img: np.ndarray,
    theta_vals: np.ndarray,
    lambd_vals: np.ndarray,
    kernel_params: dict,
    b: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a Gabor filter to an image, dividing the filter into left and right
    components based on the orientation angle. The function returns the optimal
    filter responses and corresponding parameters for each pixel in the image.

    Parameters:
    - img (np.ndarray): The input image to be filtered.
    - theta (float): The orientation angle (in radians) of the Gabor filter.
    - lambd (float): The wavelength of the Gabor filter.
    - kernel_params (dict): Dictionary containing parameters for the Gabor kernel:
        - `size`: Size of the Gabor kernel.
        - `gamma`: Spatial aspect ratio of the Gabor kernel.
        - `psi`: Phase offset of the Gabor kernel.
    - b (float, optional): Bandwidth parameter for the Gabor filters.
    Default is 1.6.


    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - An array with the angles producing the maximum response for each pixel.
        - An array with the wavelengths producing the maximum response for each
        pixel.
        - An array with the maximum responses for each pixel.
    """

    response = np.zeros_like(img, dtype=np.float64)
    thetas = np.zeros_like(img, dtype=np.float64)
    lambds = np.zeros_like(img, dtype=np.float64)

    for theta, lambd in theta_lamb_comb(theta_vals, lambd_vals):
        _, _, filtered_FN = gabor_lr(img, theta, lambd, kernel_params, b)

        mask = filtered_FN > response
        np.copyto(response, filtered_FN, where=mask)
        np.copyto(thetas, theta, where=mask)
        np.copyto(lambds, lambd, where=mask)

    return thetas, lambds, response


def _max_gabor(
    img: np.ndarray,
    theta_vals: np.ndarray,
    lambd_vals: np.ndarray,
    kernel_params: Dict[str, float],
    b: float = 1.6,
    method: str = "full",
):
    r""" """
    if method == "full":
        return _max_gabor_full(img, theta_vals, lambd_vals, kernel_params, b)
    elif method == "lr":
        return _max_gabor_lr(img, theta_vals, lambd_vals, kernel_params, b)
    else:
        raise ValueError(
            "Method not recognised. Must be one of {'full', 'lr'}."
        )


def lambda_to_sigma(lambd: float, b: float = 1.6) -> float:
    """
    Converts wavelength (lambda) to standard deviation (sigma) for a Gabor filter.

    This conversion uses the formula from P. Kruizinga and N. Petkov's paper,
    "Nonlinear operator for oriented texture," IEEE Transactions on Image
    Processing, vol. 8, no. 10, pp. 1395-1407, Oct. 1999 (doi: 10.1109/83.791965).

    Parameters:
    - lambd (float): Wavelength of the Gabor filter.
    - b (float, optional): Bandwidth parameter (half-response spatial-frequency)
    of the filter. Default is 1.6.

    Returns:
    - float: Standard deviation (sigma) corresponding to the provided wavelength.
    """

    sigma = lambd / np.pi * np.sqrt(np.log(2) / 2) * (2**b + 1) / (2**b - 1)
    return sigma
