import numpy as np
import skan
from skimage import measure
from skimage.morphology import remove_small_holes, skeletonize

np.float_ = float  # overrides deprecated alias with built-in float


def prune_skeleton(
    skeleton: np.ndarray, min_length: int, max_iter: int = 5
) -> np.ndarray:
    """
    Prunes branches of the skeleton that are shorter than `min_length`.

    Parameters:
    -----------
    skeleton : np.ndarray
        Binary skeleton image to be pruned.
    min_length : int
        Minimum length of branches to retain.
    max_iter : int, optional
        Maximum number of pruning iterations. Default is 5.

    Returns:
    --------
    np.ndarray
        Pruned skeleton image.

    Example:
    --------
    >>> pruned = prune_skeleton(skeleton, min_length=40)
    """
    # Remove small holes from skeleton
    skeleton = remove_small_holes(skeleton, area_threshold=100)

    # Reconstruct skeleton
    skeleton = skeletonize(skeleton)

    # Initialise skan.Skeleton object
    skeleton = skan.Skeleton(skeleton)

    for _ in range(max_iter):
        summary = skan.summarize(skeleton)
        to_cut = (
            (summary["euclidean-distance"] < min_length)
            & (summary["branch-type"] < 2)
        ) | (
            (summary["euclidean-distance"] < 2) & (summary["branch-type"] == 2)
        )
        pruned = skeleton.prune_paths(np.flatnonzero(to_cut))
        skeleton = pruned

    return remove_isolated_pixels(skeleton.skeleton_image, 2)


def remove_isolated_pixels(skeleton: np.ndarray, min_area: int) -> np.ndarray:
    """
    Removes isolated pixels from the skeleton based on minimum area threshold.

    Parameters:
    -----------
    skeleton : np.ndarray
        Binary skeleton image from which isolated pixels will be removed.
    min_area : int
        Minimum area threshold for connected components to keep.

    Returns:
    --------
    np.ndarray
        Skeleton image with isolated pixels removed.
    """
    # Initialise holder
    coords_of_interest = np.array([[0, 0]])

    # Label connected components in skeleton
    label_matrix = measure.label(skeleton)
    props = measure.regionprops(label_matrix)

    for prop in props:
        if prop.area < min_area:
            coords = np.array(prop.coords)
            coords_of_interest = np.concatenate(
                [coords_of_interest, coords], axis=0
            )

    # Exclude placeholder coordinate
    coords_of_interest = coords_of_interest[1:]
    skeleton[coords_of_interest[:, 0], coords_of_interest[:, 1]] = 0

    return skeleton
