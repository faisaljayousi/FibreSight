from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import skl_graph.map as mp
from skimage.morphology import medial_axis, skeletonize


class SkeletonisationMethod(ABC):
    """
    Abstract base class for skeletonisation methods.

    Subclasses must implement the `__call__` method.
    """

    @abstractmethod
    def __call__(
        self, binary_image: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply the skeletonisation method to ``binary_image``.

        Parameters
        ----------
        binary_image : np.ndarray
            The binary image to be skeletonised.

        Returns
        -------
        np.ndarray
            The skeletonised image.
        """
        pass


class ParallelThinning(SkeletonisationMethod):
    """
    Skeletonisation method using Guo's parallel thinning algorithm.

    References
    ----------
    [Guo92] Z. Guo and R. W. Hall, "Parallel thinning with
           two-subiteration algorithms," Comm. ACM, vol. 32, no. 3,
           pp. 359-373, 1989. :DOI:`10.1145/62065.62074`
    """

    def __call__(self, binary_image):
        return mp.SKLMapFromObjectMap(binary_image, with_width=True)


class LeeSkeleton(SkeletonisationMethod):
    """
    Skeletonisation method using the 'Lee' algorithm from the
    `skimage.morphology` library.

    References
    ----------
    [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
           via 3-D medial surface/axis thinning algorithms.
           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.
    """

    def __call__(self, binary_image):
        return skeletonize(binary_image, method="lee")


class ZhangSkeleton(SkeletonisationMethod):
    """
    Skeletonisation method using the 'Zhang' algorithm from the
    `skimage.morphology` library.

    References
    ----------
    [Zha84] A fast parallel algorithm for thinning digital patterns,
            T. Y. Zhang and C. Y. Suen, Communications of the ACM,
            March 1984, Volume 27, Number 3.
    """

    def __call__(self, binary_image):
        return skeletonize(binary_image, method="zhang")


class MedialAxisSkeleton(SkeletonisationMethod):
    """
    Skeletonisation method using the Medial Axis algorithm from the
    `skimage.morphology` library.
    """

    def __call__(self, binary_image):
        return medial_axis(binary_image)
