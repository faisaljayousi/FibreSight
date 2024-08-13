from abc import ABC, abstractmethod

import numpy as np
from skimage.filters import threshold_mean, threshold_otsu


class ThresholdingMethod(ABC):
    """
    Abstract base class for thresholding methods.

    Subclasses must implement the `__call__` method.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ """
        pass


class OtsuThresholding(ThresholdingMethod):
    def __call__(self, image):
        global_thresh = threshold_otsu(image)
        return image > global_thresh


class MeanThresholding(ThresholdingMethod):
    def __call__(self, image):
        threshold = threshold_mean(image)
        return image > threshold
