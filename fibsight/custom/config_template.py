from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GraphConfig:
    """
    Configuration class for graph processing methods in image analysis.

    Parameters
    ----------
    binary_method : Optional[str]
        The name of the thresholding method used to binarize the image.
        Default is "otsu".
    method_callable : Optional[Callable]
        A custom callable function for thresholding. If provided,
        it takes precedence over `binary_method`.
    skeleton_method : str
        The method used for skeletonisation of binary images.
        Default is "parallel_thinning".
    prune : bool
        Whether to prune small branches during skeletonisation. Default is True.
    min_length : int
        The minimum length of skeleton segments to retain during pruning.
        Default is 20.
    max_iter : int
        The maximum number of iterations for skeletonisation algorithms.
        Defaults is 5.
    partition_method : str
        The method used for partitioning the image/graph. Defaults is "voronoi".

    Methods
    -------
    get_binary_method() -> str
        Returns the name of the binary method. If `method_callable` is provided,
        its name is returned; otherwise, `binary_method` is returned.
    """

    binary_method: str = "otsu"
    method_callable: Optional[Callable] = None
    skeleton_method: str = "parallel_thinning"
    prune: bool = True
    min_length: int = 20
    max_iter: int = 5
    partition_method: str = "voronoi"

    def get_binary_method(self) -> str:
        """
        Retrieves the binary method name used for thresholding the image.

        The method checks if a custom callable (`method_callable`) is provided.
        If so, the name of this callable is returned, which takes precedence
        over the default `binary_method`. If no callable is provided, the
        method returns the value of `binary_method`.

        Returns
        -------
        str
            The name of the thresholding method to be used.
        """
        if self.method_callable:
            return self.method_callable.__name__
        return self.binary_method

    # def set_binary_method(self):
    #     """Sets the binary method based on the presence of method_callable."""
    #     if self.method_callable:
    #         self.binary_method = self.method_callable.__name__
    #     # This condition can be used to reset the binary method if needed.
    #     elif self.binary_method is None:
    #         self.binary_method = "otsu"
