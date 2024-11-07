import inspect
import logging
from typing import Callable, Dict, Optional, Union

import numpy as np

import fibsight as fs
from fibsight.custom.config_template import GraphConfig
from fibsight.graphs.process import ImageToGraph
from fibsight.statistics.statistics_factory import Statistic, StatisticsFactory
from fibsight.texture.gabor import _max_gabor
from fibsight.utils import log_method

config = GraphConfig()


class FibreDescriptor:
    def __init__(
        self,
        image: np.ndarray,
        config: GraphConfig = config,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        image : np.ndarray
            The input image to be processed.
        config : Type[GraphConfig], optional
            A configuration object specifying parameters for image pipeline,
            including binary thresholding, skeletonisation, and partition
            methods. Default is `GraphConfig()` with preset values.
        verbose : int, optional
            Verbosity level for logging. Default is 0.
        """
        # Inits
        self.image = image
        self.config = config
        self.verbose = verbose

        # Initialise enhanced image
        self.enhanced = np.copy(image)

        # ImageToGraph instance using input image
        self.summary = ImageToGraph(self.image)

        # Configure logger
        self.logger = logging.getLogger(__name__)
        fs.set_logging_level(self.verbose)

    def register_partition_method(self, name: str, method: Callable):
        self.summary.register_partition_method(name, method)
        self.logger.info(
            f"Partition method '{name}' "
            f"has been registered in FibreDescriptor."
        )

    @log_method
    def enhance(
        self, lambdas: np.ndarray, kernel_params: Dict, method="full", b=1.6
    ) -> None:
        self.angles, self.lambdas, self.enhanced = _max_gabor(
            self.image,
            theta_vals=np.arange(0, np.pi, np.pi / 16),
            lambd_vals=lambdas,
            kernel_params=kernel_params,
            method=method,
            b=b,
        )

    @log_method
    def make_graph(self) -> None:
        """
        Processes the enhanced image to create a graph representation
        with various steps based on the provided configuration.

        This method performs the following operations:
        1. **Image Preprocessing**: #TODO
        2. **Binary Image Creation**: Creates a binary image using the specified
        thresholding method from the configuration. Default: Otsu's method.
        3. **Skeletonisation**: Converts the binary image into a skeleton
        using the chosen method and parameters from the configuration.
        Default: ParallelThinning.
        4. **Graph Generation**: Constructs a graph representation from
        the skeletonised image.
        5. **Degree Map Computation**: Computes the degree map of the graph.
        6. **Partition Application**: Applies the specified partition method
        to the graph to partition the image.

        The method relies on the attributes of the instance,
        particularly the `self.enhanced`, `self.summary.noise_mask`,
        and `self.config` for configuration details. The `config` parameter
        must be set prior to calling this method to ensure correct execution.

        Raises
        ------
        ValueError
            If any required attribute or configuration is missing or invalid.

        Notes
        -----
        - Ensure that `self.config` is properly initialised with valid methods
        and parameters before invoking this method.
        """
        self.summary.image = self.enhanced * self.summary.noise_mask

        self.summary.make_binary(self.config)

        self.summary.make_skeleton(
            self.config.skeleton_method,
            prune=self.config.prune,
            min_length=self.config.min_length,
            max_iter=self.config.max_iter,
        )

        self.summary.make_graph()

        self.summary.degreeMap()

        self.summary.apply_partition_method(self.config.partition_method)

    @log_method
    def compute_score_partition(
        self,
        x: np.ndarray,
        statistic: Union[str, Statistic],
        mask: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Computes a score for each region defined by the partition
        and stores the result.

        Parameters
        ----------
        x : np.ndarray
            Pixel-wise data array with the same shape as `self.image`.
        statistic : Union[str, Statistic]
            The statistic to compute, either as a string
            (resolved by StatisticsFactory) or a Statistic instance.
        mask : np.ndarray, optional
            Binary mask array with the same shape as `self.image`,
            where 1 indicates valid pixels and 0 indicates ignored pixels.
            Default is None.
        weights : np.ndarray, optional
            Weights for the pixels, with the same shape as `self.image`.
            Default is None.
        **kwargs : dict
            Additional arguments to pass to the statistic method.

        Raises
        ------
        ValueError
            If `x`, `mask`, or `weights` do not have the same shape
            as `self.image`.
        """
        if x.shape != self.image.shape:
            raise ValueError(
                f"'x' must have the same shape as 'self.image'. "
                f"Got {x.shape} and {self.image.shape}."
            )
        if mask is not None and mask.shape != self.image.shape:
            raise ValueError(
                f"'mask' must have the same shape as 'self.image'. "
                f"Got {mask.shape} and {self.image.shape}."
            )
        if weights is not None and weights.shape != self.image.shape:
            raise ValueError(
                f"'weights' must have the same shape as 'self.image'. "
                f"Got {weights.shape} and {self.image.shape}."
            )

        # Initialise output array
        local_description = np.zeros_like(self.image)

        # Use all pixels if no mask is provided
        mask = np.ones_like(self.image, dtype=int) if mask is None else mask

        region_ids = np.unique(self.summary.partition)

        # Handle statistic using factory if necessary
        statistic_method = self._resolve_statistic_method(statistic)

        # Get signature of method to check for 'weights'
        func_signature = inspect.signature(statistic_method)
        has_weights = "weights" in func_signature.parameters

        for region_id in region_ids:
            region_mask = (self.summary.partition == region_id) & (mask == 1)
            window = x[region_mask]

            if window.size > 0:
                w = (
                    weights[region_mask]
                    if (weights is not None and has_weights)
                    else None
                )
                if w is not None:
                    w = w / w.max()

                if has_weights:
                    stat = statistic_method(
                        window.ravel(), weights=w, **kwargs
                    )
                else:
                    stat = statistic_method(window.ravel(), **kwargs)
            else:
                stat = np.nan

            local_description[region_mask] = stat

        self.local_description = local_description

    def _resolve_statistic_method(
        self, statistic: Union[str, Statistic]
    ) -> Callable:
        """
        Resolves the statistic method either from a string
        or directly if it is an instance of Statistic.

        Parameters
        ----------
        statistic : Union[str, Statistic]
            The statistic to compute.

        Returns
        -------
        Callable
            The resolved statistic method.

        Raises
        ------
        ValueError
            If the statistic is not a valid type.
        """
        if isinstance(statistic, str):
            return StatisticsFactory().get_method(statistic)
        elif isinstance(statistic, Statistic):
            return statistic
        else:
            raise ValueError(
                "statistic must be either a string representing the "
                "method name or an instance of Statistic."
            )
