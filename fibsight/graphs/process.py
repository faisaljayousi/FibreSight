import logging
from typing import Callable, Dict, Optional, Union

import numpy as np
import skl_graph.map as mp_
from scipy.spatial import Voronoi
from skl_graph.graph import skl_graph_t

import fibsight as fs
from fibsight.skeletonisation.skel_factory import (
    ParallelThinning,
    SkeletonisationFactory,
    SkeletonisationMethod,
)
from fibsight.thresholding.thresholding_factory import ThresholdingFactory

from .prune import prune_skeleton


class ImageToGraph:
    def __init__(self, image: np.ndarray, verbose: int = 0):
        self.image = image
        self.image_copy = np.copy(image)
        self.verbose = verbose

        self.binary = None  # change to ones?
        self.skl_graph = None
        self.skl_width_map = None
        self.noise_mask = np.ones_like(image)
        self.counter_nodes = 0
        self.degMap = None

        self.thresholding_factory = ThresholdingFactory()
        self.partition_methods: Dict[str, Callable] = {}

        # Configure logger
        self.logger = logging.getLogger(__name__)
        fs.set_logging_level(self.verbose)

    def __repr__(self):
        # TODO
        # Attributes and description
        return

    def make_binary(self, config):
        if self.image is None:
            self.logger.error(
                "Image not set. Please set the image before thresholding."
            )
            return

        # Use the binary method from the GraphConfig
        method_to_use = config.get_binary_method()

        if method_to_use not in self.thresholding_factory.methods:
            # If the method is not defined in the factory,
            # check if a callable is provided
            if config.method_callable:
                try:
                    self.register_thresholding_method(
                        method_to_use, config.method_callable
                    )
                    self.logger.info(
                        f"Registered new thresholding method: "
                        f"'{method_to_use}'"
                    )
                except ValueError as e:
                    self.logger.error(
                        f"Failed to register thresholding method: {e}"
                    )
                    return
            else:
                self.logger.error(
                    f"Thresholding method '{method_to_use}' "
                    "not found and no callable provided."
                )
                return

        try:
            strategy = self.thresholding_factory.get_method(method_to_use)
            self.binary = strategy(self.image)
            self.logger.info(
                f"Binary image created using the '{method_to_use}' method."
            )
        except ValueError as e:
            self.logger.error(
                f"Error in applying thresholding method "
                f"'{method_to_use}': {e}"
            )

    def make_skeleton(
        self,
        method: Optional[Union[str, SkeletonisationMethod]] = None,
        prune: bool = False,
        min_length: int = 20,
        max_iter: int = 5,
    ):
        if self.binary is None:
            raise ValueError(
                "Binary image not set. "
                "Please call 'make_binary()' before skeletonisation."
            )

        # Handle different cases
        if isinstance(method, SkeletonisationMethod):
            skeleton_func = method
        elif isinstance(method, str):
            factory = SkeletonisationFactory()
            skeleton_func = factory.get_method(method)
        else:
            raise ValueError(
                f"Invalid method type. Expected a string, callable, "
                f"or an instance of SkeletonisationMethod, "
                f"but got {type(method)}."
            )

        result = skeleton_func(self.binary)

        # Handle width map (where applicable)
        if isinstance(skeleton_func, ParallelThinning) or (
            callable(method) and "with_width" in method.__code__.co_varnames
        ):
            self.skl_map, self.skl_width_map = result
        else:
            self.skl_map = result

        if prune:
            self.skl_map = prune_skeleton(
                self.skl_map, min_length=min_length, max_iter=max_iter
            )

    def make_graph(self):
        if mp_.SKLMapFromThickVersion(self.skl_map, should_only_check=True):
            self.skl_graph = skl_graph_t.NewFromSKLMap(self.skl_map)
        else:
            mp_.SKLMapFromThickVersion(self.skl_map, in_place=True)
            self.skl_graph = skl_graph_t.NewFromSKLMap(self.skl_map)

    def degreeMap(self):
        self.degMap = np.zeros_like(self.image, dtype=float)
        degs = np.array(self.skl_graph.degree)
        coords, deg = zip(*degs)
        self._coords = self.skl_to_nx_coordinates(coords)
        self.degMap[self._coords[:, 1], self._coords[:, 0]] = deg

    def voronoi(self):
        coords = self._coords
        vor = Voronoi(coords)
        self.partition = self._voronoi_to_array(vor, self.image.shape)

    def apply_partition_method(self, method_name: str):
        if method_name.lower() == "voronoi":
            self.voronoi()
        elif method_name in self.partition_methods:
            method = self.partition_methods[method_name]
            method(self)
        else:
            raise ValueError(f"Partition method '{method_name}' not found.")

    @staticmethod
    def skl_to_nx_coordinates(coordinates):
        coordinates = [list(map(int, n.split("-"))) for n in coordinates]
        coordinates = np.flip(np.array(coordinates), axis=1)
        return coordinates

    @staticmethod
    def _voronoi_to_array(vor, image_shape) -> np.ndarray:
        def get_region_of_pixel(voronoi, x, y):
            # Find the index of the nearest point to the given pixel
            index = voronoi.point_region[
                np.argmin(np.sum((voronoi.points - [x, y]) ** 2, axis=1))
            ]
            return index

        labels_image = np.zeros(image_shape, dtype=np.int32)

        for i in range(labels_image.shape[0]):
            for j in range(labels_image.shape[1]):
                labels_image[i, j] = get_region_of_pixel(vor, j, i)

        return labels_image

    def register_partition_method(self, name: str, method: Callable):
        if not callable(method):
            raise ValueError("The method needs to be callable3")
        self.partition_methods[name] = method

    def register_thresholding_method(self, method_name, method_callable):
        self.thresholding_factory.register_method(method_name, method_callable)
