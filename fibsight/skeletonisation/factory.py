from typing import Dict, Type

from .skeleton_methods import (
    LeeSkeleton,
    MedialAxisSkeleton,
    ParallelThinning,
    SkeletonisationMethod,
    ZhangSkeleton,
)


class SkeletonisationFactory:
    """
    Factory class to manage and retrieve skeletonisation methods.

    This class provides functionality to retrieve predefined skeletonisation
    methods by name and to register new ones. It allows for dynamic selection
    and extension of skeletonisation methods.
    """

    def __init__(self):
        """
        Initialise factory with built-in skeletonisation methods.

        Built-in methods include:
            - Parallel Thinning
            - Lee's method
            - Zhang's method
            - Medial Axis
        """
        self.methods: Dict[str, Type[SkeletonisationMethod]] = {
            "parallel_thinning": ParallelThinning,
            "lee": LeeSkeleton,
            "zhang": ZhangSkeleton,
            "medial_axis": MedialAxisSkeleton,
        }

    def get_method(self, method: str) -> SkeletonisationMethod:
        """
        Retrieve an instance of the skeletonisation method.

        Parameters
        ----------
        method : str
            The name of the skeletonisation method to retrieve.

        Returns
        -------
        SkeletonisationMethod
            An instance of the requested skeletonisation method.

        Raises
        ------
        ValueError
            If the method is not recognised.
        """
        method_class = self.methods.get(method.lower())
        if method_class is None:
            raise ValueError(
                f"Method not recognised. Available methods: "
                f"{list(self.methods.keys())}"
            )
        return method_class()

    def register_method(
        self, name: str, method_class: Type[SkeletonisationMethod]
    ):
        """
        Register a new skeletonisation method.

        Parameters
        ----------
        name : str
            The name under which to register the method. This name will be
            used to retrieve the method later.
        method_class : Type[SkeletonisationMethod]
            A class implementing the skeletonisation method. Must be a subclass
            of SkeletonisationMethod.

        Raises
        ------
        ValueError
            If the method_class is not a subclass of SkeletonisationMethod.
        """
        if not issubclass(method_class, SkeletonisationMethod):
            raise ValueError(
                "Registered method must be a subclass of SkeletonisationMethod"
            )
        self.methods[name.lower()] = method_class

    def __repr__(self) -> str:
        """
        Provides a summary of the available skeletonisation methods.

        Returns
        -------
        str
            A string representation of the SkeletonisationFactory instance,
            listing all available methods.
        """
        available_methods = ", ".join(self.methods.keys())
        return f"SkeletonisationFactory(available_methods={available_methods})"
