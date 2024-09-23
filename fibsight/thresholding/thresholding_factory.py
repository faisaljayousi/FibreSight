from typing import Type

from .thresholding_methods import (
    MeanThresholding,
    OtsuThresholding,
    ThresholdingMethod,
)


class ThresholdingFactory:
    """
    Factory class to retrieve and register thresholding methods.
    """

    def __init__(self):
        """Initialise factory with built-in thresholding methods."""
        self.methods = {
            "otsu": OtsuThresholding,
            "mean": MeanThresholding,
        }

    def get_method(self, method: str) -> ThresholdingMethod:
        """
        Retrieve an instance of the thresholding method.

        Parameters
        ----------
        method : str
            The name of the thresholding method to retrieve.

        Returns
        -------
        ThresholdingMethod
            An instance of the requested thresholding method.

        Raises
        ------
        ValueError
            If the method name is not recognised or registered.
        """
        method_class = self.methods.get(method.lower())
        if method_class is None:
            raise ValueError(
                f"Method not recognised. Available methods: "
                f"{list(self.methods.keys())}"
            )
        return method_class()

    def register_method(
        self, name: str, method_class: Type[ThresholdingMethod]
    ):
        """
        Register a new thresholding method with the factory.

        Parameters
        ----------

        """
        if not issubclass(method_class, ThresholdingMethod):
            raise ValueError(
                "Registered method must be a subclass of ThresholdingMethod"
            )
        self.methods[name.lower()] = method_class

    def __repr__(self) -> str:
        return (
            f"ThresholdingFactory("
            f"available_methods={list(self.methods.keys())})"
        )
