"""
This file initialises the FibreSight library, including version management and
logging configuration. The `set_logging_level` function allows for
customisation of the logging verbosity.

Author: Faisal Jayousi
Email: fjayousi90@gmail.com
"""

import logging

from fibsight.pipeline.characterise import FibreDescriptor, ImageToGraph
from fibsight.thresholding.thresholding_factory import (
    ThresholdingFactory,
    ThresholdingMethod,
)

from ._version import __version__

__all__ = [
    "__version__",
    "FibreDescriptor",
    "ImageToGraph",
    "ThresholdingFactory",
    "ThresholdingMethod",
]

# Configure logger
# Should only be called once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def set_logging_level(verbose: int):
    """
    Set the logging level based on the provided verbosity.

    Parameters:
    -----------
    verbose : int
        Verbosity level:
        - 0: Set logging level to WARNING.
        - 1: Set logging level to INFO.
        - 2: Set logging level to DEBUG.
        - >2: Set logging level to DEBUG.

    Raises:
    -------
    ValueError
        If the verbosity level is negative.
    """
    try:
        if verbose < 0:
            raise ValueError("Verbosity level must be non-negative.")
        elif verbose > 1:
            logger.setLevel(logging.DEBUG)
        elif verbose == 0:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
    except ValueError as e:
        logger.error(f"Error setting logging level: {e}")
        raise
