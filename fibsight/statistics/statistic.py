from abc import ABC, abstractmethod

class Statistic(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Compute the statistic.

        Parameters
        ----------
        *args : tuple
            Positional arguments specific to the statistic method.
        **kwargs : dict
            Keyword arguments specific to the statistic method.

        Returns
        -------
        float
            Computed statistic.
        """
        pass