from typing import Dict, Type

from .score import AlignScore, Statistic


class StatisticsFactory:
    def __init__(self):
        self.methods: Dict[str, Type[Statistic]] = {
            "align_score": AlignScore,
        }

    def get_method(self, method: str):
        method_class = self.methods.get(method.lower())
        if method_class is None:
            available_methods = ", ".join(self.methods.keys())
            raise ValueError(
                f"Method not recognised. Available methods: "
                f"{available_methods}"
            )
        return method_class()

    def register_method(self, name: str, method_class: Type[Statistic]):
        if not issubclass(method_class, Statistic):
            raise ValueError(
                "Registered method must be a subclass of `Statistic`."
            )
        self.methods[name.lower()] = method_class

    def __repr__(self):
        available_methods = ", ".join(self.methods.keys())
        return f"StatisticsFactory(" f"available_methods={available_methods})"
