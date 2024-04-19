from collections import defaultdict
from typing import List, Union, Optional, Callable, Dict, DefaultDict

import giskardpy.casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.data_types import Derivatives, PrefixName
from giskardpy.qp.free_variable import FreeVariable


class WeightGain:
    _name: str
    gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]

    def __init__(self, name: str, gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        self._name = name
        self.gains = gains

    @property
    def name(self) -> str:
        return str(self._name)


class LinearWeightGain(WeightGain):
    def __init__(self,
                 name: str,
                 gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        super().__init__(name, gains)


class QuadraticWeightGain(WeightGain):
    def __init__(self,
                 name: str,
                 gains: List[DefaultDict[Derivatives, Dict[FreeVariable, float]]]):
        super().__init__(name, gains)
