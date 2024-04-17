from collections import namedtuple
from typing import List, Union, Optional, Callable, Dict

import giskardpy.casadi_wrapper as cas
from giskardpy.god_map import god_map
from giskardpy.data_types import Derivatives, PrefixName


class WeightGain:
    _name: str

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return str(self._name)


class LinearWeightGain(WeightGain):
    def __init__(self,
                 name: str,
                 gains: Dict[str, cas.Expression]):
        super().__init__(name)
        self.gains = gains


class QuadraticWeightGain(WeightGain):
    def __init__(self,
                 name: str,
                 gains: Dict[str, cas.Expression]):
        super().__init__(name)
        self.gains = gains
