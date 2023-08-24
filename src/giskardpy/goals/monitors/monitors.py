import abc
from abc import ABC
import giskardpy.casadi_wrapper as cas


class MonitorInterface(ABC):
    @abc.abstractmethod
    def get_weight_expression(self) -> cas.Expression:
        ...


class Monitor(MonitorInterface):
    def __init__(self, expression):
        self.expression = expression

    def get_weight_expression(self):
        return self.expression


class SwitchMonitor(MonitorInterface):
    def __init__(self, expression):
        self.expression = expression

    def get_weight_expression(self):
        return self.expression


class AlwaysOne(SwitchMonitor):
    def __init__(self):
        super().__init__(cas.Expression(1))

class AlwaysZero(SwitchMonitor):
    def __init__(self):
        super().__init__(cas.Expression(0))

