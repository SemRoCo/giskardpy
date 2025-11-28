import time
from dataclasses import field, dataclass
from typing import Optional

from giskardpy.motion_statechart.context import ExecutionContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode


@dataclass
class CheckMaxTrajectoryLength(MotionStatechartNode):
    length: float

    def __post_init__(self):
        self.observation_expression = context.time_symbol > self.length


@dataclass(eq=False, repr=False)
class Print(MotionStatechartNode):
    message: str = ""

    def on_tick(self, context: ExecutionContext) -> ObservationStateValues:
        print(self.message)
        return ObservationStateValues.TRUE


# @dataclass
# class Sleep(MotionStatechartNode):
#     seconds: float
#     start_time: Optional[float] = field(default=None, init=False)
#
#     def on_start(self, context: ExecutionContext):
#         self.start_time = None
#
#     def on_tick(self, context: ExecutionContext) -> Optional[float]:
#         if self.start_time is None:
#             self.start_time = god_map.time
#         return god_map.time - self.start_time >= self.seconds


@dataclass
class CountSeconds(MotionStatechartNode):
    """
    This node counts X seconds and then turns True.
    Only counts while in state RUNNING.
    """

    seconds: float = field(kw_only=True)
    _start_time: float = field(init=False)

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        difference = time.time() - self._start_time
        if difference >= self.seconds:
            return ObservationStateValues.TRUE
        return None

    def on_start(self, context: ExecutionContext):
        self._start_time = time.time()


@dataclass(repr=False, eq=False)
class CountTicks(MotionStatechartNode):
    """
    This node counts 'threshold'-many ticks and then turns True.
    Only counts while in state RUNNING.
    """

    ticks: int = field(kw_only=True)
    counter: int = field(init=False)

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        self.counter += 1
        if self.counter >= self.ticks:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_start(self, context: ExecutionContext):
        self.counter = 0


@dataclass
class Pulse(MotionStatechartNode):
    """
    Will stay True for a single tick, then turn False.
    """

    _triggered: bool = field(default=False, init=False)

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        if not self._triggered:
            self._triggered = True
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_reset(self, context: ExecutionContext):
        self._triggered = False
