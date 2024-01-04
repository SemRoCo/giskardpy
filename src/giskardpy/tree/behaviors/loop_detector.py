from collections import defaultdict

from py_trees import Status

from giskardpy.data_types import JointStates
from giskardpy.exceptions import ExecutionException
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class LoopDetector(GiskardBehavior):
    past_joint_states: set

    @profile
    def __init__(self, name, precision: int = 4):
        super().__init__(name)
        self.precision = 4
        self.window_size = 21

    @record_time
    @profile
    def initialise(self):
        super().initialise()
        self.past_joint_states = set()
        self.velocity_limits = defaultdict(lambda: 1.)
        self.velocity_limits.update(god_map.world.get_all_free_variable_velocity_limits())
        for name, threshold in self.velocity_limits.items():
            if threshold < 0.001:
                self.velocity_limits[name] = 0.001

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        current_js = god_map.world.state
        planning_time = god_map.time
        rounded_js = self.round_js(current_js)
        if planning_time >= self.window_size and rounded_js in self.past_joint_states:
            logging.loginfo('found loop, stopped planning.')
            run_time = self.get_runtime()
            msg = f'found goal trajectory with length {planning_time * god_map.qp_controller_config.sample_period:.3}s in {run_time:.3}s'
            logging.loginfo(msg)
            raise ExecutionException(msg)
        self.past_joint_states.add(rounded_js)
        return Status.RUNNING

    def round_js(self, js: JointStates) -> tuple:
        return tuple(round(state.position / self.velocity_limits[name], self.precision) for name, state in js.items())
