from collections import defaultdict

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


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
        self.velocity_limits.update(self.world.get_all_free_variable_velocity_limits())
        for name, threshold in self.velocity_limits.items():
            if threshold < 0.001:
                self.velocity_limits[name] = 0.001

    @record_time
    @profile
    def update(self):
        current_js = self.get_god_map().get_data(identifier.joint_states)
        planning_time = self.get_god_map().get_data(identifier.time)
        rounded_js = self.round_js(current_js)
        if planning_time >= self.window_size and rounded_js in self.past_joint_states:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            logging.loginfo('found loop, stopped planning.')
            run_time = self.get_runtime()
            logging.loginfo('found goal trajectory with length {:.3f}s in {:.3f}s'.format(planning_time * sample_period,
                                                                                          run_time))
            return Status.SUCCESS
        self.past_joint_states.add(rounded_js)
        return Status.RUNNING

    def round_js(self, js: JointStates) -> tuple:
        return tuple(round(state.position / self.velocity_limits[name], self.precision) for name, state in js.items())
