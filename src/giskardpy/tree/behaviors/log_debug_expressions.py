import numpy as np
from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class LogDebugExpressionsPlugin(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)
        self.number_of_joints = len(self.world.controlled_joints)
        self.sample_period = self.god_map.get_data(identifier.sample_period)

    @profile
    def initialise(self):
        self.trajectory = self.god_map.get_data(identifier.debug_trajectory)

    @record_time
    @profile
    def update(self):
        debug_data = self.god_map.get_data(identifier.debug_expressions_evaluated)
        if len(debug_data) > 0:
            time = self.god_map.get_data(identifier.time) - 1
            last_mjs = None
            if time >= 1:
                last_mjs = self.trajectory.get_exact(time-1)
            js = JointStates()
            for name, value in debug_data.items():
                if last_mjs is not None:
                    velocity = value - last_mjs[name].position
                else:
                    if isinstance(value, np.ndarray):
                        velocity = np.zeros(value.shape)
                    else:
                        velocity = 0
                js[name].position = value
                js[name].velocity = velocity/self.sample_period
            self.trajectory.set(time, js)
        return Status.RUNNING
