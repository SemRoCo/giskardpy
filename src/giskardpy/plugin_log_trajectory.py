import numpy as np
import pylab as plt
from itertools import product
from giskardpy.plugin import NewPluginBase
from giskardpy.data_types import Trajectory


def diff(p1, p2):
    pass

class NewLogTrajPlugin(NewPluginBase):
    def __init__(self, trajectory_identifier, joint_state_identifier, time_identifier):
        """
        :type trajectory_identifier: str
        :type joint_state_identifier: str
        :type time_identifier: str
        """
        self.trajectory_identifier = trajectory_identifier
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        super(NewLogTrajPlugin, self).__init__()

    def setup(self):
        super(NewLogTrajPlugin, self).setup()

    def initialize(self):
        self.stop_universe = False
        self.past_joint_states = set()
        self.trajectory = Trajectory()
        self.god_map.safe_set_data([self.trajectory_identifier], self.trajectory)
        super(NewLogTrajPlugin, self).initialize()

    def update(self):
        current_js = self.god_map.safe_get_data([self.joint_state_identifier])
        time = self.god_map.safe_get_data([self.time_identifier])
        trajectory = self.god_map.safe_get_data([self.trajectory_identifier])
        trajectory.set(time, current_js)
        self.god_map.safe_set_data([self.trajectory_identifier], trajectory)

        return super(NewLogTrajPlugin, self).update()


