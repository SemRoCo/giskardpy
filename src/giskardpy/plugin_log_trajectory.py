import numpy as np
import pylab as plt
from itertools import product

from giskardpy.identifier import trajectory_identifier, time_identifier, js_identifier
from giskardpy.plugin import PluginBase
from giskardpy.data_types import Trajectory


def diff(p1, p2):
    pass

class LogTrajPlugin(PluginBase):
    def __init__(self):
        """
        :type trajectory_identifier: str
        :type joint_state_identifier: str
        :type time_identifier: str
        """
        super(LogTrajPlugin, self).__init__()

    def setup(self):
        super(LogTrajPlugin, self).setup()

    def initialize(self):
        self.stop_universe = False
        self.past_joint_states = set()
        self.trajectory = Trajectory()
        self.god_map.safe_set_data([trajectory_identifier], self.trajectory)
        super(LogTrajPlugin, self).initialize()

    def update(self):
        current_js = self.god_map.safe_get_data([js_identifier])
        time = self.god_map.safe_get_data([time_identifier])
        trajectory = self.god_map.safe_get_data([trajectory_identifier])
        trajectory.set(time, current_js)
        self.god_map.safe_set_data([trajectory_identifier], trajectory)

        return super(LogTrajPlugin, self).update()


