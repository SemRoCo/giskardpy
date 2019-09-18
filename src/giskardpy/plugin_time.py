from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import ClosestPointInfo, Trajectory
from giskardpy.identifier import closest_point
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import KeyDefaultDict


class TimePlugin(GiskardBehavior):
    # def __init__(self, name):
    #     super(TimePlugin, self).__init__(name)

    # def initialise(self):
    #     self.get_god_map().safe_set_data(closest_point, {})
    #     self.get_god_map().safe_set_data(identifier.time_identifier, -1)
    #     self.get_god_map().safe_set_data(identifier.trajectory_identifier, Trajectory())

    def update(self):
        with self.god_map:
            self.get_god_map().set_data(identifier.time, self.get_god_map().get_data(identifier.time) + 1)
        return Status.RUNNING
