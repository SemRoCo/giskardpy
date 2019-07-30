from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import ClosestPointInfo, Trajectory
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import KeyDefaultDict


class CleanUp(GiskardBehavior):
    def __init__(self, name):
        super(CleanUp, self).__init__(name)

    def initialise(self):
        self.get_god_map().safe_set_data(identifier.closest_point, {})
        # self.get_god_map().safe_set_data(identifier.closest_point, None)
        self.get_god_map().safe_set_data(identifier.time, 1)
        current_js = self.get_god_map().safe_get_data(identifier.joint_states)
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        self.get_god_map().safe_set_data(identifier.trajectory, trajectory)

    def update(self):
        return Status.SUCCESS
