from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import ClosestPointInfo, Trajectory
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import KeyDefaultDict


class CleanUp(GiskardBehavior):
    def __init__(self, name):
        super(CleanUp, self).__init__(name)
        # FIXME this is the smallest hack to reverse (some) update godmap changes, constraints need some kind of finalize
        self.general_options = deepcopy(self.get_god_map().safe_get_data(identifier.general_options))

    def initialise(self):
        self.get_god_map().safe_set_data(identifier.closest_point, {})
        # self.get_god_map().safe_set_data(identifier.closest_point, None)
        self.get_god_map().safe_set_data(identifier.time, 1)
        current_js = self.get_god_map().safe_get_data(identifier.joint_states)
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        self.get_god_map().safe_set_data(identifier.trajectory, trajectory)
        # to reverse update godmap changes
        self.get_god_map().safe_set_data(identifier.general_options, deepcopy(self.general_options))

    def update(self):
        return Status.SUCCESS
