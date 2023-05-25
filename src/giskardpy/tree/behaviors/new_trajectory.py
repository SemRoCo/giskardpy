from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.model.trajectory import Trajectory
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class NewTrajectory(GiskardBehavior):
    @record_time
    @profile
    def initialise(self):
        current_js = deepcopy(self.god_map.get_data(identifier.joint_states))
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        self.god_map.set_data(identifier.trajectory, trajectory)
        trajectory = Trajectory()
        self.god_map.set_data(identifier.debug_trajectory, trajectory)

    def update(self):
        return Status.SUCCESS
