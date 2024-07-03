from copy import deepcopy

from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.model.trajectory import Trajectory
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class NewTrajectory(GiskardBehavior):
    @record_time
    @profile
    def initialise(self):
        current_js = deepcopy(god_map.world.state)
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        god_map.trajectory = trajectory

    def update(self):
        return Status.SUCCESS
