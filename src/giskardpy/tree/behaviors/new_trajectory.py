from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.god_map_user import GodMap
from giskardpy.model.trajectory import Trajectory
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class NewTrajectory(GiskardBehavior):
    @record_time
    @profile
    def initialise(self):
        current_js = deepcopy(GodMap.get_world().state)
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        GodMap.god_map.set_data(identifier.trajectory, trajectory)

    def update(self):
        return Status.SUCCESS
