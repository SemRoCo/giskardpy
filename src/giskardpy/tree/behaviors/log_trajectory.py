from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class LogTrajPlugin(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        current_js = deepcopy(god_map.world.state)
        time = god_map.time
        trajectory = god_map.trajectory
        trajectory.set(time, current_js)
        god_map.set_data(identifier.trajectory, trajectory)
        return Status.SUCCESS
