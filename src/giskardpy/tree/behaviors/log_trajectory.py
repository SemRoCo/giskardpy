from copy import deepcopy

from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class LogTrajPlugin(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        current_js = deepcopy(god_map.world.state)
        god_map.trajectory.set(god_map.control_cycle_counter, current_js)
        return Status.SUCCESS
