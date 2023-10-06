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
        time = god_map.time
        god_map.trajectory.set(time, current_js)
        return Status.SUCCESS
