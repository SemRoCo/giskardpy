from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.god_map_user import GodMap
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class LogTrajPlugin(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        current_js = deepcopy(GodMap.world.state)
        time = GodMap.god_map.get_data(identifier.time)
        trajectory = GodMap.god_map.get_data(identifier.trajectory)
        trajectory.set(time, current_js)
        GodMap.god_map.set_data(identifier.trajectory, trajectory)
        return Status.SUCCESS
