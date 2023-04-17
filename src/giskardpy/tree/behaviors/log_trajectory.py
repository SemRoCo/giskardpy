from copy import deepcopy

from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class LogTrajPlugin(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        current_js = deepcopy(self.world.state)
        time = self.get_god_map().get_data(identifier.time)
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        trajectory.set(time, current_js)
        self.get_god_map().set_data(identifier.trajectory, trajectory)
        return Status.RUNNING
