from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.plugin import GiskardBehavior


class LogTrajPlugin(GiskardBehavior):
    @profile
    def update(self):
        current_js = self.world.state
        time = self.get_god_map().get_data(identifier.time)
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        trajectory.set(time, current_js)
        self.get_god_map().set_data(identifier.trajectory, trajectory)
        return Status.RUNNING
