from py_trees import Status

from giskardpy.identifier import trajectory_identifier, time_identifier, js_identifier
from giskardpy.data_types import Trajectory
from giskardpy.plugin import GiskardBehavior


class LogTrajPlugin(GiskardBehavior):
    def initialise(self):
        self.stop_universe = False
        self.past_joint_states = set()
        self.trajectory = Trajectory()
        self.get_god_map().safe_set_data(trajectory_identifier, self.trajectory)
        super(LogTrajPlugin, self).initialise()

    def update(self):
        current_js = self.get_god_map().safe_get_data(js_identifier)
        time = self.get_god_map().safe_get_data(time_identifier)
        trajectory = self.get_god_map().safe_get_data(trajectory_identifier)
        trajectory.set(time, current_js)
        self.get_god_map().safe_set_data(trajectory_identifier, trajectory)

        return Status.RUNNING


