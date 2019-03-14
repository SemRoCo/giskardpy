from giskardpy.identifier import trajectory_identifier, time_identifier, js_identifier
from giskardpy.plugin import PluginBase
from giskardpy.data_types import Trajectory


class LogTrajPlugin(PluginBase):
    def initialize(self):
        self.stop_universe = False
        self.past_joint_states = set()
        self.trajectory = Trajectory()
        self.get_god_map().safe_set_data(trajectory_identifier, self.trajectory)
        super(LogTrajPlugin, self).initialize()

    def update(self):
        current_js = self.get_god_map().safe_get_data(js_identifier)
        time = self.get_god_map().safe_get_data(time_identifier)
        trajectory = self.get_god_map().safe_get_data(trajectory_identifier)
        trajectory.set(time, current_js)
        self.get_god_map().safe_set_data(trajectory_identifier, trajectory)

        return super(LogTrajPlugin, self).update()


