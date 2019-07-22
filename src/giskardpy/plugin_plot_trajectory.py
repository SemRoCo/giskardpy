from py_trees import Status

from giskardpy import identifier
from giskardpy.data_types import ClosestPointInfo, Trajectory
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import KeyDefaultDict, plot_trajectory


class PlotTrajectory(GiskardBehavior):
    def initialise(self):
        self.path_to_data_folder = self.get_god_map().safe_get_data(identifier.data_folder)

    def update(self):
        trajectory = self.get_god_map().safe_get_data(identifier.trajectory)
        if trajectory:
            sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
            controlled_joints = self.get_robot().controlled_joints
            plot_trajectory(trajectory, controlled_joints, self.path_to_data_folder, sample_period)
        return Status.SUCCESS
