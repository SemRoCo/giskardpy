import traceback

from py_trees import Status

from giskardpy import identifier
from giskardpy.logging import logwarn
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import plot_trajectory


class PlotlbA(GiskardBehavior):
    def __init__(self, name, order):
        super(PlotlbA, self).__init__(name)
        self.order = order
        # self.velocity_threshold = self.get_god_map().get_data(identifier.PlotTrajectory_velocity_threshold)
        self.scaling = self.get_god_map().get_data(identifier.PlotTrajectory_scaling)
        # self.normalize_position = self.get_god_map().get_data(identifier.PlotTrajectory_normalize_position)
        self.tick_stride = self.get_god_map().get_data(identifier.PlotTrajectory_tick_stride)
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    def update(self):
        trajectory = self.get_god_map().get_data(identifier.lbA_trajectory)
        if trajectory and len(trajectory.items()) > 0:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(trajectory, controlled_joints, self.path_to_data_folder, sample_period, self.order,
                                None, self.scaling, False, self.tick_stride,
                                file_name='lbA.pdf')
            except Exception:
                traceback.print_exc()
                logwarn(u'failed to save trajectory pdf')
        return Status.SUCCESS
