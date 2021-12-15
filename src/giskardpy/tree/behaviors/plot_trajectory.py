from py_trees import Status

from giskardpy import identifier
from giskardpy.utils.logging import logwarn
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import plot_trajectory


class PlotTrajectory(GiskardBehavior):
    def __init__(self, name, enabled, history, velocity_threshold, scaling, normalize_position, tick_stride, order=4):
        super(PlotTrajectory, self).__init__(name)
        self.order = order
        self.history = history
        self.velocity_threshold = velocity_threshold
        self.scaling = scaling
        self.normalize_position = normalize_position
        self.tick_stride = tick_stride
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    def update(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        if trajectory:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            # controlled_joints = self.god_map.get_data(identifier.controlled_joints)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(trajectory, controlled_joints, self.path_to_data_folder, sample_period, self.order,
                                self.velocity_threshold, self.scaling, self.normalize_position, self.tick_stride,
                                history=self.history)
            except Exception as e:
                logwarn('failed to save trajectory pdf')
        return Status.SUCCESS
