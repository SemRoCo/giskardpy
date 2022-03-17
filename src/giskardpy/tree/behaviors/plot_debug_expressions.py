import traceback

from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.utils.logging import logwarn
from giskardpy.utils.utils import plot_trajectory


class PlotDebugExpressions(PlotTrajectory):
    def __init__(self, name, enabled, history, scaling, tick_stride, wait=False, order=2):
        super(PlotDebugExpressions, self).__init__(name=name,
                                                   enabled=enabled,
                                                   history=history,
                                                   velocity_threshold=None,
                                                   scaling=scaling,
                                                   normalize_position=False,
                                                   tick_stride=tick_stride,
                                                   wait=wait,
                                                   order=order)

    def plot(self):
        trajectory = self.get_god_map().get_data(identifier.debug_trajectory)
        if trajectory and len(trajectory.items()) > 0:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(tj=trajectory,
                                controlled_joints=controlled_joints,
                                path_to_data_folder=self.path_to_data_folder,
                                sample_period=sample_period,
                                order=self.order,
                                velocity_threshold=None,
                                scaling=self.scaling,
                                normalize_position=False,
                                tick_stride=self.tick_stride,
                                file_name='debug.pdf',
                                history=self.history)
            except Exception:
                traceback.print_exc()
                logwarn('failed to save debug.pdf')

