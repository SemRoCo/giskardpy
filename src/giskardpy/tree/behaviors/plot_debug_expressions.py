import traceback

from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.utils.logging import logwarn
from giskardpy.utils.utils import plot_trajectory


class PlotDebugExpressions(PlotTrajectory):
    def __init__(self, name, enabled, wait=False, **kwargs):
        super(PlotDebugExpressions, self).__init__(name=name,
                                                   enabled=enabled,
                                                   velocity_threshold=None,
                                                   normalize_position=False,
                                                   wait=wait,
                                                   **kwargs)

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
                                file_name='debug.pdf',
                                **self.kwargs)
            except Exception:
                traceback.print_exc()
                logwarn('failed to save debug.pdf')

