import traceback

from py_trees import Status

from giskardpy import identifier
from giskardpy.utils.logging import logwarn
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.utils import plot_trajectory


class PlotDebugExpressions(GiskardBehavior):
    def __init__(self, name, enabled, history, scaling, tick_stride, order=2):
        super(PlotDebugExpressions, self).__init__(name)
        self.order = order
        self.scaling = scaling
        self.history = history
        self.tick_stride = tick_stride
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    def update(self):
        trajectory = self.get_god_map().get_data(identifier.debug_trajectory)
        if trajectory and len(trajectory.items()) > 0:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(trajectory, controlled_joints, self.path_to_data_folder, sample_period, self.order,
                                None, self.scaling, False, self.tick_stride,
                                file_name='debug.pdf', history=self.history)
            except Exception:
                traceback.print_exc()
                logwarn('failed to save debug. pdf')
        return Status.SUCCESS
