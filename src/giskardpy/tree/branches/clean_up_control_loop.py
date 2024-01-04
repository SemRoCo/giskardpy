from py_trees import Sequence

from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.delete_monitors_behaviors import DeleteMonitors
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.behaviors.plot_goal_gantt_chart import PlotGanttChart
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.tree.behaviors.reset_joint_state import ResetWorldState
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.utils.decorators import toggle_on, toggle_off


class CleanupControlLoop(Sequence):
    reset_world_state = ResetWorldState

    def __init__(self, name: str = 'clean up control loop'):
        super().__init__(name)
        self.add_child(TimePlugin())
        self.add_child(SetZeroVelocity('set zero vel 1'))
        self.add_child(LogTrajPlugin('log post processing'))
        self.add_child(GoalCleanUp('clean up goals'))
        self.add_child(DeleteMonitors())
        self.reset_world_state = ResetWorldState()
        self.remove_reset_world_state()

    def add_plot_trajectory(self, normalize_position: bool = False, wait: bool = False):
        self.insert_child(PlotTrajectory('plot trajectory', wait=wait, normalize_position=normalize_position),
                          index=-1)

    def add_plot_debug_trajectory(self, normalize_position: bool = False, wait: bool = False):
        self.add_child(PlotDebugExpressions('plot debug trajectory', wait=wait, normalize_position=normalize_position))

    def add_plot_gantt_chart(self):
        self.insert_child(PlotGanttChart(), 0)

    @toggle_on('has_reset_world_state')
    def add_reset_world_state(self):
        self.add_child(self.reset_world_state)

    @toggle_off('has_reset_world_state')
    def remove_reset_world_state(self):
        try:
            self.remove_child(self.reset_world_state)
        except ValueError as e:
            pass  # it's fine, happens if it's called before add
