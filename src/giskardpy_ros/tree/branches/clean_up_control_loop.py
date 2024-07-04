from py_trees import Sequence

from giskardpy_ros.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy_ros.tree.behaviors.delete_monitors_behaviors import DeleteMonitors
from giskardpy_ros.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy_ros.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy_ros.tree.behaviors.plot_debug_expressions import PlotDebugExpressions
from giskardpy_ros.tree.behaviors.plot_goal_gantt_chart import PlotGanttChart
from giskardpy_ros.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy_ros.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy_ros.tree.behaviors.reset_joint_state import ResetWorldState
from giskardpy_ros.tree.behaviors.time import TimePlugin
from giskardpy_ros.tree.decorators import failure_is_success
from giskardpy.utils.decorators import toggle_on, toggle_off


class CleanupControlLoop(Sequence):
    reset_world_state = ResetWorldState

    def __init__(self, name: str = 'clean up control loop'):
        super().__init__(name)
        self.add_child(PublishFeedback())
        self.add_child(TimePlugin())
        self.add_child(SetZeroVelocity('set zero vel 1'))
        self.add_child(LogTrajPlugin('log post processing'))
        self.add_child(GoalCleanUp('clean up goals'))
        self.add_child(DeleteMonitors())
        self.reset_world_state = failure_is_success(ResetWorldState)()
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
