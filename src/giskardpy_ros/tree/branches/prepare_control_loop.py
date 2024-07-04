from py_trees import Sequence

from giskardpy_ros.tree.behaviors.cleanup import CleanUpPlanning
from giskardpy_ros.tree.behaviors.compile_debug_expressions import CompileDebugExpressions
from giskardpy_ros.tree.behaviors.compile_monitors import CompileMonitors
from giskardpy_ros.tree.behaviors.init_qp_controller import InitQPController
from giskardpy_ros.tree.behaviors.new_trajectory import NewTrajectory
from giskardpy_ros.tree.behaviors.plot_task_graph import PlotTaskMonitorGraph
from giskardpy_ros.tree.behaviors.ros_msg_to_goal import ParseActionGoal, AddBaseTrajFollowerGoal, SetExecutionMode
from giskardpy_ros.tree.behaviors.set_tracking_start_time import SetTrackingStartTime


class PrepareControlLoop(Sequence):
    has_compile_debug_expressions: bool

    def __init__(self, name: str = 'prepare control loop'):
        super().__init__(name)
        self.has_compile_debug_expressions = False
        self.add_child(CleanUpPlanning('CleanUpPlanning'))
        self.add_child(NewTrajectory('NewTrajectory'))
        self.add_child(SetExecutionMode())
        self.add_child(ParseActionGoal('RosMsgToGoal'))
        self.add_child(InitQPController('InitQPController'))
        self.add_child(CompileMonitors())
        self.add_child(SetTrackingStartTime('start tracking time'))

    def add_plot_goal_graph(self):
        self.add_child(PlotTaskMonitorGraph())

    def add_compile_debug_expressions(self):
        if not self.has_compile_debug_expressions:
            self.add_child(CompileDebugExpressions())
            self.has_compile_debug_expressions = True


class PrepareBaseTrajControlLoop(Sequence):
    has_compile_debug_expressions: bool

    def __init__(self, name: str = 'prepare control loop'):
        super().__init__(name)
        self.has_compile_debug_expressions = False
        self.add_child(CleanUpPlanning('CleanUpPlanning'))
        self.add_child(AddBaseTrajFollowerGoal())
        self.add_child(InitQPController('InitQPController'))
        self.add_child(CompileMonitors(traj_tracking=True))
        self.add_child(SetTrackingStartTime('start tracking time'))

    def add_plot_goal_graph(self):
        self.add_child(PlotTaskMonitorGraph())

    def add_compile_debug_expressions(self):
        if not self.has_compile_debug_expressions:
            self.add_child(CompileDebugExpressions())
            self.has_compile_debug_expressions = True
