from py_trees import Sequence

from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.evaluate_debug_expressions import EvaluateDebugExpressions
from giskardpy.tree.behaviors.evaluate_monitors import EvaluateMonitors
from giskardpy.tree.behaviors.goal_done import GoalDone
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.loop_detector import LoopDetector
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.behaviors.notify_state_change import NotifyStateChange
from giskardpy.tree.behaviors.real_kinematic_sim import RealKinSimPlugin
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.time_real import RosTime
from giskardpy.tree.branches.check_monitors import CheckMonitors
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.send_controls import SendControls
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import success_is_running


class ControlLoop(AsyncBehavior):
    publish_state: PublishState
    synchronization: Synchronization
    send_controls: SendControls
    check_monitors: CheckMonitors
    debug_added: bool = False

    def __init__(self, name: str = 'control_loop'):
        super().__init__(name)
        self.publish_state = success_is_running(PublishState)('publish state 2')
        self.synchronization = success_is_running(Synchronization)()
        self.check_monitors = CheckMonitors()

        self.add_child(self.synchronization)

        if god_map.is_collision_checking_enabled():
            self.add_child(CollisionChecker('collision checker'))

        self.add_child(success_is_running(EvaluateMonitors)())
        self.add_child(ControllerPlugin('controller'))

        if god_map.is_closed_loop():
            self.send_controls = success_is_running(SendControls)()

            self.add_child(success_is_running(RosTime)())
            self.add_child(success_is_running(RealKinSimPlugin)('kin sim'))
            self.add_child(self.send_controls)
        else:
            self.add_child(success_is_running(TimePlugin)('increase time closed loop'))
            self.add_child(success_is_running(KinSimPlugin)('kin sim'))
            self.add_child(success_is_running(LogTrajPlugin)('log closed loop control'))

        self.add_child(self.check_monitors)
        self.add_child(self.publish_state)

    def add_evaluate_debug_expressions(self):
        if not self.debug_added:
            self.insert_child(EvaluateDebugExpressions(), 1)
            self.debug_added = True
