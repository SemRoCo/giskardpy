from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.evaluate_debug_expressions import EvaluateDebugExpressions
from giskardpy.tree.behaviors.evaluate_monitors import EvaluateMonitors
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.real_kinematic_sim import RealKinSimPlugin
from giskardpy.tree.behaviors.time import TimePlugin, RosTime, ControlCycleCounter
from giskardpy.tree.branches.check_monitors import CheckMonitors
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.send_controls import SendControls
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.decorators import success_is_running, failure_is_running
from giskardpy.utils.decorators import toggle_on, toggle_off


class ControlLoop(AsyncBehavior):
    publish_state: PublishState
    projection_synchronization: Synchronization
    closed_loop_synchronization: Synchronization
    check_monitors: CheckMonitors
    debug_added: bool = False
    in_projection: bool
    time: TimePlugin
    ros_time: RosTime
    kin_sim: KinSimPlugin
    real_kin_sim: RealKinSimPlugin
    send_controls: SendControls
    log_traj: LogTrajPlugin

    def __init__(self, name: str = 'control_loop', log_traj: bool = True):
        super().__init__(name)
        self.publish_state = success_is_running(PublishState)('publish state 2')
        self.projection_synchronization = success_is_running(Synchronization)()
        self.check_monitors = CheckMonitors()
        # projection plugins
        self.time = success_is_running(TimePlugin)()
        self.kin_sim = success_is_running(KinSimPlugin)('kin sim')

        self.ros_time = success_is_running(RosTime)()
        self.real_kin_sim = success_is_running(RealKinSimPlugin)('real kin sim')
        self.send_controls = success_is_running(SendControls)()
        self.closed_loop_synchronization = success_is_running(Synchronization)()

        self.add_child(failure_is_running(GoalCanceled)('goal canceled', god_map.giskard.action_server_name))

        if god_map.is_collision_checking_enabled():
            self.add_child(CollisionChecker('collision checker'))

        self.add_child(success_is_running(EvaluateMonitors)())
        self.add_child(self.check_monitors)
        self.add_child(ControllerPlugin('controller'))

        self.add_child(success_is_running(ControlCycleCounter)())

        self.log_traj = success_is_running(LogTrajPlugin)('add traj point')

        if log_traj:
            self.add_child(self.log_traj)
        self.add_child(self.publish_state)

    @toggle_on('in_projection')
    def switch_to_projection(self):
        self.remove_closed_loop_behaviors()
        self.add_projection_behaviors()

    @toggle_off('in_projection')
    def switch_to_closed_loop(self):
        assert god_map.is_closed_loop()
        self.remove_projection_behaviors()
        self.add_closed_loop_behaviors()

    def remove_projection_behaviors(self):
        self.remove_child(self.projection_synchronization)
        self.remove_child(self.time)
        self.remove_child(self.kin_sim)

    def remove_closed_loop_behaviors(self):
        self.remove_child(self.closed_loop_synchronization)
        self.remove_child(self.ros_time)
        self.remove_child(self.real_kin_sim)
        self.remove_child(self.send_controls)

    def add_projection_behaviors(self):
        self.insert_child(self.projection_synchronization, 1)
        self.insert_child(self.time, -2)
        self.insert_child(self.kin_sim, -2)
        self.in_projection = True

    def add_closed_loop_behaviors(self):
        self.insert_child(self.closed_loop_synchronization, 1)
        self.insert_child(self.ros_time, -2)
        self.insert_child(self.real_kin_sim, -2)
        self.insert_child(self.send_controls, -2)
        self.in_projection = False

    def add_evaluate_debug_expressions(self, log_traj: bool):
        if not self.debug_added:
            self.insert_child(EvaluateDebugExpressions(log_traj=log_traj), 3)
            self.debug_added = True
