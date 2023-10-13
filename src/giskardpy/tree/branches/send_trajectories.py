from typing import Dict

from py_trees import Sequence

from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.my_types import PrefixName, Derivatives
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
from giskardpy.tree.behaviors.send_trajectory import SendFollowJointTrajectory
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.time_real import RosTime
from giskardpy.tree.branches.check_monitors import CheckMonitors
from giskardpy.tree.branches.control_loop import ControlLoop
from giskardpy.tree.branches.prepare_control_loop import PrepareControlLoop, PrepareBaseTrajControlLoop
from giskardpy.tree.branches.process_goal import ProcessGoal
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.send_controls import SendControls
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.composites.better_parallel import Parallel, ParallelPolicy
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import success_is_running
from giskardpy.utils.decorators import catch_and_raise_to_blackboard


class ExecuteTraj(Sequence):
    base_closed_loop: ControlLoop
    prepare_base_control: PrepareBaseTrajControlLoop
    move_robots: Parallel

    def __init__(self, name: str = 'execute traj'):
        super().__init__(name)
        self.move_robots = Parallel(name='move robot', policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        self.add_child(self.move_robots)
        self.prepare_base_control = PrepareBaseTrajControlLoop()
        self.insert_child(self.prepare_base_control, 0)

        self.base_closed_loop = ControlLoop()
        self.move_robots.add_child(self.base_closed_loop)

    def add_follow_joint_traj_action_server(self, namespace: str, state_topic: str, group_name: str,
                                            fill_velocity_values: bool,
                                            path_tolerance: Dict[Derivatives, float] = None):
        behavior = SendFollowJointTrajectory(action_namespace=namespace, state_topic=state_topic, group_name=group_name,
                                             fill_velocity_values=fill_velocity_values, path_tolerance=path_tolerance)
        self.move_robots.add_child(behavior)

    def add_base_traj_action_server(self, cmd_vel_topic: str, track_only_velocity: bool = False,
                                    joint_name: PrefixName = None):
        self.base_closed_loop.send_controls.add_send_cmd_velocity(cmd_vel_topic=cmd_vel_topic,
                                                                  joint_name=joint_name)
        # todo handle if this is called twice
        # self.insert_node_behind_node_of_type(self.execution_name, SetTrackingStartTime,
        #                                      CleanUpBaseController('CleanUpBaseController', clear_markers=False))
        # self.insert_node_behind_node_of_type(self.execution_name, SetTrackingStartTime,
        #                                      InitQPController('InitQPController for base'))
        # self.insert_node_behind_node_of_type(self.execution_name, SetTrackingStartTime,
        #                                      SetDriveGoals('SetupBaseTrajConstraints'))

        # real_time_tracking = AsyncBehavior(self.base_closed_loop_control_name)
        # self.insert_node(real_time_tracking, self.move_robots_name)
        # sync_tf_nodes = self.get_nodes_of_type(SyncTfFrames)
        # for node in sync_tf_nodes:
        #     self.insert_node(success_is_running(SyncTfFrames)(node.name + '*', node.joint_map),
        #                      self.base_closed_loop_control_name)
        # odom_nodes = self.get_nodes_of_type(SyncOdometry)
        # for node in odom_nodes:
        #     new_node = success_is_running(SyncOdometry)(odometry_topic=node.odometry_topic,
        #                                                 joint_name=node.joint_name,
        #                                                 name_suffix='*')
        #     self.insert_node(new_node, self.base_closed_loop_control_name)
        # self.insert_node(RosTime('time'), self.base_closed_loop_control_name)
        # self.insert_node(ControllerPlugin('base controller'), self.base_closed_loop_control_name)
        # self.insert_node(RealKinSimPlugin('base kin sim'), self.base_closed_loop_control_name)
        # # todo debugging
        # # if self.god_map.get_data(identifier.PlotDebugTF_enabled):
        # #     real_time_tracking.add_child(DebugMarkerPublisher('debug marker publisher'))
        # # if self.god_map.unsafe_get_data(identifier.PublishDebugExpressions)['enabled_base']:
        # #     real_time_tracking.add_child(PublishDebugExpressions('PublishDebugExpressions',
        # #                                                          **self.god_map.unsafe_get_data(
        # #                                                              identifier.PublishDebugExpressions)))
        # # if self.god_map.unsafe_get_data(identifier.PlotDebugTF)['enabled_base']:
        # #     real_time_tracking.add_child(DebugMarkerPublisher('debug marker publisher',
        # #                                                       **self.god_map.unsafe_get_data(
        # #                                                           identifier.PlotDebugTF)))
        #
        # self.insert_node(SendTrajectoryToCmdVel(cmd_vel_topic=cmd_vel_topic,
        #                                         track_only_velocity=track_only_velocity,
        #                                         joint_name=joint_name), self.base_closed_loop_control_name)
