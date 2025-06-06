from typing import List, Dict, Any

import control_msgs
from actionlib import SimpleActionClient
from rospy import ROSException
from rostopic import ROSTopicException
from sensor_msgs.msg import JointState

from giskardpy.exceptions import ExecutionException, FollowJointTrajectory_INVALID_JOINTS, \
    FollowJointTrajectory_INVALID_GOAL, FollowJointTrajectory_OLD_HEADER_TIMESTAMP, \
    FollowJointTrajectory_PATH_TOLERANCE_VIOLATED, FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED, \
    ExecutionTimeoutException, ExecutionSucceededPrematurely, ExecutionPreemptedException
from giskardpy.god_map import god_map
from giskardpy.model.joints import OneDofJoint, OmniDrive
from giskardpy.data_types import PrefixName, Derivatives

try:
    import pr2_controllers_msgs.msg
except ImportError:
    pass
import py_trees
import rospy
import rostopic
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryResult, \
    JointTolerance
from py_trees_ros.actions import ActionClient

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import raise_to_blackboard, wait_for_topic_to_appear
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class SendFollowJointTrajectory(ActionClient, GiskardBehavior):
    error_code_to_str = {value: name for name, value in vars(FollowJointTrajectoryResult).items() if
                         isinstance(value, int)}

    try:
        supported_action_types = [control_msgs.msg.FollowJointTrajectoryActionGoal,
                                  pr2_controllers_msgs.msg.JointTrajectoryActionGoal]
        supported_state_types = [control_msgs.msg.JointTrajectoryControllerState,
                                 pr2_controllers_msgs.msg.JointTrajectoryControllerState]
    except NameError:
        supported_action_types = [control_msgs.msg.FollowJointTrajectoryActionGoal]
        supported_state_types = [control_msgs.msg.JointTrajectoryControllerState]

    action_client: SimpleActionClient

    @record_time
    @profile
    def __init__(self, namespace: str, group_name: str,
                 goal_time_tolerance: float = 1, fill_velocity_values: bool = True,
                 path_tolerance: Dict[Derivatives, float] = None):
        self.group_name = group_name
        self.delay = rospy.Duration(0)
        self.namespace = namespace
        self.action_namespace = f'{self.namespace}/follow_joint_trajectory'
        GiskardBehavior.__init__(self, str(self))
        self.min_deadline: rospy.Time
        self.max_deadline: rospy.Time
        self.controlled_joints: List[OneDofJoint] = []
        self.fill_velocity_values = fill_velocity_values
        self.goal_time_tolerance = rospy.Duration(goal_time_tolerance)
        self.path_tolerance = path_tolerance

        params: Dict[str, Any] = rospy.get_param(self.namespace)

        actual_type = wait_for_topic_to_appear(self.action_namespace + '/goal', self.supported_action_types)
        action_type = eval(actual_type._type.replace('/', '.msg.')[:-4])

        ActionClient.__init__(self, str(self), action_type, None, self.action_namespace)

        controlled_joint_names = [PrefixName(j, self.group_name) for j in params['joints']]
        if len(controlled_joint_names) == 0:
            raise ValueError(f'\'{self.action_namespace}\' has no joints')

        for joint in god_map.world.joints.values():
            if isinstance(joint, OneDofJoint):
                if joint.free_variable.name in controlled_joint_names:
                    self.controlled_joints.append(joint)
                    controlled_joint_names.remove(joint.free_variable.name)
            elif isinstance(joint, OmniDrive):
                degrees_of_freedom = {joint.x.name, joint.y.name, joint.yaw.name}
                if set(controlled_joint_names) == degrees_of_freedom:
                    self.controlled_joints.append(joint)
                    for position_variable in degrees_of_freedom:
                        controlled_joint_names.remove(position_variable)
        if len(controlled_joint_names) > 0:
            raise ValueError(f'{self.action_namespace} provides the following joints '
                             f'that are not known to giskard: {controlled_joint_names}')
        god_map.world.register_controlled_joints(controlled_joint_names)
        controlled_joint_names = [j.name for j in self.controlled_joints]
        loginfo(f'Successfully connected to \'{self.action_namespace}\'.')
        loginfo(f'Flagging the following joints as controlled: {controlled_joint_names}.')
        god_map.world.register_controlled_joints(controlled_joint_names)

    @record_time
    @profile
    def initialise(self):
        super().initialise()
        self.delay = god_map.time_delay
        trajectory = god_map.trajectory
        goal = FollowJointTrajectoryGoal()
        start_time = god_map.tracking_start_time
        fill_velocity_values = god_map.fill_trajectory_velocity_values
        if fill_velocity_values is None:
            fill_velocity_values = self.fill_velocity_values
        goal.trajectory = trajectory.to_msg(god_map.qp_controller_config.sample_period,
                                            start_time,
                                            self.controlled_joints,
                                            fill_velocity_values)

        if self.path_tolerance is not None:
            for i, joint_name in enumerate(goal.trajectory.joint_names):
                jt = JointTolerance()
                jt.name = joint_name
                jt.position = self.path_tolerance[Derivatives.position]
                jt.velocity = self.path_tolerance[Derivatives.velocity]
                jt.acceleration = self.path_tolerance[Derivatives.acceleration]
                goal.path_tolerance.append(jt)

        self.action_goal = goal
        deadline = self.action_goal.trajectory.header.stamp + \
                   self.action_goal.trajectory.points[-1].time_from_start + \
                   self.action_goal.goal_time_tolerance
        self.min_deadline = deadline - self.goal_time_tolerance + self.delay
        self.max_deadline = deadline + self.goal_time_tolerance + self.delay
        self.cancel_tries = 0

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        """
        Check only to see whether the underlying action server has
        succeeded, is running, or has cancelled/aborted for some reason and
        map these to the usual behaviour return states.

        overriding this shit because of the fucking prints
        """
        current_time = rospy.get_rostime()
        # self.logger.debug("{0}.update()".format(self.__class__.__name__))
        if not self.action_client:
            self.feedback_message = 'no action client, did you call setup() on your tree?'
            return py_trees.Status.INVALID
        # pity there is no 'is_connected' api like there is for c++
        if not self.sent_goal:
            self.action_client.send_goal(self.action_goal)
            logging.loginfo(f'Sending trajectory to \'{self.action_namespace}\'.')
            self.sent_goal = True
            self.feedback_message = "sent goal to the action server"
            return py_trees.Status.RUNNING
        if self.action_client.get_state() in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
            result = self.action_client.get_result()
            self.feedback_message = self.error_code_to_str[result.error_code]
            msg = f'\'{self.namespace}\' failed to execute goal. ' \
                  f'Error: \'{self.error_code_to_str[result.error_code]}\''
            logging.logerr(msg)
            if result.error_code == FollowJointTrajectoryResult.INVALID_GOAL:
                e = FollowJointTrajectory_INVALID_GOAL(msg)
            elif result.error_code == FollowJointTrajectoryResult.INVALID_JOINTS:
                e = FollowJointTrajectory_INVALID_JOINTS(msg)
            elif result.error_code == FollowJointTrajectoryResult.OLD_HEADER_TIMESTAMP:
                e = FollowJointTrajectory_OLD_HEADER_TIMESTAMP(msg)
            elif result.error_code == FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED:
                e = FollowJointTrajectory_PATH_TOLERANCE_VIOLATED(msg)
            elif result.error_code == FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED:
                e = FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED(msg)
            else:
                e = ExecutionException(msg)
            raise_to_blackboard(e)
            return py_trees.Status.FAILURE
        if self.action_client.get_state() in [GoalStatus.PREEMPTED, GoalStatus.PREEMPTING]:
            if rospy.get_rostime() > self.max_deadline:
                msg = f'\'{self.namespace}\' preempted, ' \
                      'probably because it took to long to execute the goal.'
                raise_to_blackboard(ExecutionTimeoutException(msg))
            else:
                msg = f'\'{self.namespace}\' preempted. Stopping execution.'
                raise_to_blackboard(ExecutionPreemptedException(msg))
            logging.logerr(msg)
            return py_trees.Status.FAILURE

        result = self.action_client.get_result()
        if result:
            if current_time < self.min_deadline:
                msg = f'\'{self.namespace}\' executed too quickly, stopping execution.'
                e = ExecutionSucceededPrematurely(msg)
                raise_to_blackboard(e)
                return py_trees.Status.FAILURE
            self.feedback_message = "goal reached"
            logging.loginfo(f'\'{self.namespace}\' successfully executed the trajectory.')
            return py_trees.Status.SUCCESS

        if current_time > self.max_deadline:
            self.action_client.cancel_goal()
            msg = f'Cancelling \'{self.namespace}\' because it took to long to execute the goal.'
            logging.logerr(msg)
            self.cancel_tries += 1
            if self.cancel_tries > 5:
                logging.logwarn(f'\'{self.namespace}\' didn\'t cancel execution after 5 tries.')
                raise_to_blackboard(ExecutionTimeoutException(msg))
                return py_trees.Status.FAILURE
            return py_trees.Status.RUNNING

        self.feedback_message = "moving"
        return py_trees.Status.RUNNING

    def terminate(self, new_status):
        """
        If running and the current goal has not already succeeded, cancel it.

        Args:
            new_status (:class:`~py_trees.common.Status`): the behaviour is transitioning to this new status
        """
        # self.logger.debug("%s.terminate(%s)" % (self.__class__.__name__, "%s->%s" % (
        # self.status, new_status) if self.status != new_status else "%s" % new_status))
        if self.action_client is not None and self.sent_goal:
            motion_state = self.action_client.get_state()
            if ((motion_state == GoalStatus.PENDING) or (motion_state == GoalStatus.ACTIVE) or
                    (motion_state == GoalStatus.PREEMPTING) or (motion_state == GoalStatus.RECALLING)):
                logging.logwarn('Cancelling \'{}\''.format(self.namespace))
                self.action_client.cancel_goal()
        self.sent_goal = False

    def __str__(self):
        return f'{self.__class__.__name__}\n/{self.group_name}/{self.namespace}'
