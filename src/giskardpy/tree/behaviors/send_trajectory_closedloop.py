from typing import List
from copy import deepcopy
import control_msgs
from rospy import ROSException
from rostopic import ROSTopicException
from sensor_msgs.msg import JointState

from giskardpy.exceptions import ExecutionException, FollowJointTrajectory_INVALID_JOINTS, \
    FollowJointTrajectory_INVALID_GOAL, FollowJointTrajectory_OLD_HEADER_TIMESTAMP, \
    FollowJointTrajectory_PATH_TOLERANCE_VIOLATED, FollowJointTrajectory_GOAL_TOLERANCE_VIOLATED, \
    ExecutionTimeoutException, ExecutionSucceededPrematurely, ExecutionPreemptedException
from giskardpy.model.joints import OneDofJoint, OmniDrive
from giskardpy.my_types import PrefixName
from giskardpy.model.trajectory import Trajectory

try:
    import pr2_controllers_msgs.msg
except ImportError:
    pass
import py_trees
import rospy
import rostopic
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryResult
from py_trees_ros.actions import ActionClient
from giskardpy.my_types import Derivatives

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import raise_to_blackboard, \
    catch_and_raise_to_blackboard


class SendFollowJointTrajectoryClosedLoop(ActionClient, GiskardBehavior):
    error_code_to_str = {value: name for name, value in vars(FollowJointTrajectoryResult).items() if
                         isinstance(value, int)}

    try:
        supported_action_types = [control_msgs.msg.FollowJointTrajectoryAction,
                                  pr2_controllers_msgs.msg.JointTrajectoryAction]
        supported_state_types = [control_msgs.msg.JointTrajectoryControllerState,
                                 pr2_controllers_msgs.msg.JointTrajectoryControllerState]
    except NameError:
        supported_action_types = [control_msgs.msg.FollowJointTrajectoryAction]
        supported_state_types = [control_msgs.msg.JointTrajectoryControllerState]

    def __init__(self, action_namespace: str, state_topic: str, group_name: str,
                 goal_time_tolerance: float = 1, fill_velocity_values: bool = True):
        self.group_name = group_name
        self.action_namespace = action_namespace
        GiskardBehavior.__init__(self, str(self))
        self.fill_velocity_values = fill_velocity_values
        self.goal_time_tolerance = rospy.Duration(goal_time_tolerance)
        self.controlled_joints: List[OneDofJoint] = []
        self.joint_names = rospy.get_param('/hsrb/arm_trajectory_controller/joints')

        action_msg_type = None
        while not action_msg_type and not rospy.is_shutdown():
            try:
                action_msg_type, _, _ = rostopic.get_topic_class(f'{self.action_namespace}/goal')
                if action_msg_type is None:
                    raise ROSTopicException()
                try:
                    action_msg_type = eval(action_msg_type._type.replace('/', '.msg.')[:-4])
                    if action_msg_type not in self.supported_action_types:
                        raise TypeError()
                except Exception as e:
                    raise TypeError('Action server of type \'{}\' is not supported. '
                                    'Must be one of: {}'.format(action_msg_type, self.supported_action_types))
            except ROSTopicException as e:
                logging.logwarn('Couldn\'t connect to {}. Is it running?'.format(self.action_namespace))
                rospy.sleep(1)

        ActionClient.__init__(self, str(self), action_msg_type, None, self.action_namespace)
        loginfo(f'Successfully connected to \'{self.action_namespace}\'.')

        msg = None
        controlled_joint_names = []
        while not msg and not rospy.is_shutdown():
            try:
                status_msg_type, _, _ = rostopic.get_topic_class(state_topic)
                if status_msg_type is None:
                    raise ROSTopicException()
                if status_msg_type not in self.supported_state_types:
                    raise TypeError('State topic of type \'{}\' is not supported. '
                                    'Must be one of: {}'.format(status_msg_type, self.supported_state_types))
                msg = rospy.wait_for_message(state_topic, status_msg_type, timeout=2.0)
                if isinstance(msg, JointState):
                    controlled_joint_names = msg.name
                elif isinstance(msg, control_msgs.msg.JointTrajectoryControllerState) \
                        or isinstance(msg, pr2_controllers_msgs.msg.JointTrajectoryControllerState):
                    controlled_joint_names = msg.joint_names
            except (ROSException, ROSTopicException) as e:
                logging.logwarn(f'Couldn\'t connect to {state_topic}. Is it running?')
                rospy.sleep(1)
        controlled_joint_names = [PrefixName(j, self.group_name) for j in controlled_joint_names]
        if len(controlled_joint_names) == 0:
            raise ValueError(f'\'{state_topic}\' has no joints')

        for joint in self.world.joints.values():
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
            raise ValueError(f'{state_topic} provides the following joints '
                             f'that are not known to giskard: {controlled_joint_names}')
        self.world.register_controlled_joints(controlled_joint_names)
        controlled_joint_names = [j.name for j in self.controlled_joints]
        loginfo(f'Successfully connected to \'{state_topic}\'.')
        loginfo(f'Flagging the following joints as controlled: {controlled_joint_names}.')
        self.world.register_controlled_joints(controlled_joint_names)

    def update(self):
        try:
            qp_data = self.god_map.get_data(identifier.qp_solver_solution)
            dt = self.get_god_map().get_data(identifier.sample_period)
        except Exception:
            return
        goal = FollowJointTrajectoryGoal()
        traj = Trajectory()
        js = deepcopy(self.god_map.get_data(identifier.joint_states))
        try:
            for i in range(0, 9):
                for joint_name in self.joint_names:
                    key = self.world.joints['hsrb/' + joint_name].free_variables[0].position_name
                    velocity = qp_data[Derivatives.velocity][key]
                    js[joint_name].position += velocity * dt
                traj.set(i, js)
        except KeyError:
            print(KeyError)
            return py_trees.Status.RUNNING

        start_time = rospy.get_rostime() + rospy.Duration(0)
        fill_velocity_values = self.god_map.get_data(identifier.fill_trajectory_velocity_values)
        if fill_velocity_values is None:
            fill_velocity_values = self.fill_velocity_values
        goal.trajectory = traj.to_msg(dt, start_time, self.controlled_joints,
                                      fill_velocity_values)
        self.action_client.send_goal(goal)
        return py_trees.Status.RUNNING