from abc import ABC
from typing import List, Optional

import numpy as np
import rospy
import rostopic
from geometry_msgs.msg import Twist
from py_trees import Status
from rospy import ROSException
from rostopic import ROSTopicException

import giskardpy.identifier as identifier
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.goals.goal import Goal
from giskardpy.goals.set_prediction_horizon import SetPredictionHorizon
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.my_types import Derivatives
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import catch_and_raise_to_blackboard
from giskardpy.utils.logging import loginfo


# can be used during closed-loop control, instead of for tracking a trajectory
class SendCmdVel(GiskardBehavior, ABC):
    supported_state_types = [Twist]

    @profile
    def __init__(self, cmd_vel_topic: str, goal_time_tolerance: float = 1, track_only_velocity: bool = False,
                 joint_name: Optional[str] = None):
        self.cmd_vel_topic = cmd_vel_topic
        super().__init__(str(self))
        self.threshold = np.array([0.0, 0.0, 0.0])
        self.goal_time_tolerance = rospy.Duration(goal_time_tolerance)
        self.track_only_velocity = track_only_velocity

        loginfo(f'Waiting for cmd_vel topic \'{self.cmd_vel_topic}\' to appear.')
        try:
            msg_type, _, _ = rostopic.get_topic_class(self.cmd_vel_topic)
            if msg_type is None:
                raise ROSTopicException()
            if msg_type not in self.supported_state_types:
                raise TypeError(f'Cmd_vel topic of type \'{msg_type}\' is not supported. '
                                f'Must be one of: \'{self.supported_state_types}\'')
            self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        except ROSException as e:
            logging.logwarn(f'Couldn\'t connect to {self.cmd_vel_topic}. Is it running?')
            rospy.sleep(1)

        if joint_name is None:
            for joint in self.world.joints.values():
                if isinstance(joint, (OmniDrive, DiffDrive)):
                    # FIXME can only handle one drive
                    # self.controlled_joints = [joint]
                    self.joint = joint
            if not hasattr(self, 'joint'):
                #TODO
                pass
        else:
            joint_name = self.world.search_for_joint_name(joint_name)
            self.joint = self.world.joints[joint_name]
        self.world.register_controlled_joints([self.joint.name])
        loginfo(f'Received controlled joints from \'{cmd_vel_topic}\'.')

    def __str__(self):
        return f'{super().__str__()} ({self.cmd_vel_topic})'

    def solver_cmd_to_twist(self, cmd) -> Twist:
        twist = Twist()
        try:
            twist.linear.x = cmd.free_variable_data[self.joint.x_vel.name][0]
            if abs(twist.linear.x) < self.threshold[0]:
                twist.linear.x = 0
        except:
            twist.linear.x = 0
        try:
            twist.linear.y = cmd.free_variable_data[self.joint.y_vel.name][0]
            if abs(twist.linear.y) < self.threshold[1]:
                twist.linear.y = 0
        except:
            twist.linear.y = 0
        try:
            twist.angular.z = cmd.free_variable_data[self.joint.yaw.name][0]
            if abs(twist.angular.z) < self.threshold[2]:
                twist.angular.z = 0
        except:
            twist.angular.z = 0
        return twist

    @catch_and_raise_to_blackboard
    @profile
    def update(self):
        cmd = self.god_map.get_data(identifier.qp_solver_solution)
        twist = self.solver_cmd_to_twist(cmd)
        self.vel_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.vel_pub.publish(Twist())
        super().terminate(new_status)
