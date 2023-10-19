from abc import ABC
from typing import List, Optional

import numpy as np
import rospy
import rostopic
from geometry_msgs.msg import Twist
from py_trees import Status
from rospy import ROSException
from rostopic import ROSTopicException

from giskardpy.god_map import god_map
from giskardpy.model.joints import OmniDrive, DiffDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import catch_and_raise_to_blackboard
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import wait_for_topic_to_appear


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

        wait_for_topic_to_appear(self.cmd_vel_topic, self.supported_state_types)

        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        if joint_name is None:
            for joint in god_map.world.joints.values():
                if isinstance(joint, (OmniDrive, DiffDrive)):
                    # FIXME can only handle one drive
                    # self.controlled_joints = [joint]
                    self.joint = joint
            if not hasattr(self, 'joint'):
                #TODO
                pass
        else:
            joint_name = god_map.world.search_for_joint_name(joint_name)
            self.joint = god_map.world.joints[joint_name]
        god_map.world.register_controlled_joints([self.joint.name])
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
        cmd = god_map.qp_solver_solution
        twist = self.solver_cmd_to_twist(cmd)
        self.vel_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.vel_pub.publish(Twist())
        super().terminate(new_status)
