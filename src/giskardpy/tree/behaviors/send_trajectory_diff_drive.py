import abc
from abc import ABC
from threading import Thread
from typing import List

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
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import SendTrajectoryClosedLoop
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import catch_and_raise_to_blackboard



class DiffDriveCmdVel(SendTrajectoryClosedLoop):
    min_deadline: rospy.Time
    max_deadline: rospy.Time
    update_thread: Thread
    supported_state_types = [Twist]

    @profile
    def __init__(self, name, cmd_vel_topic, goal_time_tolerance=1):
        super().__init__(name)
        self.goal_time_tolerance = rospy.Duration(goal_time_tolerance)

        loginfo(f'Waiting for cmd_vel topic \'{cmd_vel_topic}\' to appear.')
        try:
            msg_type, _, _ = rostopic.get_topic_class(cmd_vel_topic)
            if msg_type is None:
                raise ROSTopicException()
            if msg_type not in self.supported_state_types:
                raise TypeError(f'Cmd_vel topic of type \'{msg_type}\' is not supported. '
                                f'Must be one of: \'{self.supported_state_types}\'')
            self.vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        except ROSException as e:
            logging.logwarn(f'Couldn\'t connect to {cmd_vel_topic}. Is it running?')
            rospy.sleep(1)

        for joint in self.world.joints.values():
            if isinstance(joint, DiffDrive):
                # FIXME can only handle one drive
                self.controlled_joints = [joint]
                self.joint = joint
        self.world.register_controlled_joints(self.controlled_joints)
        loginfo(f'Received controlled joints from \'{cmd_vel_topic}\'.')

    def get_drive_goals(self) -> List[Goal]:
        return [SetPredictionHorizon(god_map=self.god_map, prediction_horizon=13),
                BaseTrajFollower(god_map=self.god_map, joint_name=self.joint.name)]

    @profile
    @catch_and_raise_to_blackboard
    def initialise(self):
        super().initialise()
        self.trajectory = self.get_god_map().get_data(identifier.trajectory)
        sample_period = self.god_map.unsafe_get_data(identifier.sample_period)
        self.start_time = self.god_map.unsafe_get_data(identifier.tracking_start_time)
        self.trajectory = self.trajectory.to_msg(sample_period, self.start_time, [self.joint], True)
        self.end_time = self.start_time + self.trajectory.points[-1].time_from_start + self.goal_time_tolerance
        # self.update_thread = Thread(target=self.worker)
        # self.update_thread.start()

    @catch_and_raise_to_blackboard
    def update(self):
        t = rospy.get_rostime()
        if self.start_time > t:
            self.vel_pub.publish(Twist())
            return Status.RUNNING
        if t <= self.end_time:
            cmd = self.god_map.get_data(identifier.qp_solver_solution)
            twist = Twist()
            try:
                twist.linear.x = cmd[0][self.joint.x_vel.position_name]
            except:
                twist.linear.x = 0
            # try:
            #     twist.linear.y = cmd[0][self.joint.y_vel.position_name]
            # except:
            # twist.linear.y = 0
            try:
                twist.angular.z = cmd[0][self.joint.rot_vel.position_name]
            except:
                twist.angular.z = 0
            # print(f'twist: {twist.linear.x:.4} {twist.linear.y:.4} {twist.angular.z:.4}')
            # print_dict(self.god_map.get_data(identifier.debug_expressions_evaluated))
            # print('-----------------')
            self.vel_pub.publish(twist)
            return Status.RUNNING
        self.vel_pub.publish(Twist())
        return Status.SUCCESS
        # if self.update_thread.is_alive():
        #     return Status.RUNNING
        # return Status.SUCCESS

