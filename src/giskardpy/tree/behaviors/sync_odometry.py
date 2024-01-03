from queue import Queue, Empty
from typing import Optional

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from py_trees import Status
from rospy import ROSException

from giskardpy.data_types import JointStates
from giskardpy.model.joints import OmniDrive
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.math import rpy_from_quaternion
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class SyncOdometry(GiskardBehavior):

    @profile
    def __init__(self, odometry_topic: str, joint_name: PrefixName, name_suffix: str = ''):
        self.odometry_topic = odometry_topic
        super().__init__(str(self) + name_suffix)
        self.joint_name = joint_name
        self.last_msg = None
        self.lock = Queue(maxsize=1)

    def __str__(self):
        return f'{super().__str__()} ({self.odometry_topic})'

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def setup(self, timeout=0.0):
        msg: Optional[Odometry] = None
        odom = True
        while msg is None and not rospy.is_shutdown():
            try:
                try:
                    msg = rospy.wait_for_message(self.odometry_topic, Odometry, rospy.Duration(1))
                except:
                    msg = rospy.wait_for_message(self.odometry_topic, PoseWithCovarianceStamped, rospy.Duration(1))
                    odom = False
                self.lock.put(msg)
            except ROSException as e:
                logging.logwarn(f'Waiting for topic \'{self.odometry_topic}\' to appear.')
        self.joint: OmniDrive = self.world.joints[self.joint_name]
        if odom:
            self.odometry_sub = rospy.Subscriber(self.odometry_topic, Odometry, self.cb, queue_size=1)
        else:
            self.odometry_sub = rospy.Subscriber(self.odometry_topic, PoseWithCovarianceStamped, self.cb, queue_size=1)

        return super().setup(timeout)

    def cb(self, data: Odometry):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        try:
            odometry: Odometry = self.lock.get()
            self.joint.update_transform(odometry.pose.pose)

        except Empty:
            pass
        return Status.SUCCESS


class SyncOdometryNoLock(GiskardBehavior):

    @profile
    def __init__(self, odometry_topic: str, joint_name: PrefixName, name_suffix: str = ''):
        self.odometry_topic = odometry_topic
        super().__init__(str(self) + name_suffix)
        self.joint_name = joint_name
        self.last_msg = None

    def __str__(self):
        return f'{super().__str__()} ({self.odometry_topic})'

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def setup(self, timeout=0.0):
        msg: Optional[Odometry] = None
        odom = True
        while msg is None and not rospy.is_shutdown():
            try:
                try:
                    msg = rospy.wait_for_message(self.odometry_topic, Odometry, rospy.Duration(1))
                except:
                    msg = rospy.wait_for_message(self.odometry_topic, PoseWithCovarianceStamped, rospy.Duration(1))
                    odom = False
                # self.lock.put(msg)
            except ROSException as e:
                logging.logwarn(f'Waiting for topic \'{self.odometry_topic}\' to appear.')
        self.joint: OmniDrive = self.world.joints[self.joint_name]
        if odom:
            self.odometry_sub = rospy.Subscriber(self.odometry_topic, Odometry, self.cb, queue_size=1)
        else:
            self.odometry_sub = rospy.Subscriber(self.odometry_topic, PoseWithCovarianceStamped, self.cb, queue_size=1)

        return super().setup(timeout)

    def cb(self, data: Odometry):
        self.odom = data

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        try:
            odometry: Odometry = self.odom
            self.joint.update_transform(odometry.pose.pose)

        except Empty:
            pass
        return Status.SUCCESS
