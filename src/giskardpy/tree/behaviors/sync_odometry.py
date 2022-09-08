from queue import Queue, Empty
from typing import Optional

import rospy
import rostopic
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from py_trees import Status
from pybullet import getAxisAngleFromQuaternion
from rospy import ROSException, AnyMsg

import giskardpy.utils.tfwrapper as tf
from giskardpy.data_types import JointStates
from giskardpy.exceptions import GiskardException
from giskardpy.model.joints import OmniDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class SyncOdometry(GiskardBehavior):

    @profile
    def __init__(self, odometry_topic: str):
        self.odometry_topic = odometry_topic
        super().__init__(str(self))
        self.last_msg = None
        self.lock = Queue(maxsize=1)

    def __str__(self):
        return f'{super().__str__()} ({self.odometry_topic})'

    @catch_and_raise_to_blackboard
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
        root_link = msg.header.frame_id
        try:
            child_link = msg.child_frame_id
        except:
            child_link = 'base_footprint'
        joints = self.world.compute_chain(root_link, child_link, True, False, True, True)
        if len(joints) != 1:
            raise GiskardException(f'Chain between {root_link} and {child_link} should be one joint, but its {joints}')
        self.brumbrum = joints[0]
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
    @profile
    def update(self):
        joint: OmniDrive = self.world.joints[self.brumbrum]
        try:
            odometry: Odometry = self.lock.get()
            pose = odometry.pose.pose
            self.last_msg = JointStates()
            self.world.state[joint.x_name].position = pose.position.x
            self.world.state[joint.y_name].position = pose.position.y
            axis, angle = getAxisAngleFromQuaternion([pose.orientation.x,
                                                      pose.orientation.y,
                                                      pose.orientation.z,
                                                      pose.orientation.w])
            if axis[-1] < 0:
                angle = -angle
            self.world.state[joint.rot_name].position = angle

        except Empty:
            pass
        self.world.notify_state_change()
        return Status.RUNNING
