from queue import Queue, Empty
from typing import Optional

import rospy
from nav_msgs.msg import Odometry
from py_trees import Status
from pybullet import getAxisAngleFromQuaternion
from rospy import ROSException

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.exceptions import GiskardException
from giskardpy.model.joints import OmniDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class SyncOdometry(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    @profile
    def __init__(self, name, odometry_topic: str):
        super().__init__(name)
        self.map_frame = self.get_god_map().unsafe_get_data(identifier.map_frame)
        self.odometry_topic = odometry_topic
        self.last_msg = None
        self.lock = Queue(maxsize=1)

    @profile
    @catch_and_raise_to_blackboard
    def setup(self, timeout=0.0):
        msg: Optional[Odometry] = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(self.odometry_topic, Odometry, rospy.Duration(1))
                self.lock.put(msg)
            except ROSException as e:
                logging.logwarn(f'Waiting for topic \'{self.odometry_topic}\' to appear.')
        root_link = msg.header.frame_id
        child_link = msg.child_frame_id
        joints = self.world.compute_chain(root_link, child_link, True, False, True, True)
        if len(joints) != 1:
            raise GiskardException(f'Chain between {root_link} and {child_link} should be one joint, but its {joints}')
        self.brumbrum = joints[0]
        self.odometry_sub = rospy.Subscriber(self.odometry_topic, Odometry, self.cb, queue_size=1)
        return super().setup(timeout)

    def cb(self, data):
        try:
            self.lock.get_nowait()
        except Empty:
            pass
        self.lock.put(data)

    @profile
    @catch_and_raise_to_blackboard
    def update(self):
        try:
            if self.last_msg is None:
                odometry: Odometry = self.lock.get()
            else:
                odometry = self.lock.get_nowait()
            pose = odometry.pose.pose
            joint: OmniDrive = self.world.joints[self.brumbrum]
            self.last_msg = JointStates()
            self.last_msg[joint.x_name].position = pose.position.x
            self.last_msg[joint.y_name].position = pose.position.y
            self.last_msg[joint.rot_name].position = getAxisAngleFromQuaternion([pose.orientation.x,
                                                                                 pose.orientation.y,
                                                                                 pose.orientation.z,
                                                                                 pose.orientation.w])[1]
        except Empty:
            pass

        self.world.state.update(self.last_msg)
        self.world.notify_state_change()
        return Status.RUNNING
