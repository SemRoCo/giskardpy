import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
from py_trees import Status
from tf2_msgs.msg import TFMessage

from giskardpy import logging
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import normalize_quaternion_msg


class TFPlugin(GiskardBehavior):
    """
    TODO
    """

    def __init__(self, name):
        super(TFPlugin, self).__init__(name)
        self.original_links = set(self.get_robot().get_link_names())
        self.tf_pub = rospy.Publisher(u'/tf', TFMessage, queue_size=10)

    def initialise(self):
        self.attached_links = set(self.get_robot().get_link_names()) - self.original_links

    def update(self):
        try:
            if self.attached_links:
                tf_msg = TFMessage()
                for link_name in self.attached_links:
                    fk = self.get_robot().get_fk_pose(self.get_robot().get_parent_link_of_link(link_name), link_name)
                    tf = TransformStamped()
                    tf.header = fk.header
                    tf.header.stamp = rospy.get_rostime()
                    tf.child_frame_id = link_name
                    tf.transform.translation.x = fk.pose.position.x
                    tf.transform.translation.y = fk.pose.position.y
                    tf.transform.translation.z = fk.pose.position.z
                    tf.transform.rotation = normalize_quaternion_msg(fk.pose.orientation)
                    tf_msg.transforms.append(tf)
                self.tf_pub.publish(tf_msg)
        except KeyError as e:
            pass
        return Status.SUCCESS
