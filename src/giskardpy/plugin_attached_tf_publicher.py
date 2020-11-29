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

    def update(self):
        try:
            with self.get_god_map():
                robot_links = set(self.unsafe_get_robot().get_link_names())
                attached_links = robot_links - self.original_links
                if attached_links:
                    tf_msg = TFMessage()
                    get_fk = self.unsafe_get_robot().get_fk_pose
                    for link_name in attached_links:
                        parent_link_name = self.unsafe_get_robot().get_parent_link_of_link(link_name)
                        fk = get_fk(parent_link_name, link_name)
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
        except UnboundLocalError as e:
            pass
        except ValueError as e:
            pass
        return Status.SUCCESS
