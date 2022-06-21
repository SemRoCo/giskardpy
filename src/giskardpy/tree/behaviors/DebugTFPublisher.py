import rospy
from geometry_msgs.msg import TransformStamped, Quaternion
from py_trees import Status
from tf2_msgs.msg import TFMessage

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import normalize_quaternion_msg


class DebugTFPublisher(GiskardBehavior):
    """
    Published tf for attached and evironment objects.
    """

    @profile
    def __init__(self, name, tf_topic):
        super(DebugTFPublisher, self).__init__(name)
        self.tf_pub = rospy.Publisher(tf_topic, TFMessage, queue_size=10)

    def make_transform(self, parent_frame, child_frame, pose_list):
        q = Quaternion(pose_list[3], pose_list[4], pose_list[5], pose_list[6])
        tf = TransformStamped()
        tf.header.frame_id = parent_frame
        tf.header.stamp = rospy.get_rostime()
        tf.child_frame_id = child_frame
        tf.transform.translation.x = pose_list[0]
        tf.transform.translation.y = pose_list[1]
        tf.transform.translation.z = pose_list[2]
        tf.transform.rotation = normalize_quaternion_msg(q)
        return tf

    @profile
    def update(self):
        try:
            with self.get_god_map() as god_map:
                tfs = god_map.get_data(identifier.debug_tfs)
                if len(tfs) > 0:
                    tf_msg = TFMessage()
                    for t in tfs:
                        tf_msg.transforms.append(self.make_transform(t[0], t[1], t[1:]))
                    self.tf_pub.publish(tf_msg)

        except KeyError as e:
            pass
        except UnboundLocalError as e:
            pass
        except ValueError as e:
            pass
        return Status.SUCCESS
