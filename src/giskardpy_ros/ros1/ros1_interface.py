from typing import List, Type, Optional

import genpy
import rospy
import rostopic
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from rospy import ROSException
from rostopic import ROSTopicException

from giskardpy.middleware import get_middleware


def wait_for_topic_to_appear(topic_name: str,
                             supported_types: Optional[List[Type[genpy.Message]]] = None,
                             sleep_time: float = 1) \
        -> Type[genpy.Message]:
    waiting_message = f'Waiting for topic \'{topic_name}\' to appear...'
    msg_type = None
    while msg_type is None and not rospy.is_shutdown():
        get_middleware().loginfo(waiting_message)
        try:
            rostopic.get_info_text(topic_name)
            msg_type, _, _ = rostopic.get_topic_class(topic_name)
            if msg_type is None:
                raise ROSTopicException()
            if supported_types is not None and msg_type not in supported_types:
                raise TypeError(f'Topic of type \'{msg_type}\' is not supported. '
                                f'Must be one of: \'{supported_types}\'')
            else:
                get_middleware().loginfo(f'\'{topic_name}\' appeared.')
                return msg_type
        except (ROSException, ROSTopicException) as e:
            rospy.sleep(sleep_time)


def make_pose_from_parts(pose, frame_id, position, orientation):
    if pose is None:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*(position if position is not None else [0, 0, 0]))
        pose.pose.orientation = Quaternion(*(orientation if orientation is not None else [0, 0, 0, 1]))
    return pose


def wait_for_publisher(publisher: rospy.Publisher):
    while publisher.get_num_connections() == 0:
        rospy.sleep(0.1)
