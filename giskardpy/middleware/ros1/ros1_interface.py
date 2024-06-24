from typing import List, Type

import genpy
import rospkg
import rospy
import rostopic
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from rospy import ROSException
from rostopic import ROSTopicException

from giskardpy.middleware import logging


def wait_for_topic_to_appear(topic_name: str, supported_types: List[Type[genpy.Message]]) -> Type[genpy.Message]:
    waiting_message = f'Waiting for topic \'{topic_name}\' to appear...'
    msg_type = None
    while msg_type is None and not rospy.is_shutdown():
        logging.loginfo(waiting_message)
        try:
            rostopic.get_info_text(topic_name)
            msg_type, _, _ = rostopic.get_topic_class(topic_name)
            if msg_type is None:
                raise ROSTopicException()
            if msg_type not in supported_types:
                raise TypeError(f'Topic of type \'{msg_type}\' is not supported. '
                                f'Must be one of: \'{supported_types}\'')
            else:
                logging.loginfo(f'\'{topic_name}\' appeared.')
                return msg_type
        except (ROSException, ROSTopicException) as e:
            rospy.sleep(1)


def make_pose_from_parts(pose, frame_id, position, orientation):
    if pose is None:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*(position if position is not None else [0, 0, 0]))
        pose.pose.orientation = Quaternion(*(orientation if orientation is not None else [0, 0, 0, 1]))
    return pose


rospack = rospkg.RosPack()


def resolve_ros_iris(path: str) -> str:
    """
    e.g. 'package://giskardpy/data'
    """
    if 'package://' in path:
        split = path.split('package://')
        prefix = split[0]
        result = prefix
        for suffix in split[1:]:
            package_name, suffix = suffix.split('/', 1)
            real_path = rospack.get_path(package_name)
            result += f'{real_path}/{suffix}'
        return result
    else:
        return path
