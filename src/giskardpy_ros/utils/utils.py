from __future__ import division
import hashlib

import genpy
import rostopic
# I only do this, because otherwise test/test_integration_pr2.py::TestWorldManipulation::test_unsupported_options
# fails on github actions
import urdf_parser_py.urdf as up

import errno
import inspect
import json
import os
import pkgutil
import sys
from collections import OrderedDict
from contextlib import contextmanager
from functools import cached_property
from typing import Type, Optional, Dict, Any, List, Union, Tuple

import numpy as np
import roslaunch
import rospkg
import rospy
from genpy import Message
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from py_trees import Blackboard
from rospy import ROSException
from rospy_message_converter.message_converter import \
    convert_ros_message_to_dictionary as original_convert_ros_message_to_dictionary, \
    convert_dictionary_to_ros_message as original_convert_dictionary_to_ros_message
from rostopic import ROSTopicException
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.god_map import god_map
from giskardpy.data_types import PrefixName
from giskardpy.utils import logging


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stdout


@contextmanager
def suppress_stdout():
    devnull = os.open('/dev/null', os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(devnull)


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def get_all_classes_in_package(package_name: str, parent_class: Optional[Type] = None, silent: bool = False) \
        -> Dict[str, Type]:
    """
    :param package_name: e.g. giskardpy_ros.goals
    :param parent_class: e.g. Goal
    :return:
    """
    classes = {}
    package = __import__(package_name, fromlist="dummy")
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        try:
            module = __import__(f'{package.__name__}.{modname}', fromlist="dummy")
        except Exception as e:
            if not silent:
                logging.logwarn(f'Failed to load {modname}: {str(e)}')
            continue
        for name2, value2 in inspect.getmembers(module, inspect.isclass):
            if parent_class is None or issubclass(value2, parent_class) and package_name in str(value2):
                classes[name2] = value2
    return classes


def limits_from_urdf_joint(urdf_joint):
    lower_limits = {}
    upper_limits = {}
    if not urdf_joint.type == 'continuous':
        try:
            lower_limits[0] = max(urdf_joint.safety_controller.soft_lower_limit, urdf_joint.limit.lower)
            upper_limits[0] = min(urdf_joint.safety_controller.soft_upper_limit, urdf_joint.limit.upper)
        except AttributeError:
            try:
                lower_limits[0] = urdf_joint.limit.lower
                upper_limits[0] = urdf_joint.limit.upper
            except AttributeError:
                pass
    try:
        lower_limits[1] = -urdf_joint.limit.velocity
        upper_limits[1] = urdf_joint.limit.velocity
    except AttributeError:
        pass
    return lower_limits, upper_limits


#
# CONVERSION FUNCTIONS FOR ROS MESSAGES
#

def to_joint_state_position_dict(msg):
    """
    Converts a ROS message of type sensor_msgs/JointState into a dict that maps name to position
    :param msg: ROS message to convert.
    :type msg: JointState
    :return: Corresponding MultiJointState instance.
    :rtype: OrderedDict[str, float]
    """
    js = OrderedDict()
    for joint_name, i in sorted(zip(msg.name, range(len(msg.name))), key=lambda x: x[0]):
        js[joint_name] = msg.position[i]
    return js


def print_joint_state(joint_msg):
    print_dict(to_joint_state_position_dict(joint_msg))


def print_dict(d):
    print('{')
    for key, value in d.items():
        print(f'{key}: {value},')
    print('}')


def write_dict(d, f):
    json.dump(d, f, sort_keys=True, indent=4, separators=(',', ': '))
    f.write('\n')


def position_dict_to_joint_states(joint_state_dict: Dict[str, float]) -> JointState:
    """
    :param joint_state_dict: maps joint_name to position
    :return: velocity and effort are filled with 0
    """
    js = JointState()
    for k, v in joint_state_dict.items():
        js.name.append(k)
        js.position.append(v)
        js.velocity.append(0)
        js.effort.append(0)
    return js


def dict_to_joint_states(joint_state_dict):
    """
    :param joint_state_dict: maps joint_name to position
    :type joint_state_dict: dict
    :return: velocity and effort are filled with 0
    :rtype: JointState
    """
    js = JointState()
    for k, v in sorted(joint_state_dict.items()):
        js.name.append(k)
        js.position.append(v.position)
        js.velocity.append(v.velocity)
        js.effort.append(0)
    return js


def msg_to_list(thing):
    """
    :param thing: ros msg
    :rtype: list
    """
    if isinstance(thing, QuaternionStamped):
        thing = thing.quaternion
    if isinstance(thing, Quaternion):
        return [thing.x,
                thing.y,
                thing.z,
                thing.w]
    if isinstance(thing, PointStamped):
        thing = thing.point
    if isinstance(thing, PoseStamped):
        thing = thing.pose
    if isinstance(thing, Vector3Stamped):
        thing = thing.vector
    if isinstance(thing, Point) or isinstance(thing, Vector3):
        return [thing.x,
                thing.y,
                thing.z]
    if isinstance(thing, Pose):
        return [thing.position.x,
                thing.position.y,
                thing.position.z,
                thing.orientation.x,
                thing.orientation.y,
                thing.orientation.z,
                thing.orientation.w]


def create_path(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def cm_to_inch(cm):
    return cm * 0.393701


def get_file_hash(file_path: str, algorithm: str = "sha256") -> Optional[str]:
    try:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        print(f"Error occurred while hashing: {e}")
        return None


def clear_cached_properties(instance: Any):
    """
    Clears the cache of all cached_property attributes of an instance.

    Args:
        instance: The instance for which to clear all cached_property caches.
    """
    for attr in dir(instance):
        if isinstance(getattr(type(instance), attr, None), cached_property):
            if attr in instance.__dict__:
                del instance.__dict__[attr]


def resolve_ros_iris_in_urdf(input_urdf):
    """
    Replace all instances of ROS IRIs with a urdfs string with global paths in the file system.
    :param input_urdf: URDF in which the ROS IRIs shall be replaced.
    :type input_urdf: str
    :return: URDF with replaced ROS IRIs.
    :rtype: str
    """
    output_urdf = ''
    for line in input_urdf.split('\n'):
        output_urdf += resolve_ros_iris(line)
        output_urdf += '\n'
    return output_urdf


rospack = rospkg.RosPack()


def resolve_ros_iris(path: str) -> str:
    """
    e.g. 'package://giskardpy_ros/data'
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


def write_to_tmp(file_name: str, file_str: str) -> str:
    """
    Writes a URDF string into a temporary file on disc. Used to deliver URDFs to PyBullet that only loads file.
    :param file_name: Name of the temporary file without any path information, e.g. 'pr2.urdfs'
    :param file_str: URDF as an XML string that shall be written to disc.
    :return: Complete path to where the urdfs was written, e.g. '/tmp/pr2.urdfs'
    """
    new_path = to_tmp_path(file_name)
    create_path(new_path)
    with open(new_path, 'w') as f:
        f.write(file_str)
    return new_path


def to_tmp_path(file_name: str) -> str:
    path = god_map.giskard.tmp_folder
    return resolve_ros_iris(f'{path}{file_name}')


def load_from_tmp(file_name: str):
    new_path = to_tmp_path(file_name)
    create_path(new_path)
    with open(new_path, 'r') as f:
        loaded_file = f.read()
    return loaded_file


def fix_obj(file_name):
    logging.loginfo(f'Attempting to fix {file_name}.')
    with open(file_name, 'r') as f:
        lines = f.readlines()
        fixed_obj = ''
        for line in lines:
            if line.startswith('f '):
                new_line = 'f '
                for part in line.split(' ')[1:]:
                    f_number = part.split('//')[0]
                    new_part = '//'.join([f_number, f_number])
                    new_line += ' ' + new_part
                fixed_obj += new_line + '\n'
            else:
                fixed_obj += line
        with open(file_name, 'w') as f:
            f.write(fixed_obj)


def launch_launchfile(file_name: str):
    launch_file = resolve_ros_iris(file_name)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
    with suppress_stderr():
        launch.start()
        # launch.shutdown()


blackboard_exception_name = 'exception'


def raise_to_blackboard(exception):
    Blackboard().set(blackboard_exception_name, exception)


def has_blackboard_exception():
    return hasattr(Blackboard(), blackboard_exception_name) \
        and getattr(Blackboard(), blackboard_exception_name) is not None


def get_blackboard_exception():
    return Blackboard().get(blackboard_exception_name)


def clear_blackboard_exception():
    raise_to_blackboard(None)


def make_pose_from_parts(pose, frame_id, position, orientation):
    if pose is None:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*(position if position is not None else [0, 0, 0]))
        pose.pose.orientation = Quaternion(*(orientation if orientation is not None else [0, 0, 0, 1]))
    return pose


def convert_ros_message_to_dictionary(message) -> dict:
    if isinstance(message, list):
        for i, element in enumerate(message):
            message[i] = convert_ros_message_to_dictionary(element)
    elif isinstance(message, dict):
        for k, v in message.copy().items():
            message[k] = convert_ros_message_to_dictionary(v)

    elif isinstance(message, tuple):
        list_values = list(message)
        for i, element in enumerate(list_values):
            list_values[i] = convert_ros_message_to_dictionary(element)
        message = tuple(list_values)

    elif isinstance(message, Message):

        type_str_parts = str(type(message)).split('.')
        part1 = type_str_parts[0].split('\'')[1]
        part2 = type_str_parts[-1].split('\'')[0]
        message_type = f'{part1}/{part2}'
        d = {'message_type': message_type,
             'message': original_convert_ros_message_to_dictionary(message)}
        return d

    return message


def replace_prefix_name_with_str(d: dict) -> dict:
    new_d = d.copy()
    for k, v in d.items():
        if isinstance(k, PrefixName):
            del new_d[k]
            new_d[str(k)] = v
        if isinstance(v, PrefixName):
            new_d[k] = str(v)
        if isinstance(v, dict):
            new_d[k] = replace_prefix_name_with_str(v)
    return new_d


def convert_dictionary_to_ros_message(json):
    # maybe somehow search for message that fits to structure of json?
    return original_convert_dictionary_to_ros_message(json['message_type'], json['message'])


def trajectory_to_np(tj, joint_names):
    """
    :type tj: Trajectory
    :return:
    """
    names = list(sorted([i for i in tj._points[0.0].keys() if i in joint_names]))
    position = []
    velocity = []
    times = []
    for time, point in tj.items():
        position.append([point[joint_name].position for joint_name in names])
        velocity.append([point[joint_name].velocity for joint_name in names])
        times.append(time)
    position = np.array(position)
    velocity = np.array(velocity)
    times = np.array(times)
    return names, position, velocity, times


_pose_publisher = None


def publish_pose(pose: PoseStamped):
    global _pose_publisher
    if _pose_publisher is None:
        _pose_publisher = rospy.Publisher('~visualization_marker_array', MarkerArray)
        rospy.sleep(1)
    m = Marker()
    m.header = pose.header
    m.pose = pose.pose
    m.action = m.ADD
    m.type = m.ARROW
    m.id = 1337
    m.ns = 'giskard_debug_poses'
    m.scale.x = 0.1
    m.scale.y = 0.05
    m.scale.z = 0.025
    m.color.r = 1
    m.color.a = 1
    ms = MarkerArray()
    ms.markers.append(m)
    _pose_publisher.publish(ms)


def int_to_bit_list(number: int) -> List[int]:
    return [2 ** i * int(bit) for i, bit in enumerate(reversed("{0:b}".format(number))) if int(bit) != 0]


def split_pose_stamped(pose: PoseStamped) -> Tuple[PointStamped, QuaternionStamped]:
    point = PointStamped()
    point.header = pose.header
    point.point = pose.pose.position

    quaternion = QuaternionStamped()
    quaternion.header = pose.header
    quaternion.quaternion = pose.pose.orientation
    return point, quaternion


def json_str_to_kwargs(json_str: str) -> Dict[str, Any]:
    d = json.loads(json_str)
    return json_to_kwargs(d)


def json_to_kwargs(d: dict) -> Dict[str, Any]:
    if isinstance(d, list):
        for i, element in enumerate(d):
            d[i] = json_to_kwargs(element)

    if isinstance(d, dict):
        if 'message_type' in d:
            d = convert_dictionary_to_ros_message(d)
        else:
            for key, value in d.copy().items():
                d[key] = json_to_kwargs(value)

    return d


def kwargs_to_json(kwargs: Dict[str, Any]) -> str:
    for k, v in kwargs.copy().items():
        if v is None:
            del kwargs[k]
        else:
            kwargs[k] = thing_to_json(v)
    kwargs = replace_prefix_name_with_str(kwargs)
    return json.dumps(kwargs)


def thing_to_json(thing: Any) -> Any:
    if isinstance(thing, list):
        return [thing_to_json(x) for x in thing]
    if isinstance(thing, dict):
        return {k: thing_to_json(v) for k, v in thing.items()}
    if isinstance(thing, Message):
        return convert_ros_message_to_dictionary(thing)
    return thing


def string_shortener(original_str: str, max_lines: int, max_line_length: int) -> str:
    if len(original_str) < max_line_length:
        return original_str
    lines = []
    start = 0
    for _ in range(max_lines):
        end = start + max_line_length
        lines.append(original_str[start:end])
        if end >= len(original_str):
            break
        start = end

    result = '\n'.join(lines)

    # Check if string is cut off and add "..."
    if len(original_str) > start:
        result = result + '...'

    return result


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


def get_ros_msgs_constant_name_by_value(ros_msg_class: genpy.Message, value: Union[str, int, float]) -> str:
    for attr_name in dir(ros_msg_class):
        if not attr_name.startswith('_'):
            attr_value = getattr(ros_msg_class, attr_name)
            if attr_value == value:
                return attr_name
    raise AttributeError(f'Message type {ros_msg_class} has no constant that matches {value}.')


class ImmutableDict(dict):
    """
    A dict that prevent reassignment of keys.
    """

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f'Key "{key}" already exists. Cannot reassign value.')
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self.__setitem__(k, v)


def is_running_in_pytest():
    return "pytest" in sys.modules
