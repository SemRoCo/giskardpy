from __future__ import division

# I only do this, because otherwise test/test_integration_pr2.py::TestWorldManipulation::test_unsupported_options
# fails on github actions
import urdf_parser_py.urdf as up

import errno
import inspect
import json
import os
import pkgutil
import sys
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from time import time
from typing import Type, Optional, Dict

import numpy as np
import roslaunch
import rospkg
import rospy
import trimesh
from genpy import Message
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from py_trees import Status, Blackboard
from rospy_message_converter.message_converter import \
    convert_ros_message_to_dictionary as original_convert_ros_message_to_dictionary, \
    convert_dictionary_to_ros_message as original_convert_dictionary_to_ros_message
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy import identifier
from giskardpy.exceptions import DontPrintStackTrace
from giskardpy.god_map import GodMap
from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector


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
    :param package_name: e.g. giskardpy.goals
    :param parent_class: e.g. Goal
    :return:
    """
    classes = {}
    package = __import__(package_name, fromlist="dummy")
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        try:
            module = __import__(f'{package.__name__}.{modname}', fromlist="dummy")
        except:
            if not silent:
                logging.loginfo(f'Failed to load {modname}')
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
    for i, joint_name in enumerate(msg.name):
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


def position_dict_to_joint_states(joint_state_dict):
    """
    :param joint_state_dict: maps joint_name to position
    :type joint_state_dict: dict
    :return: velocity and effort are filled with 0
    :rtype: JointState
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
    path = GodMap().get_data(identifier.tmp_folder)
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


def convert_to_decomposed_obj_and_save_in_tmp(file_name: str, log_path='/tmp/giskardpy/vhacd.log'):
    first_group_name = list(GodMap().get_data(identifier.world).groups.keys())[0]
    resolved_old_path = resolve_ros_iris(file_name)
    short_file_name = file_name.split('/')[-1][:-3]
    decomposed_obj_file_name = f'{first_group_name}/{short_file_name}obj'
    new_path = to_tmp_path(decomposed_obj_file_name)
    if not os.path.exists(new_path):
        mesh = trimesh.load(resolved_old_path, force='mesh')
        obj_str = trimesh.exchange.obj.export_obj(mesh)
        write_to_tmp(decomposed_obj_file_name, obj_str)
        logging.loginfo(f'converting {file_name} to obj and saved in {new_path}')
        # if not trimesh.convex.is_convex(mesh):
        #     pybullet.vhacd(new_path, new_path, log_path)

    return new_path


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


def convert_ros_message_to_dictionary(message: Message) -> dict:
    type_str_parts = str(type(message)).split('.')
    part1 = type_str_parts[0].split('\'')[1]
    part2 = type_str_parts[-1].split('\'')[0]
    message_type = f'{part1}/{part2}'
    d = {'message_type': message_type,
         'message': original_convert_ros_message_to_dictionary(message)}
    return d


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
