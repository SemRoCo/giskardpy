from __future__ import division

import errno
import json
import os
import pydot
import pylab as plt
import re
import rospkg
import subprocess
import sys
import math
from collections import defaultdict, OrderedDict, deque
from contextlib import contextmanager
from functools import wraps
from itertools import product

import numpy as np
import pkg_resources
import rospy
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from giskard_msgs.msg import WorldBody
from numpy import pi
from py_trees import common, Chooser, Selector, Sequence, Behaviour
from py_trees.composites import Parallel
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_multiply, quaternion_conjugate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

from giskardpy import logging
from giskardpy.data_types import SingleJointState
from giskardpy.plugin import PluginBehavior
from giskardpy.tfwrapper import kdl_to_pose, np_to_kdl

r = rospkg.RosPack()


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


class KeyDefaultDict(defaultdict):
    """
    A default dict where the key is passed as parameter to the factory function.
    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def urdfs_equal(urdf1, urdf2):
    """
    Compairs two urdfs.
    :type urdf1: str
    :type urdf2: str
    :rtype: bool
    """
    # return hashlib.md5(urdf1).hexdigest() == hashlib.md5(urdf2).hexdigest()
    return urdf1 == urdf2


def sphere_volume(radius):
    """
    :type radius: float
    :rtype: float
    """
    return (4 / 3.) * pi * radius ** 3


def sphere_surface(radius):
    """
    :type radius: float
    :rtype: float
    """
    return 4 * pi * radius ** 2


def cube_volume(length, width, height):
    """
    :type length: float
    :type width: float
    :type height: float
    :rtype: float
    """
    return length * width * height


def cube_surface(length, width, height):
    """
    :type length: float
    :type width: float
    :type height: float
    :rtype: float
    """
    return 2 * (length * width + length * height + width * height)


def cylinder_volume(r, h):
    """
    :type r: float
    :type h: float
    :rtype: float
    """
    return pi * r ** 2 * h


def cylinder_surface(r, h):
    """
    :type r: float
    :type h: float
    :rtype: float
    """
    return 2 * pi * r * (h + r)


# def closest_point_constraint_violated(closest_point_infos, tolerance=0.9):
#     """
#     :param closest_point_infos: dict mapping a link name to a ClosestPointInfo
#     :type closest_point_infos: dict
#     :type tolerance: float
#     :return: whether of not the contact distance for any link has been violated
#     :rtype: bool
#     """
#     for link_name, cpi_info in closest_point_infos.items():  # type: (str, Collision)
#         if cpi_info.contact_distance < cpi_info.min_dist * tolerance:
#             logging.loginfo(u'collision constraints violated: {}'.format(cpi_info.link_a, cpi_info.link_b,
#                                                                          cpi_info.contact_distance))
#             return True
#     return False

def calculate_waypoint2D(target, origin, distance):
    """
    Calculates a waypoint in front of the target.
    :param target: The target position
    :param origin: The origin position
    :param distance: The distance of the new waypoint to the target
    :return:
    """
    distance = abs(distance)
    x = target.x - origin.x
    y = target.y - origin.y

    if x == 0.0:
        if y < 0.0:
            return Point(target.x, target.y + distance, target.z)
        else:
            return Point(target.x, target.y - distance, target.z)

    alpha = math.atan(y / x)
    dx = math.cos(alpha) * distance
    dy = math.sin(alpha) * distance

    if x > 0.0:
        return Point(target.x - dx, target.y - dy, target.z)
    else:
        return Point(target.x + dx, target.y + dy, target.z)


def qv_mult(quaternion, vector):
    """
    Transforms a vector by a quaternion
    :param quaternion: Quaternion
    :type quaternion: list
    :param vector: vector
    :type vector: list
    :return: transformed vector
    :type: list
    """
    q = quaternion
    v = [vector[0], vector[1], vector[2], 0]
    return quaternion_multiply(quaternion_multiply(q, v), quaternion_conjugate(q))[:-1]


#
# CONVERSION FUNCTIONS FOR ROS MESSAGES
#

def to_tf_quaternion(msg):
    """
    converts a geometry_msgs/Quaternion to tf_quaternion[x, y, z, w]
    :param msg: geometry_msgs/Quaternion
    :return: [x, y, z, w]
    """
    return [msg.x, msg.y, msg.z, msg.w]

def to_joint_state_dict(msg):
    """
    Converts a ROS message of type sensor_msgs/JointState into an instance of MultiJointState.
    :param msg: ROS message to convert.
    :type msg: JointState
    :return: Corresponding MultiJointState instance.
    :rtype: OrderedDict[str, SingleJointState]
    """
    mjs = OrderedDict()
    for i, joint_name in enumerate(msg.name):
        sjs = SingleJointState()
        sjs.name = joint_name
        sjs.position = msg.position[i]
        try:
            sjs.velocity = msg.velocity[i]
        except IndexError:
            sjs.velocity = 0
        try:
            sjs.effort = msg.effort[i]
        except IndexError:
            sjs.effort = 0
        mjs[joint_name] = sjs
    return mjs


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
        print("\'{}\': {},".format(key, value))
    print('}')

def write_dict(d, f):
    json.dump(d,f, sort_keys=True, indent=4, separators=(',', ': '))
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


def normalize_quaternion_msg(quaternion):
    q = Quaternion()
    rotation = np.array([quaternion.x,
                         quaternion.y,
                         quaternion.z,
                         quaternion.w])
    normalized_rotation = rotation / np.linalg.norm(rotation)
    q.x = normalized_rotation[0]
    q.y = normalized_rotation[1]
    q.z = normalized_rotation[2]
    q.w = normalized_rotation[3]
    return q


def to_point_stamped(frame_id, point):
    """
    Creates a PointStamped from a frame id and a list of floats.
    :type frame_id: str
    :param point: list containing 3 floats
    :type point: list
    :rtype: geometry_msgs.msg._PointStamped.PointStamped
    """
    p = PointStamped()
    p.header.frame_id = frame_id
    p.point = Point(*point)
    return p


def to_vector3_stamped(frame_id, vector):
    """
    Creates a Vector3 msg from a frame id and list of floats.
    :type frame_id: str
    :type vector: list
    :rtype: Vector3Stamped
    """
    v = Vector3Stamped()
    v.header.frame_id = frame_id
    v.vector = Vector3(*vector)
    return v


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


def position_dist(position1, position2):
    return np.linalg.norm(np.array(msg_to_list(position2)) - np.array(msg_to_list(position1)))


def create_path(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def plot_trajectory(tj, controlled_joints, path_to_data_folder, sample_period, order=3, velocity_threshold=0.0, scaling=0.2, normalize_position=False, tick_stride=1.0):
    """
    :type tj: Trajectory
    :param controlled_joints: only joints in this list will be added to the plot
    :type controlled_joints: list
    :param velocity_threshold: only joints that exceed this velocity threshold will be added to the plot. Use a negative number if you want to include every joint
    :param scaling: determines how much the x axis is scaled with the length(time) of the trajectory
    :param normalize_position: centers the joint positions around 0 on the y axis
    :param tick_stride: the distance between ticks in the plot. if tick_stride <= 0 pyplot determines the ticks automatically
    """

    def ceil(val, base=0.0, stride=1.0):
        base = base % stride
        return np.ceil((float)(val - base) / stride) * stride + base

    def floor(val, base=0.0, stride=1.0):
        base = base % stride
        return np.floor((float)(val - base) / stride) * stride + base

    order = max(order, 2)
    if len(tj._points) <= 0:
        return
    colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
    line_styles = [u'', u'--', u'-.', u':']
    fmts = [u''.join(i) for i in product(line_styles, colors)]
    data = [[] for i in range(order)]
    times = []
    names = list(sorted([i for i in tj._points[0.0].keys() if i in controlled_joints]))
    for time, point in tj.items():
        for i in range(order):
            if i == 0:
                data[0].append([point[joint_name].position for joint_name in names])
            elif i == 1:
                data[1].append([point[joint_name].velocity for joint_name in names])
        times.append(time)
    data[0] = np.array(data[0])
    data[1] = np.array(data[1])
    if(normalize_position):
        data[0] = data[0] - (data[0].max(0) + data[0].min(0)) / 2
    for i in range(2, order):
        data[i] = np.diff(data[i - 1], axis=0, prepend=0)
    times = np.array(times) * sample_period

    f, axs = plt.subplots(order, sharex=True, gridspec_kw={'hspace': 0.2})
    f.set_size_inches(w=(times[-1] - times[0]) * scaling, h=order * 3.5)

    plt.xlim(times[0], times[-1])

    if tick_stride > 0:
        first = ceil(times[0], stride=tick_stride)
        last = floor(times[-1], stride=tick_stride)
        ticks = np.arange(first, last, tick_stride)
        ticks = np.insert(ticks, 0, times[0])
        ticks = np.append(ticks, last)
        ticks = np.append(ticks, times[-1])
        for i in range(order):
            axs[i].set_title(r'$p' + '\'' * i + "$")
            axs[i].xaxis.set_ticks(ticks)
    else:
        for i in range(order):
            axs[i].set_title(r'$p' + '\'' * i + "$")
    for i in range(len(controlled_joints)):
        if any(abs(data[1][:, i]) > velocity_threshold):
            for j in range(order):
                axs[j].plot(times, data[j][:, i], fmts[i], label=names[i])


    axs[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    axs[-1].set_xlabel(u'time [s]')
    for i in range(order):
        axs[i].grid()

    plt.savefig(path_to_data_folder + u'trajectory.pdf', bbox_inches="tight")


def resolve_ros_iris_in_urdf(input_urdf):
    """
    Replace all instances of ROS IRIs with a urdfs string with global paths in the file system.
    :param input_urdf: URDF in which the ROS IRIs shall be replaced.
    :type input_urdf: str
    :return: URDF with replaced ROS IRIs.
    :rtype: str
    """
    output_urdf = u''
    for line in input_urdf.split(u'\n'):
        output_urdf += resolve_ros_iris(line)
        output_urdf += u'\n'
    return output_urdf


rospack = rospkg.RosPack()


def resolve_ros_iris(path):
    if u'package://' in path:
        split = path.split(u'package://')
        prefix = split[0]
        result = prefix
        for suffix in split[1:]:
            package_name, suffix = suffix.split(u'/', 1)
            real_path = rospack.get_path(package_name)
            result += u'{}/{}'.format(real_path, suffix)
        return result
    else:
        return path


def convert_dae_to_obj(path):
    path = path.replace(u'\'', u'')
    file_name = path.split(u'/')[-1]
    name, file_format = file_name.split(u'.')
    if u'dae' in file_format:
        input_path = resolve_ros_iris(path)
        new_path = u'/tmp/giskardpy/{}.obj'.format(name)
        create_path(new_path)
        try:
            subprocess.check_call([u'meshlabserver', u'-i', input_path, u'-o', new_path])
        except Exception as e:
            logging.logerr(u'meshlab not installed, can\'t convert dae to obj')
        return new_path
    return path


def write_to_tmp(filename, urdf_string):
    """
    Writes a URDF string into a temporary file on disc. Used to deliver URDFs to PyBullet that only loads file.
    :param filename: Name of the temporary file without any path information, e.g. 'pr2.urdfs'
    :type filename: str
    :param urdf_string: URDF as an XML string that shall be written to disc.
    :type urdf_string: str
    :return: Complete path to where the urdfs was written, e.g. '/tmp/pr2.urdfs'
    :rtype: str
    """
    new_path = u'/tmp/giskardpy/{}'.format(filename)
    create_path(new_path)
    with open(new_path, u'w') as o:
        o.write(urdf_string)
    return new_path


def render_dot_tree(root, visibility_level=common.VisibilityLevel.DETAIL, name=None):
    """
    Render the dot tree to .dot, .svg, .png. files in the current
    working directory. These will be named with the root behaviour name.

    Args:
        root (:class:`~py_trees.behaviour.Behaviour`): the root of a tree, or subtree
        visibility_level (:class`~py_trees.common.VisibilityLevel`): collapse subtrees at or under this level
        name (:obj:`str`): name to use for the created files (defaults to the root behaviour name)

    Example:

        Render a simple tree to dot/svg/png file:

        .. graphviz:: dot/sequence.dot

        .. code-block:: python

            root = py_trees.composites.Sequence("Sequence")
            for job in ["Action 1", "Action 2", "Action 3"]:
                success_after_two = py_trees.behaviours.Count(name=job,
                                                              fail_until=0,
                                                              running_until=1,
                                                              success_until=10)
                root.add_child(success_after_two)
            py_trees.display.render_dot_tree(root)

    .. tip::

        A good practice is to provide a command line argument for optional rendering of a program so users
        can quickly visualise what tree the program will execute.
    """
    graph = generate_pydot_graph(root, visibility_level)
    filename_wo_extension = root.name.lower().replace(" ", "_") if name is None else name
    logging.loginfo("Writing %s.dot/svg/png" % filename_wo_extension)
    graph.write(filename_wo_extension + '.dot')
    graph.write_png(filename_wo_extension + '.png')
    graph.write_svg(filename_wo_extension + '.svg')


def generate_pydot_graph(root, visibility_level):
    """
    Generate the pydot graph - this is usually the first step in
    rendering the tree to file. See also :py:func:`render_dot_tree`.

    Args:
        root (:class:`~py_trees.behaviour.Behaviour`): the root of a tree, or subtree
        visibility_level (:class`~py_trees.common.VisibilityLevel`): collapse subtrees at or under this level

    Returns:
        pydot.Dot: graph
    """

    def get_node_attributes(node, visibility_level):
        blackbox_font_colours = {common.BlackBoxLevel.DETAIL: "dodgerblue",
                                 common.BlackBoxLevel.COMPONENT: "lawngreen",
                                 common.BlackBoxLevel.BIG_PICTURE: "white"
                                 }
        if isinstance(node, Chooser):
            attributes = ('doubleoctagon', 'cyan', 'black')  # octagon
        elif isinstance(node, Selector):
            attributes = ('octagon', 'cyan', 'black')  # octagon
        elif isinstance(node, Sequence):
            attributes = ('box', 'orange', 'black')
        elif isinstance(node, Parallel):
            attributes = ('note', 'gold', 'black')
        elif isinstance(node, PluginBehavior):
            attributes = ('box', 'green', 'black')
        # elif isinstance(node, PluginBase) or node.children != []:
        #     attributes = ('ellipse', 'ghostwhite', 'black')  # encapsulating behaviour (e.g. wait)
        else:
            attributes = ('ellipse', 'gray', 'black')
        # if not isinstance(node, PluginBase) and node.blackbox_level != common.BlackBoxLevel.NOT_A_BLACKBOX:
        #     attributes = (attributes[0], 'gray20', blackbox_font_colours[node.blackbox_level])
        return attributes

    fontsize = 11
    graph = pydot.Dot(graph_type='digraph')
    graph.set_name(root.name.lower().replace(" ", "_"))
    # fonts: helvetica, times-bold, arial (times-roman is the default, but this helps some viewers, like kgraphviewer)
    graph.set_graph_defaults(fontname='times-roman')
    graph.set_node_defaults(fontname='times-roman')
    graph.set_edge_defaults(fontname='times-roman')
    (node_shape, node_colour, node_font_colour) = get_node_attributes(root, visibility_level)
    node_root = pydot.Node(root.name, shape=node_shape, style="filled", fillcolor=node_colour, fontsize=fontsize,
                           fontcolor=node_font_colour)
    graph.add_node(node_root)
    names = [root.name]

    def add_edges(root, root_dot_name, visibility_level):
        if visibility_level < root.blackbox_level:
            if isinstance(root, PluginBehavior):
                childrens = []
                names2 = []
                for name, children in root.get_plugins().items():
                    childrens.append(children)
                    names2.append(name)
            else:
                childrens = root.children
                names2 = [c.name for c in childrens]
            for name, c in zip(names2, childrens):
                (node_shape, node_colour, node_font_colour) = get_node_attributes(c, visibility_level)
                proposed_dot_name = name
                while proposed_dot_name in names:
                    proposed_dot_name = proposed_dot_name + "*"
                names.append(proposed_dot_name)
                node = pydot.Node(proposed_dot_name, shape=node_shape, style="filled", fillcolor=node_colour,
                                  fontsize=fontsize, fontcolor=node_font_colour)
                graph.add_node(node)
                edge = pydot.Edge(root_dot_name, proposed_dot_name)
                graph.add_edge(edge)
                if (isinstance(c, PluginBehavior) and c.get_plugins() != []) or \
                        (isinstance(c, Behaviour) and c.children != []):
                    add_edges(c, proposed_dot_name, visibility_level)

    add_edges(root, root.name, visibility_level)
    return graph


def remove_outer_tag(xml):
    """
    :param xml:
    :type xml: str
    :return:
    :rtype: str
    """
    return xml.split('>', 1)[1].rsplit('<', 1)[0]


def make_world_body_box(name=u'box', x_length=1, y_length=1, z_length=1):
    box = WorldBody()
    box.type = WorldBody.PRIMITIVE_BODY
    box.name = str(name)
    box.shape.type = SolidPrimitive.BOX
    box.shape.dimensions.append(x_length)
    box.shape.dimensions.append(y_length)
    box.shape.dimensions.append(z_length)
    return box


def make_world_body_sphere(name=u'sphere', radius=1):
    sphere = WorldBody()
    sphere.type = WorldBody.PRIMITIVE_BODY
    sphere.name = str(name)
    sphere.shape.type = SolidPrimitive.SPHERE
    sphere.shape.dimensions.append(radius)
    return sphere


def make_world_body_cylinder(name=u'cylinder', height=1, radius=1):
    cylinder = WorldBody()
    cylinder.type = WorldBody.PRIMITIVE_BODY
    cylinder.name = str(name)
    cylinder.shape.type = SolidPrimitive.CYLINDER
    cylinder.shape.dimensions = [0, 0]
    cylinder.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
    cylinder.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
    return cylinder


def make_urdf_world_body(name, urdf):
    wb = WorldBody()
    wb.name = name
    wb.type = wb.URDF_BODY
    wb.urdf = urdf
    return wb


def is_iterable(qwe):
    try:
        iter(qwe)
    except TypeError:
        return False
    return True


def homo_matrix_to_pose(m):
    return kdl_to_pose(np_to_kdl(m))


def compare_version(version1, operator, version2):
    """
    compares two version numbers by means of the given operator
    :param version1: version number 1 e.g. 0.1.0
    :type version1: str
    :param operator: ==,<=,>=,<,>
    :type operator: str
    :param version2: version number 1 e.g. 3.2.0
    :type version2: str
    :return:
    """
    version1 = version1.split('.')
    version2 = version2.split('.')
    if operator == '==':
        if (len(version1) != len(version2)):
            return False
        for i in range(len(version1)):
            if version1[i] != version2[i]:
                return False
        return True
    elif operator == '<=':
        k = min(len(version1), len(version2))
        for i in range(k):
            if version1[i] > version2[i]:
                return True
            elif version1[i] < version2[i]:
                return False
        if len(version1) < len(version2):
            return False
        else:
            return True
    elif operator == '>=':
        k = min(len(version1), len(version2))
        for i in range(k):
            if version1[i] < version2[i]:
                return True
            elif version1[i] > version2[i]:
                return False
        if len(version1) > len(version2):
            return False
        else:
            return True
    elif operator == '<':
        k = min(len(version1), len(version2))
        for i in range(k):
            if version1[i] > version2[i]:
                return True
            elif version1[i] < version2[i]:
                return False
        if len(version1) < len(version2):
            return False
        else:
            return True
    elif operator == '>':
        k = min(len(version1), len(version2))
        for i in range(k):
            if version1[i] < version2[i]:
                return True
            elif version1[i] > version2[i]:
                return False
        if len(version1) > len(version2):
            return False
        else:
            return True
    else:
        return False


def get_ros_pkg_path(ros_pkg):
    return r.get_path(ros_pkg)


def rospkg_exists(name):
    """
    checks whether a ros package with the given name and version exists
    :param name: the name and version of the ros package in requirements format e.g. giskard_msgs<=0.1.0
    :type name: str
    :return: True if it exits else False
    """
    name = name.replace(' ', '')
    version_list = name.split(',')
    version_entry1 = re.split('(==|>=|<=|<|>)', version_list[0])
    package_name = version_entry1[0]
    try:
        m = r.get_manifest(package_name)
    except Exception as e:
        logging.logwarn('package {name} not found'.format(name=name))
        return False
    if len(version_entry1) == 1:
        return True
    if not compare_version(version_entry1[2], version_entry1[1], m.version):
        logging.logwarn('found ROS package {installed_name}=={installed_version} but {r} is required}'.format(
            installed_name=package_name, installed_version=str(m.version), r=name))
        return False
    for entry in version_list[1:]:
        operator_and_version = re.split('(==|>=|<=|<|>)', entry)
        if not compare_version(operator_and_version[2], operator_and_version[1], m.version):
            logging.logwarn('found ROS package {installed_name}=={installed_version} but {r} is required}'.format(
                installed_name=package_name, installed_version=str(m.version), r=name))
            return False

    return True


def check_dependencies():
    """
    Checks whether the dependencies specified in the dependency.txt in the root folder of giskardpy are installed. If a
    dependecy is not installed a message is printed.
    """

    with open(get_ros_pkg_path('giskardpy') + '/dependencies.txt') as f:
        dependencies = f.readlines()

    dependencies = [x.split('#')[0] for x in dependencies]
    dependencies = [x.strip() for x in dependencies]

    for d in dependencies:
        try:
            pkg_resources.require(d)
        except pkg_resources.DistributionNotFound as e:
            rospkg_exists(d)
        except pkg_resources.VersionConflict as e:
            logging.logwarn('found {version_f} but version {version_r} is required'.format(version_r=str(e.req),
                                                                                           version_f=str(e.dist)))


def str_to_unique_number(s):
    # FIXME not actually unique
    return sum(ord(x) for x in s)


def memoize(function):
    memo = function.memo = {}

    @wraps(function)
    def wrapper(*args, **kwargs):
        # key = cPickle.dumps((args, kwargs))
        # key = pickle.dumps((args, sorted(kwargs.items()), -1))
        key = (args, frozenset(kwargs.items()))
        try:
            return memo[key]
        except KeyError:
            rv = function(*args, **kwargs)
            memo[key] = rv
            return rv

    return wrapper

def traj_to_msg(sample_period, trajectory, controlled_joints, fill_velocity_values):
    """
    :type traj: giskardpy.data_types.Trajectory
    :return: JointTrajectory
    """
    trajectory_msg = JointTrajectory()
    trajectory_msg.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
    trajectory_msg.joint_names = controlled_joints
    for time, traj_point in trajectory.items():
        p = JointTrajectoryPoint()
        p.time_from_start = rospy.Duration(time*sample_period)
        for joint_name in controlled_joints:
            if joint_name in traj_point:
                p.positions.append(traj_point[joint_name].position)
                if fill_velocity_values:
                    p.velocities.append(traj_point[joint_name].velocity)
            else:
                raise NotImplementedError(u'generated traj does not contain all joints')
        trajectory_msg.points.append(p)
    return trajectory_msg

def make_filter_b_mask(H):
    return H.sum(axis=1) != 0

def make_filter_masks(H, num_joint_constraints, num_hard_constraints):
    b_mask = make_filter_b_mask(H)
    s_mask = b_mask[num_joint_constraints:]
    if num_hard_constraints == 0:
        bA_mask = s_mask
    else:
        bA_mask = np.concatenate((np.array([True] * num_hard_constraints), s_mask))

    return bA_mask, b_mask





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

def publish_marker_sphere(position, frame_id=u'map', radius=0.05, id_=0):
    m = Marker()
    m.action = m.ADD
    m.ns = u'debug'
    m.id = id_
    m.type = m.SPHERE
    m.header.frame_id = frame_id
    m.pose.position.x = position[0]
    m.pose.position.y = position[1]
    m.pose.position.z = position[2]
    m.color = ColorRGBA(1,0,0,1)
    m.scale.x = radius
    m.scale.y = radius
    m.scale.z = radius

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    start = rospy.get_rostime()
    while pub.get_num_connections() < 1 and (rospy.get_rostime() - start).to_sec() < 2:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass

    pub.publish(m)

def publish_marker_vector(start, end, diameter_shaft=0.01, diameter_head=0.02,  id_=0):
    """
    assumes points to be in frame map
    :type start: Point
    :type end: Point
    :type diameter_shaft: float
    :type diameter_head: float
    :type id_: int
    """
    m = Marker()
    m.action = m.ADD
    m.ns = u'debug'
    m.id = id_
    m.type = m.ARROW
    m.points.append(start)
    m.points.append(end)
    m.color = ColorRGBA(1,0,0,1)
    m.scale.x = diameter_shaft
    m.scale.y = diameter_head
    m.scale.z = 0
    m.header.frame_id = u'map'

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    start = rospy.get_rostime()
    while pub.get_num_connections() < 1 and (rospy.get_rostime() - start).to_sec() < 2:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass
    rospy.sleep(0.3)

    pub.publish(m)

class FIFOSet(set):
    def __init__(self, data, max_length=None):
        if len(data) > max_length:
            raise ValueError('len(data) > max_length')
        super(FIFOSet, self).__init__(data)
        self.max_length = max_length
        self._data_queue = deque(data)

    def add(self, item):
        if len(self._data_queue) == self.max_length:
            to_delete = self._data_queue.popleft()
            super(FIFOSet, self).remove(to_delete)
            self._data_queue.append(item)
        super(FIFOSet, self).add(item)

    def remove(self, item):
        self.remove(item)
        self._data_queue.remove(item)
