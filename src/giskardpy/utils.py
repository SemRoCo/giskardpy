from __future__ import division

import pydot
import rospkg
from collections import defaultdict, OrderedDict
import numpy as np
from itertools import product, chain
from numpy import pi

import errno
from geometry_msgs.msg import PointStamped, Point, Vector3Stamped, Vector3, Pose, PoseStamped, QuaternionStamped, \
    Quaternion
from py_trees import common, Chooser, Selector, Sequence, Behaviour
from py_trees.composites import Parallel
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_multiply, quaternion_conjugate

from giskardpy.data_types import SingleJointState
from giskardpy.data_types import ClosestPointInfo
from contextlib import contextmanager
import sys, os
import pylab as plt

from giskardpy.plugin import PluginBehavior, NewPluginBase


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

class keydefaultdict(defaultdict):
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


# def slerp(q1, q2, t):
#     cos_half_theta = np.dot(q1, q2)
#
#     if (cos_half_theta < 0):
#         q2 = -q2
#         cos_half_theta = -cos_half_theta
#
#     if (abs(cos_half_theta) >= 1.0):
#         return q1
#
#     half_theta = math.acos(cos_half_theta)
#     sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)
#
#     if (abs(sin_half_theta) < 0.001):
#         return 0.5 * q1 + 0.5 * q2
#
#     ratio_a = np.sin((1.0 - t) * half_theta) / sin_half_theta
#     ratio_b = np.sin(t * half_theta) / sin_half_theta
#
#     return ratio_a * q1 + ratio_b * q2


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


def closest_point_constraint_violated(closest_point_infos, tolerance=0.9):
    """
    :param closest_point_infos: dict mapping a link name to a ClosestPointInfo
    :type closest_point_infos: dict
    :type tolerance: float
    :return: whether of not the contact distance for any link has been violated
    :rtype: bool
    """
    for link_name, cpi_info in closest_point_infos.items():  # type: (str, ClosestPointInfo)
        if cpi_info.contact_distance < cpi_info.min_dist * tolerance:
            print(cpi_info.link_a, cpi_info.link_b, cpi_info.contact_distance)
            return True
    return False


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


def dict_to_joint_states(joint_state_dict):
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

def create_path(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def plot_trajectory(tj, controlled_joints, path_to_data_folder):
    """
    :type tj: Trajectory
    :param controlled_joints: only joints in this list will be added to the plot
    :type controlled_joints: list
    """
    return
    if len(tj._points) <= 0:
        return
    colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
    line_styles = [u'', u'--', u'-.']
    fmts = [u''.join(x) for x in product(line_styles, colors)]
    positions = []
    velocities = []
    times = []
    names = [x for x in tj._points[0.0].keys() if x in controlled_joints]
    for time, point in tj.items():
        positions.append([v.position for j, v in point.items() if j in controlled_joints])
        velocities.append([v.velocity for j, v in point.items() if j in controlled_joints])
        times.append(time)
    positions = np.array(positions)
    velocities = np.array(velocities).T
    times = np.array(times)

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title(u'position')
    ax2.set_title(u'velocity')
    # positions -= positions.mean(axis=0)
    for i, position in enumerate(positions.T):
        ax1.plot(times, position, fmts[i], label=names[i])
        ax2.plot(times, velocities[i], fmts[i])
    box = ax1.get_position()
    # ax1.set_ylim(-3, 1)
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc=u'center', bbox_to_anchor=(1.45, 0))
    ax1.grid()
    ax2.grid()

    plt.savefig(path_to_data_folder + u'trajectory.pdf')


def resolve_ros_iris(input_urdf):
    """
    Replace all instances of ROS IRIs with a urdf string with global paths in the file system.
    :param input_urdf: URDF in which the ROS IRIs shall be replaced.
    :type input_urdf: str
    :return: URDF with replaced ROS IRIs.
    :rtype: str
    """
    rospack = rospkg.RosPack()
    output_urdf = u''
    for line in input_urdf.split(u'\n'):
        if u'package://' in line:
            prefix, suffix = line.split(u'package://', 1)
            package_name, suffix = suffix.split(u'/', 1)
            real_path = rospack.get_path(package_name)
            output_urdf += '{}{}/{}'.format(prefix, real_path, suffix)
        else:
            output_urdf += line
        output_urdf += u'\n'
    return output_urdf


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
    print("Writing %s.dot/svg/png" % filename_wo_extension)
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
        elif isinstance(node, NewPluginBase) or node.children != []:
            attributes = ('ellipse', 'ghostwhite', 'black')  # encapsulating behaviour (e.g. wait)
        else:
            attributes = ('ellipse', 'gray', 'black')
        if not isinstance(node, NewPluginBase) and node.blackbox_level != common.BlackBoxLevel.NOT_A_BLACKBOX:
            attributes = (attributes[0], 'gray20', blackbox_font_colours[node.blackbox_level])
        return attributes

    fontsize = 11
    graph = pydot.Dot(graph_type='digraph')
    graph.set_name(root.name.lower().replace(" ", "_"))
    # fonts: helvetica, times-bold, arial (times-roman is the default, but this helps some viewers, like kgraphviewer)
    graph.set_graph_defaults(fontname='times-roman')
    graph.set_node_defaults(fontname='times-roman')
    graph.set_edge_defaults(fontname='times-roman')
    (node_shape, node_colour, node_font_colour) = get_node_attributes(root, visibility_level)
    node_root = pydot.Node(root.name, shape=node_shape, style="filled", fillcolor=node_colour, fontsize=fontsize, fontcolor=node_font_colour)
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
                node = pydot.Node(proposed_dot_name, shape=node_shape, style="filled", fillcolor=node_colour, fontsize=fontsize, fontcolor=node_font_colour)
                graph.add_node(node)
                edge = pydot.Edge(root_dot_name, proposed_dot_name)
                graph.add_edge(edge)
                if (isinstance(c, PluginBehavior) and c.get_plugins() != []) or \
                        (isinstance(c, Behaviour) and c.children != []):
                    add_edges(c, proposed_dot_name, visibility_level)

    add_edges(root, root.name, visibility_level)
    return graph