from __future__ import annotations

from enum import Enum
from typing import Type, TypeVar, Union, Dict, List, Optional, Any

import inspect
import abc
from abc import ABC
from collections import defaultdict
from copy import copy
from time import time

import numpy as np
import py_trees
import pydot
import rospy
from py_trees import Chooser, common, Composite
from py_trees import Selector, Sequence
from py_trees_ros.trees import BehaviourTree
from sortedcontainers import SortedList

import giskardpy
from giskard_msgs.msg import MoveAction, MoveFeedback
from giskardpy.configs.collision_avoidance_config import CollisionCheckerLib
from giskardpy.exceptions import DuplicateNameException, BehaviorTreeException
from giskardpy.god_map import god_map
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUp, CleanUpPlanning, CleanUpBaseController
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.evaluate_debug_expressions import EvaluateDebugExpressions
from giskardpy.tree.behaviors.exception_to_execute import ClearBlackboardException
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.goal_done import GoalDone
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.instantaneous_controller_base import ControllerPluginBase
from giskardpy.tree.behaviors.joint_group_pos_controller_publisher import JointGroupPosController
from giskardpy.tree.behaviors.joint_group_vel_controller_publisher import JointGroupVelController
from giskardpy.tree.behaviors.joint_pos_controller_publisher import JointPosController
from giskardpy.tree.behaviors.joint_vel_controller_publisher import JointVelController
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.loop_detector import LoopDetector
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.behaviors.new_trajectory import NewTrajectory
from giskardpy.tree.behaviors.notify_state_change import NotifyStateChange
from giskardpy.tree.behaviors.plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.plugin_if import IF
from giskardpy.tree.behaviors.publish_debug_expressions import PublishDebugExpressions
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.real_kinematic_sim import RealKinSimPlugin
from giskardpy.tree.behaviors.ros_msg_to_goal import ParseActionGoal
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.behaviors.send_trajectory import SendFollowJointTrajectory
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import SendTrajectoryToCmdVel
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime2 import SendCmdVel
from giskardpy.tree.behaviors.set_move_result import SetMoveResult
from giskardpy.tree.behaviors.set_tracking_start_time import SetTrackingStartTime
from giskardpy.tree.behaviors.setup_base_traj_constraints import SetDriveGoals
from giskardpy.tree.behaviors.sleep import Sleep
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_configuration2 import SyncConfiguration2
from giskardpy.tree.behaviors.sync_odometry import SyncOdometry, SyncOdometryNoLock
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy.tree.behaviors.tf_publisher import TFPublisher, TfPublishingModes
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.time_real import RosTime
from giskardpy.tree.behaviors.visualization import VisualizationBehavior
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.branches.clean_up_control_loop import CleanupControlLoop
from giskardpy.tree.branches.control_loop import ControlLoop
from giskardpy.tree.branches.giskard_bt import GiskardBT
from giskardpy.tree.branches.post_processing import PostProcessing
from giskardpy.tree.branches.prepare_control_loop import PrepareControlLoop
from giskardpy.tree.branches.process_goal import ProcessGoal
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.branches.wait_for_goal import WaitForGoal
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.composites.better_parallel import ParallelPolicy, Parallel
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import failure_is_success, success_is_running, running_is_success, success_is_failure, \
    anything_is_success
from giskardpy.utils import logging
from giskardpy.utils.utils import create_path
from giskardpy.utils.utils import get_all_classes_in_package


def behavior_is_instance_of(obj: Any, type_: Type) -> bool:
    return isinstance(obj, type_) or hasattr(obj, 'original') and isinstance(obj.original, type_)


class ManagerNode:
    node: GiskardBehavior
    parent: ManagerNode
    position: int
    disabled_children: SortedList[ManagerNode]
    enabled_children: SortedList[ManagerNode]

    def __init__(self, node: GiskardBehavior, parent: ManagerNode, position: int):
        """
        :param node: the behavior that is represented by this ManagerNode
        :param parent: the parent of the behavior that is represented by this ManagerNode
        :param position: the position of the node in the list of children of the parent
        """
        self.node = node
        self.parent = parent
        self.position = position
        self.disabled_children = SortedList()
        self.enabled_children = SortedList()

    def __lt__(self, other):
        return self.position < other.position

    def __gt__(self, other):
        return self.position > other.position

    def __eq__(self, other):
        return self.node == other.node and self.parent == other.parent

    def __str__(self):
        return self.node.name

    def __repr__(self):
        return str(self)

    def disable_child(self, manager_node: ManagerNode):
        """
        marks the given manager node as disabled in the internal tree representation and removes it to the behavior tree
        """
        self.enabled_children.remove(manager_node)
        self.disabled_children.add(manager_node)
        self.node.remove_child(manager_node.node)

    def enable_child(self, manager_node: ManagerNode):
        """
        marks the given manager node as enabled in the internal tree representation and adds it to the behavior tree
        """
        self.disabled_children.discard(manager_node)
        if manager_node not in self.enabled_children:
            self.enabled_children.add(manager_node)
            idx = self.enabled_children.index(manager_node)
            self.node.insert_child(manager_node.node, idx)

    def add_child(self, manager_node: ManagerNode):
        """
        adds the given manager node to the internal tree map and the corresponding behavior to the behavior tree
        """
        if manager_node.position < 0:
            manager_node.position = 0
            if self.enabled_children:
                manager_node.position = max(manager_node.position, self.enabled_children[-1].position + 1)
            if self.disabled_children:
                manager_node.position = max(manager_node.position, self.disabled_children[-1].position + 1)
            idx = manager_node.position
        else:
            idx = self.disabled_children.bisect_left(manager_node)
            for c in self.disabled_children.islice(start=idx):
                c.position += 1
            idx = self.enabled_children.bisect_left(manager_node)
            for c in self.enabled_children.islice(start=idx):
                c.position += 1
        self.node.insert_child(manager_node.node, idx)
        self.enabled_children.add(manager_node)

    def remove_child(self, manager_node):
        """
        removes the given manager_node from the internal tree map and the corresponding behavior from the behavior tree
        :param manager_node:
        :type manager_node: TreeManager.ManagerNode
        :return:
        """
        if manager_node in self.enabled_children:
            self.enabled_children.remove(manager_node)
            self.node.remove_child(manager_node.node)
        elif manager_node in self.disabled_children:
            self.disabled_children.remove(manager_node)
        else:
            raise RuntimeError('could not remove node. this probably means that the tree is inconsistent')
        idx = self.disabled_children.bisect_right(manager_node)
        for c in self.disabled_children.islice(start=idx):
            c.position -= 1
        idx = self.enabled_children.bisect_right(manager_node)
        for c in self.enabled_children.islice(start=idx):
            c.position -= 1


def search_for(lines, function_name):
    data = []
    for i, x in enumerate(lines):
        if x.startswith('File') \
                and (function_name in lines[i + 6] or function_name in lines[i + 7]) \
                and 'behavior' in x:
            data.append((x.split('giskardpy/src/giskardpy/tree/behaviors/')[1][:-3], lines[i - 1].split(' ')[2]))
    result = defaultdict(dict)
    for file_name, time in data:
        result[file_name][function_name] = float(time)
    return result


class TreeManager(ABC):
    tree_nodes: Dict[str, ManagerNode]
    tick_rate: float = 0.05
    control_mode = ControlModes.none
    tree: GiskardBT

    @profile
    def __init__(self, control_mode: ControlModes):
        god_map.tree_manager = self
        self.action_server_name = god_map.giskard.action_server_name

        self.tree = GiskardBT(control_mode=control_mode)
        self.tree_nodes = {}

        # self.__init_map(self.tree.root, None, 0)

    def live(self):
        sleeper = rospy.Rate(1 / self.tick_rate)
        logging.loginfo('giskard is ready')
        while not rospy.is_shutdown():
            try:
                self.tick()
                sleeper.sleep()
            except KeyboardInterrupt:
                break
        logging.loginfo('giskard died')

    def tick(self):
        self.tree.tick()

    def setup(self, timeout=30):
        self.tree.setup(timeout)

    def kill_all_services(self):
        self.tree.blackboard_exchange.get_blackboard_variables_srv.shutdown()
        self.tree.blackboard_exchange.open_blackboard_watcher_srv.shutdown()
        self.tree.blackboard_exchange.close_blackboard_watcher_srv.shutdown()
        for value in god_map.tree_manager.tree_nodes.values():
            node = value.node
            for attribute_name, attribute in vars(node).items():
                if isinstance(attribute, rospy.Service):
                    attribute.shutdown(reason='life is pain')

    def __init_map(self, node, parent, idx):
        """
        initialises the internal map that represents the behavior tree. This method calls itself recursively for every
        node  in the tree
        :param node: the root node of the behavior tree
        :param parent: None if root
        :param idx: 0 if root
        :return:
        """
        manager_node = ManagerNode(node=node, parent=parent, position=idx)
        if parent is not None:
            parent.enabled_children.add(manager_node)
        # if isinstance(node, AsyncBehavior) or hasattr(node, 'original') and isinstance(node.original, AsyncBehavior):
        #     children = node._children
        #     for idx, child_name in enumerate(children):
        #         child_node = ManagerNode(node=children[child_name], parent=manager_node, position=idx)
        #         self.tree_nodes[child_name] = child_node
        #         manager_node.enabled_children.add(child_node)
        if node.name in self.tree_nodes:
            raise KeyError(f'node named {node.name} already exists')
        self.tree_nodes[node.name] = manager_node
        for idx, child in enumerate(node.children):
            self.__init_map(child, manager_node, idx)

    def disable_node(self, node_name):
        """
        disables the node with the given name
        :param node_name: the name of the node
        :return:
        """
        t = self.tree_nodes[node_name]
        if t.parent is not None:
            return t.parent.disable_child(t)
        else:
            logging.logwarn('cannot disable root node')
            return False

    def enable_node(self, node_name: str):
        """
        enables the node with the given name
        :param node_name: the name of the node
        """
        t = self.tree_nodes[node_name]
        if t.parent is not None:
            t.parent.enable_child(t)
        else:
            logging.loginfo('root node')

    def insert_node(self, node: GiskardBehavior, parent_name: str, position: int = -1):
        """
        inserts a node into the behavior tree.
        :param node: the node that will be inserted
        :param parent_name: the name of the parent node where the node will be inserted
        :param position: the node will be inserted as the nth child with n = len([x for x in children if x.position < position])
        """
        for i in range(100):
            if node.name in self.tree_nodes:
                node.name += '*'
            else:
                break
        else:
            raise ValueError(f'Node named {node.name} already exists.')
        parent = self.tree_nodes[parent_name]
        tree_node = ManagerNode(node=node, parent=parent, position=position)
        parent.add_child(tree_node)
        self.tree_nodes[node.name] = tree_node
        node.setup(1.0)

    def remove_node(self, node_name):
        """
        removes a node from the behavior tree
        :param node_name: the name of the node that will be removed
        :type node_name: str
        :return:
        """
        node = self.tree_nodes[node_name]
        parent = node.parent
        del self.tree_nodes[node_name]
        parent.remove_child(node)

    def get_node(self, node_name):
        """
        returns the behavior with the given name
        :param node_name:
        :type node_name: str
        :return: the behavior with the given name
        :rtype py_trees.behaviour.Behaviour:
        """
        return self.tree_nodes[node_name].node

    GiskardBehavior_ = TypeVar('GiskardBehavior_', bound=GiskardBehavior)

    def get_nodes_of_type(self, node_type: Type[GiskardBehavior_]) -> List[GiskardBehavior_]:
        return [node.node for node in self.tree_nodes.values() if behavior_is_instance_of(node.node, node_type)]

    def insert_node_behind_every_node_of_type(self, node_type: Type[GiskardBehavior],
                                              node_to_be_added: GiskardBehavior):
        nodes = self.get_nodes_of_type(node_type)
        for idx, node in enumerate(nodes):
            node_copy = copy(node_to_be_added)
            manager_node = self.tree_nodes[node.name]
            parent = manager_node.parent.node
            self.insert_node(node_copy, parent.name, self.tree_nodes[node.name].position + 1)

    def insert_node_behind_node_of_type(self, parent_node_name: str, node_type: Type[GiskardBehavior],
                                        node_to_be_added: GiskardBehavior):
        parent_node = self.tree_nodes[parent_node_name]
        for child_node in parent_node.enabled_children:
            if isinstance(child_node.node, node_type):
                self.insert_node(node_to_be_added, parent_node_name, child_node.position + 1)
                break

    def render(self):
        path = god_map.giskard.tmp_folder + 'tree'
        create_path(path)
        render_dot_tree(self.tree.root, name=path)


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
    # graph.write(filename_wo_extension + '.dot')
    graph.write_png(filename_wo_extension + '.png')
    # graph.write_svg(filename_wo_extension + '.svg')


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
        node_type = type(node)
        if hasattr(node, 'original'):
            node_type = type(node.original)

        if node_type == Chooser:
            attributes = ('doubleoctagon', 'cyan', 'black')  # octagon
        elif node_type == Selector:
            attributes = ('octagon', 'cyan', 'black')  # octagon
        elif node_type == Sequence:
            attributes = ('box', 'orange', 'black')
        elif node_type == Parallel:
            attributes = ('note', 'gold', 'black')
        elif node_type == AsyncBehavior:
            attributes = ('house', 'green', 'black')
        # elif isinstance(node, PluginBase) or node.children != []:
        #     attributes = ('ellipse', 'ghostwhite', 'black')  # encapsulating behaviour (e.g. wait)
        else:
            attributes = ('ellipse', 'gray', 'black')
        # if not isinstance(node, PluginBase) and node.blackbox_level != common.BlackBoxLevel.NOT_A_BLACKBOX:
        #     attributes = (attributes[0], 'gray20', blackbox_font_colours[node.blackbox_level])
        return attributes

    fontsize = 11
    fontname = 'Courier'
    graph = pydot.Dot(graph_type='digraph')
    graph.set_name(root.name.lower().replace(' ', '_'))
    # fonts: helvetica, times-bold, arial (times-roman is the default, but this helps some viewers, like kgraphviewer)
    graph.set_graph_defaults(fontname='times-roman')
    graph.set_node_defaults(fontname='times-roman')
    graph.set_edge_defaults(fontname='times-roman')
    (node_shape, node_colour, node_font_colour) = get_node_attributes(root, visibility_level)
    node_root = pydot.Node(root.name, shape=node_shape, style="filled", fillcolor=node_colour, fontsize=fontsize,
                           fontcolor=node_font_colour, fontname=fontname)
    graph.add_node(node_root)
    names = [root.name]

    def add_edges(root, root_dot_name, visibility_level):
        if visibility_level < root.blackbox_level:
            children = root.children
            names2 = [c.name for c in children]
            for name, c in zip(names2, children):
                (node_shape, node_colour, node_font_colour) = get_node_attributes(c, visibility_level)
                proposed_dot_name = name
                if hasattr(c, 'original'):
                    proposed_dot_name += f'\n{type(c).__name__}'
                color = 'black'
                if hasattr(c, 'original'):
                    original_c = c.original
                else:
                    original_c = c
                if isinstance(original_c, GiskardBehavior) and not isinstance(original_c, AsyncBehavior):
                    function_names = ['__init__', 'setup', 'initialise', 'update']
                    function_name_padding = 20
                    entry_name_padding = 8
                    number_padding = function_name_padding - entry_name_padding
                    if hasattr(original_c, '__times'):
                        time_dict = original_c.__times
                    else:
                        time_dict = {}
                    for function_name in function_names:
                        if function_name in time_dict:
                            times = time_dict[function_name]
                            average_time = np.average(times)
                            std_time = np.std(times)
                            total_time = np.sum(times)
                            if total_time > 1:
                                color = 'red'
                            proposed_dot_name += f'\n{function_name.ljust(function_name_padding, "-")}' \
                                                 f'\n{"  #calls".ljust(entry_name_padding)}{f"={len(times)}".ljust(number_padding)}' \
                                                 f'\n{"  avg".ljust(entry_name_padding)}{f"={average_time:.7f}".ljust(number_padding)}' \
                                                 f'\n{"  std".ljust(entry_name_padding)}{f"={std_time:.7f}".ljust(number_padding)}' \
                                                 f'\n{"  max".ljust(entry_name_padding)}{f"={max(times):.7f}".ljust(number_padding)}' \
                                                 f'\n{"  sum".ljust(entry_name_padding)}{f"={total_time:.7f}".ljust(number_padding)}'
                        else:
                            proposed_dot_name += f'\n{function_name.ljust(function_name_padding, "-")}'

                while proposed_dot_name in names:
                    proposed_dot_name = proposed_dot_name + "*"
                names.append(proposed_dot_name)
                node = pydot.Node(proposed_dot_name, shape=node_shape, style="filled", fillcolor=node_colour,
                                  fontsize=fontsize, fontcolor=node_font_colour, color=color, fontname=fontname)
                graph.add_node(node)
                edge = pydot.Edge(root_dot_name, proposed_dot_name)
                graph.add_edge(edge)
                if (hasattr(c, 'children') and c.children != []) or (hasattr(c, '_children') and c._children != []):
                    add_edges(c, proposed_dot_name, visibility_level)

    add_edges(root, root.name, visibility_level)
    return graph
