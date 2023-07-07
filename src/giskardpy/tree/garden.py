from __future__ import annotations
import inspect
import abc
from abc import ABC
from collections import defaultdict
from copy import copy
from time import time
from typing import Type, TypeVar, Union, Dict, List, Optional, Any

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
from giskardpy import identifier
from giskardpy.configs.data_types import CollisionCheckerLib, TfPublishingModes
from giskardpy.exceptions import DuplicateNameException, BehaviorTreeException
from giskardpy.god_map import GodMap
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUp, CleanUpPlanning, CleanUpBaseController
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.commands_remaining import CommandsRemaining
from giskardpy.tree.behaviors.evaluate_debug_expressions import EvaluateDebugExpressions
from giskardpy.tree.behaviors.exception_to_execute import ExceptionToExecute
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.goal_cleanup import GoalCleanUp
from giskardpy.tree.behaviors.goal_done import GoalDone
from giskardpy.tree.behaviors.goal_reached import GoalReached
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.instantaneous_controller_base import ControllerPluginBase
from giskardpy.tree.behaviors.joint_group_pos_controller_publisher import JointGroupPosController
from giskardpy.tree.behaviors.joint_group_vel_controller_publisher import JointGroupVelController
from giskardpy.tree.behaviors.joint_pos_controller_publisher import JointPosController
from giskardpy.tree.behaviors.joint_vel_controller_publisher import JointVelController
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.log_debug_expressions import LogDebugExpressionsPlugin
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
from giskardpy.tree.behaviors.ros_msg_to_goal import RosMsgToGoal
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.behaviors.send_trajectory import SendFollowJointTrajectory
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime import SendTrajectoryToCmdVel
from giskardpy.tree.behaviors.send_trajectory_omni_drive_realtime2 import SendCmdVel
from giskardpy.tree.behaviors.set_cmd import SetCmd
from giskardpy.tree.behaviors.set_error_code import SetErrorCode
from giskardpy.tree.behaviors.set_tracking_start_time import SetTrackingStartTime
from giskardpy.tree.behaviors.setup_base_traj_constraints import SetDriveGoals
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_configuration2 import SyncConfiguration2
from giskardpy.tree.behaviors.sync_odometry import SyncOdometry, SyncOdometryNoLock
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy.tree.behaviors.tf_publisher import TFPublisher
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.time_real import RosTime
from giskardpy.tree.behaviors.visualization import VisualizationBehavior
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.composites.better_parallel import ParallelPolicy, Parallel
from giskardpy.utils import logging
from giskardpy.utils.utils import create_path
from giskardpy.utils.utils import get_all_classes_in_package

T = TypeVar('T', bound=Union[Type[GiskardBehavior], Type[Composite]])


def running_is_success(cls: T) -> T:
    return py_trees.meta.running_is_success(cls)


def success_is_failure(cls: T) -> T:
    return py_trees.meta.success_is_failure(cls)


def failure_is_success(cls: T) -> T:
    return py_trees.meta.failure_is_success(cls)


def running_is_failure(cls: T) -> T:
    return py_trees.meta.running_is_failure(cls)


def failure_is_running(cls: T) -> T:
    return py_trees.meta.failure_is_running(cls)


def success_is_running(cls: T) -> T:
    return py_trees.meta.success_is_running(cls)


def anything_is_success(cls: T) -> T:
    return running_is_success(failure_is_success(cls))


def anything_is_failure(cls: T) -> T:
    return running_is_failure(success_is_failure(cls))


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
    god_map = GodMap()
    tree_nodes: Dict[str, ManagerNode]

    @profile
    def __init__(self, tree=None):
        self.action_server_name = self.god_map.get_data(identifier.action_server_name)
        self.config = self.god_map.get_data(identifier.giskard)

        if tree is None:
            self.tree = BehaviourTree(self.grow_giskard())
            self.setup()
        else:
            self.tree = tree
        self.tree_nodes = {}
        # self.god_map.get_data(identifier.world).reset_cache()
        # self.god_map.get_data(identifier.collision_scene).reset_collision_blacklist()

        self.__init_map(self.tree.root, None, 0)
        # self.render()

    def live(self):
        sleeper = rospy.Rate(1 / self.god_map.get_data(identifier.tree_tick_rate))
        logging.loginfo('giskard is ready')
        t = time()
        while not rospy.is_shutdown():
            try:
                self.tick()
                sleeper.sleep()
            except KeyboardInterrupt:
                break
        logging.loginfo('giskard died')

    def tick(self):
        self.tree.tick()

    @abc.abstractmethod
    def configure_visualization_marker(self,
                                       add_to_sync: Optional[bool] = None,
                                       add_to_planning: Optional[bool] = None,
                                       add_to_control_loop: Optional[bool] = None):
        ...

    @abc.abstractmethod
    def configure_max_trajectory_length(self, enabled: bool, length: float):
        ...

    @abc.abstractmethod
    def sync_joint_state_topic(self, group_name: str, topic_name: str):
        ...

    @abc.abstractmethod
    def sync_odometry_topic(self, topic_name: str, joint_name: PrefixName):
        ...

    @abc.abstractmethod
    def add_follow_joint_traj_action_server(self, namespace: str, state_topic: str, group_name: str,
                                            fill_velocity_values: bool):
        ...

    @abc.abstractmethod
    def add_joint_velocity_controllers(self, namespaces: List[str]):
        ...

    @abc.abstractmethod
    def add_joint_velocity_group_controllers(self, namespaces: List[str]):
        ...

    @abc.abstractmethod
    def add_cmd_vel_publisher(self, joint_name: PrefixName):
        ...

    @abc.abstractmethod
    def add_base_traj_action_server(self, cmd_vel_topic: str, track_only_velocity: bool = False,
                                    joint_name: PrefixName = None):
        ...

    @abc.abstractmethod
    def base_tracking_enabled(self) -> bool:
        ...

    @abc.abstractmethod
    def add_evaluate_debug_expressions(self):
        ...

    @abc.abstractmethod
    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        ...

    @abc.abstractmethod
    def add_plot_trajectory(self, normalize_position: bool = False, wait: bool = False):
        ...

    @abc.abstractmethod
    def add_plot_debug_trajectory(self, normalize_position: bool = False, wait: bool = False):
        ...

    @abc.abstractmethod
    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False,
                              publish_lbA: bool = False, publish_ubA: bool = False,
                              publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False,
                              publish_weights: bool = False, publish_g: bool = False,
                              publish_debug: bool = False, *args, **kwargs):
        ...

    @abc.abstractmethod
    def add_debug_marker_publisher(self):
        ...

    @abc.abstractmethod
    def add_tf_publisher(self, include_prefix: bool = False, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        ...

    def setup(self, timeout=30):
        self.tree.setup(timeout)

    def kill_all_services(self):
        self.tree.blackboard_exchange.get_blackboard_variables_srv.shutdown()
        self.tree.blackboard_exchange.open_blackboard_watcher_srv.shutdown()
        self.tree.blackboard_exchange.close_blackboard_watcher_srv.shutdown()
        for value in self.god_map.get_data(identifier.tree_manager).tree_nodes.values():
            node = value.node
            for attribute_name, attribute in vars(node).items():
                if isinstance(attribute, rospy.Service):
                    attribute.shutdown(reason='life is pain')

    def grow_giskard(self):
        raise NotImplementedError()

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
        path = self.god_map.get_data(identifier.tmp_folder) + 'tree'
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
            # if isinstance(root, AsyncBehavior) \
            #         or (hasattr(root, 'original') and isinstance(root.original, AsyncBehavior)):
            #     children = []
            #     names2 = []
            #     for name, child in root.get_children().items():
            #         children.append(child)
            #         names2.append(name)
            # else:
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
                # if isinstance(original_c, GiskardBehavior) and not isinstance(original_c, AsyncBehavior):
                #     function_names = ['__init__', 'setup', 'initialise', 'update']
                #     function_name_padding = 20
                #     entry_name_padding = 8
                #     number_padding = function_name_padding - entry_name_padding
                #     if hasattr(original_c, '__times'):
                #         time_dict = original_c.__times
                #     else:
                #         time_dict = {}
                #     for function_name in function_names:
                #         if function_name in time_dict:
                #             times = time_dict[function_name]
                #             average_time = np.average(times)
                #             total_time = np.sum(times)
                #             if total_time > 1:
                #                 color = 'red'
                #             proposed_dot_name += f'\n{function_name.ljust(function_name_padding, "-")}' \
                #                                  f'\n{"  avg".ljust(entry_name_padding)}{f"={average_time:.3}".ljust(number_padding)}' \
                #                                  f'\n{"  sum".ljust(entry_name_padding)}{f"={total_time:.3}".ljust(number_padding)}'
                #         else:
                #             proposed_dot_name += f'\n{function_name.ljust(function_name_padding, "-")}'

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


class StandAlone(TreeManager):
    sync_name: str = 'Synchronize'
    closed_loop_control_name: str = 'closed loop control'
    plan_postprocessing_name: str = 'plan postprocessing'
    planning2_name: str = 'planning II'

    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUpPlanning('CleanUpPlanning'))
        root.add_child(NewTrajectory('NewTrajectory'))
        root.add_child(self.grow_process_goal())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    def grow_wait_for_goal(self):
        wait_for_goal = Sequence('wait for goal')
        wait_for_goal.add_child(self.grow_Synchronize())
        wait_for_goal.add_child(GoalReceived('has goal?',
                                             self.action_server_name,
                                             MoveAction))
        return wait_for_goal

    def grow_Synchronize(self):
        sync = Sequence(self.sync_name)
        sync.add_child(WorldUpdater('update world'))
        sync.add_child(SyncTfFrames('sync tf frames1'))
        sync.add_child(CollisionSceneUpdater('update collision scene'))
        return sync

    def grow_process_goal(self):
        process_move_goal = failure_is_success(Selector)('Process goal')
        process_move_goal.add_child(success_is_failure(PublishFeedback)('publish feedback2',
                                                                        self.action_server_name,
                                                                        MoveFeedback.PLANNING))
        process_move_goal.add_child(self.grow_process_move_commands())
        process_move_goal.add_child(ExceptionToExecute('clear exception'))
        process_move_goal.add_child(failure_is_running(CommandsRemaining)('commands remaining?'))
        return process_move_goal

    def grow_process_move_commands(self):
        process_move_cmd = success_is_failure(Sequence)('Process move commands')
        process_move_cmd.add_child(SetCmd('set move cmd', self.action_server_name))
        process_move_cmd.add_child(self.grow_planning())
        process_move_cmd.add_child(SetErrorCode('set error code1', 'Planning'))
        return process_move_cmd

    def grow_planning(self):
        planning = failure_is_success(Sequence)('planning')
        planning.add_child(IF('command set?', identifier.next_move_goal))
        planning.add_child(RosMsgToGoal('RosMsgToGoal', self.action_server_name))
        planning.add_child(InitQPController('InitQPController'))
        planning.add_child(self.grow_planning2())
        # planning.add_child(planning_1)
        # planning.add_child(SetErrorCode('set error code'))
        planning.add_child(self.grow_plan_postprocessing())
        return planning

    def grow_planning2(self):
        planning_2 = failure_is_success(Selector)(self.planning2_name)
        planning_2.add_child(GoalCanceled('goal canceled2', self.action_server_name))
        planning_2.add_child(success_is_failure(PublishFeedback)('publish feedback1',
                                                                 self.action_server_name,
                                                                 MoveFeedback.PLANNING))
        # planning_2.add_child(success_is_failure(StartTimer)('start runtime timer'))
        planning_2.add_child(self.grow_planning3())
        return planning_2

    def grow_planning3(self):
        planning_3 = Sequence('planning III')
        # planning_3.add_child(PrintText('asdf'))
        planning_3.add_child(self.grow_closed_loop_control())
        # planning_3.add_child(self.grow_plan_postprocessing())
        return planning_3

    def grow_closed_loop_control(self):
        planning_4 = failure_is_success(AsyncBehavior)(self.closed_loop_control_name)
        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            planning_4.add_child(CollisionChecker('collision checker'))
        planning_4.add_child(ControllerPlugin('controller'))
        planning_4.add_child(KinSimPlugin('kin sim'))
        planning_4.add_child(LogTrajPlugin('log closed loop control'))
        # planning_4.add_child(WiggleCancel('wiggle'))
        planning_4.add_child(LoopDetector('loop detector'))
        planning_4.add_child(GoalReached('goal reached'))
        planning_4.add_child(TimePlugin('increase time closed loop'))
        planning_4.add_child(MaxTrajectoryLength('traj length check'))
        return planning_4

    def grow_plan_postprocessing(self):
        plan_postprocessing = Sequence(self.plan_postprocessing_name)
        plan_postprocessing.add_child(running_is_success(TimePlugin)('increase time plan post processing'))
        plan_postprocessing.add_child(SetZeroVelocity('set zero vel 1'))
        plan_postprocessing.add_child(running_is_success(LogTrajPlugin)('log post processing'))
        plan_postprocessing.add_child(GoalCleanUp('clean up goals'))
        return plan_postprocessing

    def configure_visualization_marker(self,
                                       add_to_sync: Optional[bool] = None,
                                       add_to_planning: Optional[bool] = None,
                                       add_to_control_loop: Optional[bool] = None):
        if add_to_sync is not None and add_to_sync:
            self.insert_node(VisualizationBehavior('visualization'), self.sync_name)
        if add_to_planning is not None and add_to_planning:
            self.insert_node(success_is_failure(VisualizationBehavior)('visualization'), self.planning2_name, 2)
            self.insert_node(anything_is_success(VisualizationBehavior)('visualization'),
                             self.plan_postprocessing_name)
        if add_to_control_loop is not None and add_to_control_loop:
            self.insert_node(success_is_running(VisualizationBehavior)('visualization'), self.closed_loop_control_name)

    def configure_max_trajectory_length(self, enabled: bool, length: float):
        nodes = self.get_nodes_of_type(MaxTrajectoryLength)
        for node in nodes:
            if enabled:
                self.enable_node(node.name)
            else:
                self.disable_node(node.name)
            node.length = length

    def add_joint_velocity_group_controllers(self, namespaces: List[str]):
        # todo new abstract decorator that uses this as default implementation
        current_function_name = inspect.currentframe().f_code.co_name
        NotImplementedError(f'stand alone mode doesn\'t support {current_function_name}.')

    def add_follow_joint_traj_action_server(self, namespace: str, state_topic: str, group_name: str,
                                            fill_velocity_values: bool):
        # todo new abstract decorator that uses this as default implementation
        current_function_name = inspect.currentframe().f_code.co_name
        NotImplementedError(f'stand alone mode doesn\'t support {current_function_name}.')

    def add_joint_velocity_controllers(self, namespaces: List[str]):
        # todo new abstract decorator that uses this as default implementation
        current_function_name = inspect.currentframe().f_code.co_name
        NotImplementedError(f'stand alone mode doesn\'t support {current_function_name}.')

    def add_cmd_vel_publisher(self, joint_name: PrefixName):
        current_function_name = inspect.currentframe().f_code.co_name
        NotImplementedError(f'stand alone mode doesn\'t support {current_function_name}.')

    def add_base_traj_action_server(self, cmd_vel_topic: str, track_only_velocity: bool = False,
                                    joint_name: PrefixName = None):
        current_function_name = inspect.currentframe().f_code.co_name
        NotImplementedError(f'stand alone mode doesn\'t support {current_function_name}.')

    def base_tracking_enabled(self) -> bool:
        return False

    def add_evaluate_debug_expressions(self):
        nodes = self.get_nodes_of_type(EvaluateDebugExpressions)
        if len(nodes) == 0:
            self.insert_node_behind_every_node_of_type(ControllerPlugin,
                                                       EvaluateDebugExpressions('evaluate debug expressions'))

    def add_plot_trajectory(self, normalize_position: bool = False, wait: bool = False):
        if len(self.get_nodes_of_type(PlotTrajectory)) > 0:
            raise BehaviorTreeException(f'add_plot_trajectory is not allowed to be called twice')
        behavior = PlotTrajectory('plot trajectory', wait=wait, normalize_position=normalize_position)
        self.insert_node(behavior, self.plan_postprocessing_name)

    def add_plot_debug_trajectory(self, normalize_position: bool = False, wait: bool = False):
        if len(self.get_nodes_of_type(PlotDebugExpressions)) > 0:
            raise BehaviorTreeException(f'add_plot_debug_trajectory is not allowed to be called twice')
        self.add_evaluate_debug_expressions()
        for node in self.get_nodes_of_type(EvaluateDebugExpressions):
            manager_node = self.tree_nodes[node.name]
            parent_node = self.tree_nodes[node.name].parent
            if parent_node.node.name == self.closed_loop_control_name:
                self.insert_node(LogDebugExpressionsPlugin('log lba'), self.closed_loop_control_name,
                                 manager_node.position + 1)
        behavior = PlotDebugExpressions('plot debug trajectory', wait=wait, normalize_position=normalize_position)
        self.insert_node(behavior, self.plan_postprocessing_name)

    def sync_6dof_joint_with_tf_frame(self, joint_name: PrefixName, tf_parent_frame: str, tf_child_frame: str):
        tf_sync_nodes = self.get_nodes_of_type(SyncTfFrames)
        for node in tf_sync_nodes:
            node.sync_6dof_joint_with_tf_frame(joint_name, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, group_name: str, topic_name: str):
        behavior = SyncConfiguration(group_name=group_name, joint_state_topic=topic_name)
        self.insert_node(behavior, self.sync_name, 2)

    def sync_odometry_topic(self, topic_name: str, joint_name: PrefixName):
        behavior = SyncOdometry(topic_name, joint_name)
        self.insert_node(behavior, self.sync_name, 2)

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False, publish_lbA: bool = False,
                              publish_ubA: bool = False, publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False, publish_weights: bool = False,
                              publish_g: bool = False, publish_debug: bool = False, *args, **kwargs):
        self.add_evaluate_debug_expressions()
        node = PublishDebugExpressions('qp data publisher',
                                       publish_lb=publish_lb,
                                       publish_ub=publish_ub,
                                       publish_lbA=publish_lbA,
                                       publish_ubA=publish_ubA,
                                       publish_bE=publish_bE,
                                       publish_Ax=publish_Ax,
                                       publish_Ex=publish_Ex,
                                       publish_xdot=publish_xdot,
                                       publish_weights=publish_weights,
                                       publish_g=publish_g,
                                       publish_debug=publish_debug)
        self.insert_node_behind_node_of_type(self.closed_loop_control_name, EvaluateDebugExpressions, node)

    def add_debug_marker_publisher(self):
        self.add_evaluate_debug_expressions()
        node = DebugMarkerPublisher('debug marker_publisher')
        self.insert_node_behind_every_node_of_type(EvaluateDebugExpressions, node)

    def add_tf_publisher(self, include_prefix: bool = False, tf_topic: str = 'tf',
                         mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects):
        node = TFPublisher('publish tf', mode=mode, tf_topic=tf_topic, include_prefix=include_prefix)
        self.insert_node(node, self.sync_name)


class OpenLoop(StandAlone):
    move_robots_name = 'move robots'
    execution_name = 'execution'
    base_closed_loop_control_name = 'base sequence'

    def add_follow_joint_traj_action_server(self, namespace: str, state_topic: str, group_name: str,
                                            fill_velocity_values: bool):
        behavior = SendFollowJointTrajectory(action_namespace=namespace, state_topic=state_topic, group_name=group_name,
                                             fill_velocity_values=fill_velocity_values)
        self.insert_node(behavior, self.move_robots_name)

    def add_base_traj_action_server(self, cmd_vel_topic: str, track_only_velocity: bool = False,
                                    joint_name: PrefixName = None):
        # todo handle if this is called twice
        self.insert_node_behind_node_of_type(self.execution_name, SetTrackingStartTime, CleanUpBaseController('CleanUpBaseController', clear_markers=False))
        self.insert_node_behind_node_of_type(self.execution_name, SetTrackingStartTime, InitQPController('InitQPController for base'))
        self.insert_node_behind_node_of_type(self.execution_name, SetTrackingStartTime, SetDriveGoals('SetupBaseTrajConstraints'))

        real_time_tracking = AsyncBehavior(self.base_closed_loop_control_name)
        self.insert_node(real_time_tracking, self.move_robots_name)
        sync_tf_nodes = self.get_nodes_of_type(SyncTfFrames)
        for node in sync_tf_nodes:
            self.insert_node(success_is_running(SyncTfFrames)(node.name + '*', node.joint_map),
                             self.base_closed_loop_control_name)
        odom_nodes = self.get_nodes_of_type(SyncOdometry)
        for node in odom_nodes:
            new_node = success_is_running(SyncOdometry)(odometry_topic=node.odometry_topic,
                                                        joint_name=node.joint_name,
                                                        name_suffix='*')
            self.insert_node(new_node, self.base_closed_loop_control_name)
        self.insert_node(RosTime('time'), self.base_closed_loop_control_name)
        self.insert_node(ControllerPlugin('base controller'), self.base_closed_loop_control_name)
        self.insert_node(RealKinSimPlugin('base kin sim'), self.base_closed_loop_control_name)
        # todo debugging
        # if self.god_map.get_data(identifier.PlotDebugTF_enabled):
        #     real_time_tracking.add_child(DebugMarkerPublisher('debug marker publisher'))
        # if self.god_map.unsafe_get_data(identifier.PublishDebugExpressions)['enabled_base']:
        #     real_time_tracking.add_child(PublishDebugExpressions('PublishDebugExpressions',
        #                                                          **self.god_map.unsafe_get_data(
        #                                                              identifier.PublishDebugExpressions)))
        # if self.god_map.unsafe_get_data(identifier.PlotDebugTF)['enabled_base']:
        #     real_time_tracking.add_child(DebugMarkerPublisher('debug marker publisher',
        #                                                       **self.god_map.unsafe_get_data(
        #                                                           identifier.PlotDebugTF)))

        self.insert_node(SendTrajectoryToCmdVel(cmd_vel_topic=cmd_vel_topic,
                                                track_only_velocity=track_only_velocity,
                                                joint_name=joint_name), self.base_closed_loop_control_name)

    def base_tracking_enabled(self) -> bool:
        return len(self.get_nodes_of_type(SendTrajectoryToCmdVel)) > 0

    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUpPlanning('CleanUpPlanning'))
        root.add_child(NewTrajectory('NewTrajectory'))
        root.add_child(self.grow_process_goal())
        root.add_child(self.grow_execution())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    def grow_Synchronize(self):
        sync = Sequence('Synchronize')
        sync.add_child(WorldUpdater('update world'))
        sync.add_child(SyncTfFrames('sync tf frames3'))
        # hardware_config: HardwareConfig = self.god_map.get_data(identifier.hardware_config)
        # for kwargs in hardware_config.joint_state_topics_kwargs:
        #     sync.add_child(running_is_success(SyncConfiguration)(**kwargs))
        # for odometry_kwargs in hardware_config.odometry_node_kwargs:
        #     sync.add_child(running_is_success(SyncOdometry)(**odometry_kwargs))
        # if self.god_map.get_data(identifier.TFPublisher_enabled):
        #     sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
        sync.add_child(CollisionSceneUpdater('update collision scene'))
        sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
        return sync

    def grow_execution(self):
        execution = failure_is_success(Sequence)(self.execution_name)
        execution.add_child(IF('execute?', identifier.execute))
        execution.add_child(SetTrackingStartTime('start start time'))
        execution.add_child(self.grow_monitor_execution())
        execution.add_child(SetZeroVelocity('set zero vel 2'))
        return execution

    def grow_monitor_execution(self):
        monitor_execution = failure_is_success(Selector)('monitor execution')
        monitor_execution.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                        self.god_map.get_data(
                                                                            identifier.action_server_name),
                                                                        MoveFeedback.EXECUTION))
        monitor_execution.add_child(self.grow_execution_cancelled())
        monitor_execution.add_child(self.grow_move_robots())
        monitor_execution.add_child(SetErrorCode('set error code2', 'Execution'))
        return monitor_execution

    def grow_execution_cancelled(self):
        execute_canceled = Sequence('execute canceled')
        execute_canceled.add_child(GoalCanceled('goal canceled1', self.action_server_name))
        execute_canceled.add_child(SetErrorCode('set error code3', 'Execution'))
        return execute_canceled

    def grow_move_robots(self):
        execution_action_server = Parallel(self.move_robots_name,
                                           policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        return execution_action_server

    def add_qp_data_publisher(self, publish_lb: bool = False, publish_ub: bool = False, publish_lbA: bool = False,
                              publish_ubA: bool = False, publish_bE: bool = False, publish_Ax: bool = False,
                              publish_Ex: bool = False, publish_xdot: bool = False, publish_weights: bool = False,
                              publish_g: bool = False, publish_debug: bool = False, add_to_base: bool = False,
                              *args, **kwargs):
        self.add_evaluate_debug_expressions()
        node = PublishDebugExpressions('qp data publisher',
                                       publish_lb=publish_lb,
                                       publish_ub=publish_ub,
                                       publish_lbA=publish_lbA,
                                       publish_ubA=publish_ubA,
                                       publish_bE=publish_bE,
                                       publish_Ax=publish_Ax,
                                       publish_Ex=publish_Ex,
                                       publish_xdot=publish_xdot,
                                       publish_weights=publish_weights,
                                       publish_g=publish_g,
                                       publish_debug=publish_debug)
        if not add_to_base:
            self.insert_node_behind_node_of_type(self.closed_loop_control_name, EvaluateDebugExpressions, node)
        else:
            self.insert_node_behind_node_of_type(self.base_closed_loop_control_name, EvaluateDebugExpressions, node)


class ClosedLoop(OpenLoop):

    def add_joint_velocity_controllers(self, namespaces: List[str]):
        behavior = JointVelController(namespaces=namespaces)
        self.insert_node_behind_node_of_type(self.closed_loop_control_name, RealKinSimPlugin, behavior)

    def add_joint_velocity_group_controllers(self, namespace: str):
        behavior = JointGroupVelController(namespace)
        self.insert_node_behind_node_of_type(self.closed_loop_control_name, RealKinSimPlugin, behavior)

    def add_base_traj_action_server(self, cmd_vel_topic: str, track_only_velocity: bool = False,
                                    joint_name: PrefixName = None):
        behavior = SendCmdVel(cmd_vel_topic, joint_name=joint_name)
        self.insert_node_behind_node_of_type(self.closed_loop_control_name, RealKinSimPlugin, behavior)

    def grow_planning3(self):
        planning_3 = Sequence('planning III')
        # planning_3.add_child(PrintText('asdf'))
        planning_3.add_child(SetTrackingStartTime('start time', offset=0.0))
        planning_3.add_child(self.grow_closed_loop_control())
        # planning_3.add_child(self.grow_plan_postprocessing())
        return planning_3

    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUpPlanning('CleanUpPlanning'))
        root.add_child(NewTrajectory('NewTrajectory'))
        root.add_child(self.grow_process_goal())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    def sync_joint_state_topic(self, group_name: str, topic_name: str):
        super().sync_joint_state_topic(group_name, topic_name)
        behavior = success_is_running(SyncConfiguration2)(group_name=group_name, joint_state_topic=topic_name)
        self.insert_node(behavior, self.closed_loop_control_name, 0)

    def sync_odometry_topic(self, topic_name: str, joint_name: PrefixName):
        super().sync_odometry_topic(topic_name, joint_name)
        behavior = success_is_running(SyncOdometryNoLock)(topic_name, joint_name)
        self.insert_node(behavior, self.closed_loop_control_name, 0)

    def grow_closed_loop_control(self):
        planning_4 = failure_is_success(AsyncBehavior)(self.closed_loop_control_name)
        planning_4.add_child(success_is_running(SyncTfFrames)('sync tf frames close loop'))
        planning_4.add_child(success_is_running(NotifyStateChange)())
        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            planning_4.add_child(CollisionChecker('collision checker'))
        planning_4.add_child(ControllerPlugin('controller'))
        planning_4.add_child(RosTime())
        planning_4.add_child(RealKinSimPlugin('kin sim'))
        # planning_4.add_child(LoopDetector('loop detector'))
        planning_4.add_child(GoalReached('goal reached', real_time=True))
        planning_4.add_child(MaxTrajectoryLength('traj length check', real_time=True))
        planning_4.add_child(GoalDone('goal done check'))
        return planning_4

# def sanity_check(god_map):
#     check_velocity_limits_reachable(god_map)


# def check_velocity_limits_reachable(god_map):
#     robot = god_map.get_data(identifier.robot)
#     sample_period = god_map.get_data(identifier.sample_period)
#     prediction_horizon = god_map.get_data(identifier.prediction_horizon)
#     print_help = False
#     for joint_name in robot.get_joint_names():
#         velocity_limit = robot.get_joint_limit_expr_evaluated(joint_name, 1, god_map)
#         jerk_limit = robot.get_joint_limit_expr_evaluated(joint_name, 3, god_map)
#         velocity_limit_horizon = max_velocity_from_horizon_and_jerk(prediction_horizon, jerk_limit, sample_period)
#         if velocity_limit_horizon < velocity_limit:
#             logging.logwarn('Joint \'{}\' '
#                             'can reach at most \'{:.4}\' '
#                             'with to prediction horizon of \'{}\' '
#                             'and jerk limit of \'{}\', '
#                             'but limit in urdf/config is \'{}\''.format(joint_name,
#                                                                         velocity_limit_horizon,
#                                                                         prediction_horizon,
#                                                                         jerk_limit,
#                                                                         velocity_limit
#                                                                         ))
#             print_help = True
#     if print_help:
#         logging.logwarn('Check utils.py/max_velocity_from_horizon_and_jerk for help.')
