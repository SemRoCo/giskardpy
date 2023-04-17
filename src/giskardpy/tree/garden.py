from collections import defaultdict
from time import time
from typing import Type, TypeVar, Union

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
from giskardpy.configs.data_types import CollisionCheckerLib, HardwareConfig
from giskardpy.god_map import GodMap
from giskardpy.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUp, CleanUpPlanning, CleanUpBaseController
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.collision_marker import CollisionMarker
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.commands_remaining import CommandsRemaining
from giskardpy.tree.behaviors.evaluate_debug_expressions import EvaluateDebugExpressions
from giskardpy.tree.behaviors.exception_to_execute import ExceptionToExecute
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.goal_reached import GoalReached
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.init_qp_controller import InitQPController
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.instantaneous_controller_base import ControllerPluginBase
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.loop_detector import LoopDetector
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.behaviors.new_trajectory import NewTrajectory
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
from giskardpy.tree.behaviors.set_cmd import SetCmd
from giskardpy.tree.behaviors.set_error_code import SetErrorCode
from giskardpy.tree.behaviors.set_tracking_start_time import SetTrackingStartTime
from giskardpy.tree.behaviors.setup_base_traj_constraints import SetDriveGoals
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_odometry import SyncOdometry
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


class ManagerNode:
    def __init__(self, node, parent, position: int):
        """
        :param node: the behavior that is represented by this ManagerNode
        :type node: ManagerNode
        :param parent: the parent of the behavior that is represented by this ManagerNode
        :type parent: ManagerNode
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

    def disable_child(self, manager_node):
        """
        marks the given manager node as disabled in the internal tree representation and removes it to the behavior tree
        :param manager_node:
        :type manager_node: ManagerNode
        :return:
        """
        self.enabled_children.remove(manager_node)
        self.disabled_children.add(manager_node)
        if isinstance(self.node, AsyncBehavior):
            self.node.remove_child(manager_node.node.name)
        else:
            self.node.remove_child(manager_node.node)

    def enable_child(self, manager_node):
        """
        marks the given manager node as enabled in the internal tree representation and adds it to the behavior tree
        :param manager_node:
        :type manager_node: TreeManager.ManagerNode
        :return:
        """
        self.disabled_children.remove(manager_node)
        self.enabled_children.add(manager_node)
        if isinstance(self.node, AsyncBehavior):
            self.node.add_child(manager_node.node)
        else:
            idx = self.enabled_children.index(manager_node)
            self.node.insert_child(manager_node.node, idx)

    def add_child(self, manager_node):
        """
        adds the given manager node to the internal tree map and the corresponding behavior to the behavior tree
        :param manager_node:
        :type manager_node: TreeManager.ManagerNode
        :return:
        """
        if isinstance(self.node, AsyncBehavior):
            self.enabled_children.add(manager_node)
            self.node.add_child(manager_node.node)
        else:
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
        if isinstance(self.node, AsyncBehavior):
            if manager_node in self.enabled_children:
                self.enabled_children.remove(manager_node)
                self.node.remove_child(manager_node.node.name)
            elif manager_node in self.disabled_children:
                self.disabled_children.remove(manager_node)
            else:
                raise RuntimeError(
                    'could not remove node from parent. this probably means that the tree is inconsistent')
        else:
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



class TreeManager:
    god_map = GodMap()

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
        self.god_map.get_data(identifier.world).reset_cache()
        self.god_map.get_data(identifier.collision_scene).reset_collision_blacklist()

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
        if isinstance(node, AsyncBehavior):
            children = node._children
            for child_name in children:
                child_node = ManagerNode(node=children[child_name], parent=manager_node, position=0)
                self.tree_nodes[child_name] = child_node
                manager_node.enabled_children.add(child_node)
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

    def enable_node(self, node_name):
        """
        enables the node with the given name
        :param node_name: the name of the node
        :type node_name: str
        :return:
        """
        t = self.tree_nodes[node_name]
        if t.parent is not None:
            t.parent.enable_child(t)
        else:
            logging.loginfo('root node')

    def insert_node(self, node, parent_name, position=-1):
        """
        inserts a node into the behavior tree.
        :param node: the node that will be inserted
        :type node: py_trees.behaviour.Behaviour
        :param parent_name: the name of the parent node where the node will be inserted
        :type parent_name: str
        :param position: the node will be inserted as the nth child with n = len([x for x in children if x.position < position])
        :type position: int
        :return:
        """
        if node.name in self.tree_nodes:
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
            if isinstance(root, AsyncBehavior) \
                    or (hasattr(root, 'original') and isinstance(root.original, AsyncBehavior)):
                children = []
                names2 = []
                for name, child in root.get_children().items():
                    children.append(child)
                    names2.append(name)
            else:
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
                    number_padding = function_name_padding-entry_name_padding
                    if hasattr(original_c, '__times'):
                        time_dict = original_c.__times
                    else:
                        time_dict = {}
                    for function_name in function_names:
                        if function_name in time_dict:
                            times = time_dict[function_name]
                            average_time = np.average(times)
                            total_time = np.sum(times)
                            if total_time > 1:
                                color = 'red'
                            proposed_dot_name += f'\n{function_name.ljust(function_name_padding, "-")}' \
                                                 f'\n{"  avg".ljust(entry_name_padding)}{f"={average_time:.3}".ljust(number_padding)}' \
                                                 f'\n{"  sum".ljust(entry_name_padding)}{f"={total_time:.3}".ljust(number_padding)}'
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


class StandAlone(TreeManager):
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
        sync = Sequence('Synchronize')
        sync.add_child(WorldUpdater('update world'))
        sync.add_child(SyncTfFrames('sync tf frames',
                                    **self.god_map.unsafe_get_data(identifier.SyncTfFrames)))
        if self.god_map.get_data(identifier.TFPublisher_enabled):
            sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
        sync.add_child(CollisionSceneUpdater('update collision scene'))
        if self.god_map.get_data(identifier.enable_VisualizationBehavior):
            sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
        return sync

    def grow_process_goal(self):
        process_move_goal = failure_is_success(Selector)('Process goal')
        process_move_goal.add_child(success_is_failure(PublishFeedback)('publish feedback',
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
        process_move_cmd.add_child(SetErrorCode('set error code', 'Planning'))
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
        planning_2 = failure_is_success(Selector)('planning II')
        planning_2.add_child(GoalCanceled('goal canceled', self.action_server_name))
        planning_2.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                 self.action_server_name,
                                                                 MoveFeedback.PLANNING))
        if self.god_map.get_data(identifier.enable_VisualizationBehavior) \
                and not self.god_map.get_data(identifier.VisualizationBehavior_in_planning_loop):
            planning_2.add_child(running_is_failure(VisualizationBehavior)('visualization'))
        if self.god_map.get_data(identifier.enable_CPIMarker) \
                and self.god_map.get_data(identifier.collision_checker) is not None \
                and not self.god_map.get_data(identifier.CPIMarker_in_planning_loop):
            planning_2.add_child(running_is_failure(CollisionMarker)('cpi marker'))
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
        planning_4 = failure_is_success(AsyncBehavior)('closed loop control')
        if self.god_map.get_data(identifier.enable_VisualizationBehavior) \
                and self.god_map.get_data(identifier.VisualizationBehavior_in_planning_loop):
            planning_4.add_child(VisualizationBehavior('visualization'))
        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            planning_4.add_child(CollisionChecker('collision checker'))
            if self.god_map.get_data(identifier.enable_CPIMarker) \
                    and self.god_map.get_data(identifier.CPIMarker_in_planning_loop):
                planning_4.add_child(CollisionMarker('cpi marker'))
        planning_4.add_child(ControllerPlugin('controller'))
        if self.god_map.get_data(identifier.debug_expr_needed):
            planning_4.add_child(EvaluateDebugExpressions('evaluate debug expressions'))
        planning_4.add_child(KinSimPlugin('kin sim'))
        planning_4.add_child(LogTrajPlugin('log'))
        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            planning_4.add_child(LogDebugExpressionsPlugin('log lba'))
        if self.god_map.get_data(identifier.PlotDebugTF_enabled):
            planning_4.add_child(DebugMarkerPublisher('debug marker publisher'))
        if self.god_map.unsafe_get_data(identifier.PublishDebugExpressions)['enabled']:
            planning_4.add_child(PublishDebugExpressions('PublishDebugExpressions',
                                                         **self.god_map.unsafe_get_data(
                                                             identifier.PublishDebugExpressions)))
        # planning_4.add_child(WiggleCancel('wiggle'))
        planning_4.add_child(LoopDetector('loop detector'))
        planning_4.add_child(GoalReached('goal reached'))
        planning_4.add_child(TimePlugin())
        if self.god_map.get_data(identifier.MaxTrajectoryLength_enabled):
            kwargs = self.god_map.get_data(identifier.MaxTrajectoryLength)
            planning_4.add_child(MaxTrajectoryLength('traj length check', **kwargs))
        return planning_4

    def grow_plan_postprocessing(self):
        plan_postprocessing = Sequence('plan postprocessing')
        plan_postprocessing.add_child(running_is_success(TimePlugin)())
        plan_postprocessing.add_child(SetZeroVelocity())
        plan_postprocessing.add_child(running_is_success(LogTrajPlugin)('log'))
        if self.god_map.get_data(identifier.enable_VisualizationBehavior) \
                and not self.god_map.get_data(identifier.VisualizationBehavior_in_planning_loop):
            plan_postprocessing.add_child(
                anything_is_success(VisualizationBehavior)('visualization', ensure_publish=True))
        if self.god_map.get_data(identifier.enable_CPIMarker) \
                and self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none \
                and not self.god_map.get_data(identifier.CPIMarker_in_planning_loop):
            plan_postprocessing.add_child(anything_is_success(CollisionMarker)('collision marker'))
        if self.god_map.get_data(identifier.PlotTrajectory_enabled):
            kwargs = self.god_map.get_data(identifier.PlotTrajectory)
            plan_postprocessing.add_child(PlotTrajectory('plot trajectory', **kwargs))
        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            kwargs = self.god_map.get_data(identifier.PlotDebugTrajectory)
            plan_postprocessing.add_child(PlotDebugExpressions('plot debug expressions', **kwargs))
        return plan_postprocessing


class OpenLoop(StandAlone):
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
        sync.add_child(SyncTfFrames('sync tf frames',
                                    **self.god_map.unsafe_get_data(identifier.SyncTfFrames)))
        hardware_config: HardwareConfig = self.god_map.get_data(identifier.hardware_config)
        for kwargs in hardware_config.joint_state_topics_kwargs:
            sync.add_child(running_is_success(SyncConfiguration)(**kwargs))
        for odometry_kwargs in hardware_config.odometry_node_kwargs:
            sync.add_child(running_is_success(SyncOdometry)(**odometry_kwargs))
        if self.god_map.get_data(identifier.TFPublisher_enabled):
            sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
        sync.add_child(CollisionSceneUpdater('update collision scene'))
        sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
        return sync

    def grow_execution(self):
        execution = failure_is_success(Sequence)('execution')
        execution.add_child(IF('execute?', identifier.execute))
        if self.add_real_time_tracking:
            execution.add_child(CleanUpBaseController('CleanUpBaseController', clear_markers=False))
            execution.add_child(SetDriveGoals('SetupBaseTrajConstraints'))
            execution.add_child(InitQPController('InitQPController for base'))
        execution.add_child(SetTrackingStartTime('start start time'))
        execution.add_child(self.grow_monitor_execution())
        execution.add_child(SetZeroVelocity())
        return execution

    def grow_monitor_execution(self):
        monitor_execution = failure_is_success(Selector)('monitor execution')
        monitor_execution.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                        self.god_map.get_data(
                                                                            identifier.action_server_name),
                                                                        MoveFeedback.EXECUTION))
        monitor_execution.add_child(self.grow_execution_cancelled())
        monitor_execution.add_child(self.grow_move_robots())
        monitor_execution.add_child(SetErrorCode('set error code', 'Execution'))
        return monitor_execution

    def grow_execution_cancelled(self):
        execute_canceled = Sequence('execute canceled')
        execute_canceled.add_child(GoalCanceled('goal canceled', self.action_server_name))
        execute_canceled.add_child(SetErrorCode('set error code', 'Execution'))
        return execute_canceled

    @property
    def add_real_time_tracking(self):
        drive_interfaces = self.config.hardware_config.send_trajectory_to_cmd_vel_kwargs
        return len(drive_interfaces) > 0

    def grow_move_robots(self):
        execution_action_server = Parallel('move robots',
                                           policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        hardware_config: HardwareConfig = self.god_map.get_data(identifier.hardware_config)
        for follow_joint_trajectory_config in hardware_config.follow_joint_trajectory_interfaces_kwargs:
            execution_action_server.add_child(SendFollowJointTrajectory(**follow_joint_trajectory_config))
        if self.add_real_time_tracking:
            for drive_interface in hardware_config.send_trajectory_to_cmd_vel_kwargs:
                real_time_tracking = AsyncBehavior('base sequence')
                real_time_tracking.add_child(success_is_running(SyncTfFrames)('sync tf frames',
                                                                              **self.god_map.unsafe_get_data(
                                                                                  identifier.SyncTfFrames)))
                for odometry_kwargs in hardware_config.odometry_node_kwargs:
                    real_time_tracking.add_child(SyncOdometry(**odometry_kwargs))
                real_time_tracking.add_child(RosTime('time'))
                real_time_tracking.add_child(ControllerPluginBase('base controller'))
                real_time_tracking.add_child(RealKinSimPlugin('kin sim'))
                if self.god_map.get_data(identifier.PlotDebugTF_enabled):
                    real_time_tracking.add_child(DebugMarkerPublisher('debug marker publisher'))
                if self.god_map.unsafe_get_data(identifier.PublishDebugExpressions)['enabled_base']:
                    real_time_tracking.add_child(PublishDebugExpressions('PublishDebugExpressions',
                                                                         **self.god_map.unsafe_get_data(
                                                                             identifier.PublishDebugExpressions)))
                if self.god_map.unsafe_get_data(identifier.PlotDebugTF)['enabled_base']:
                    real_time_tracking.add_child(DebugMarkerPublisher('debug marker publisher',
                                                                         **self.god_map.unsafe_get_data(
                                                                             identifier.PlotDebugTF)))

                real_time_tracking.add_child(SendTrajectoryToCmdVel(**drive_interface))
                execution_action_server.add_child(real_time_tracking)
        return execution_action_server


class ClosedLoop(OpenLoop):

    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUp('cleanup'))
        root.add_child(self.grow_process_goal())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    # def grow_sync_branch(self):
    #     sync = Sequence('Synchronize')
    #     sync.add_child(WorldUpdater('update world'))
    #     sync.add_child(running_is_success(SyncConfiguration)('update robot configuration', RobotName))
    #     sync.add_child(SyncLocalization('update robot localization', RobotName))
    #     sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
    #     sync.add_child(CollisionSceneUpdater('update collision scene'))
    #     sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
    #     return sync

    def grow_planning3(self):
        planning_3 = Sequence('planning III', sleep=0)
        planning_3.add_child(self.grow_closed_loop_control())
        return planning_3

    def grow_closed_loop_control(self):
        planning_4 = AsyncBehavior('planning IIII')
        action_servers = self.god_map.get_data(identifier.robot_interface)
        behaviors = get_all_classes_in_package(giskardpy.tree.behaviors)
        for i, (execution_action_server_name, params) in enumerate(action_servers.items()):
            C = behaviors[params['plugin']]
            del params['plugin']
            planning_4.add_child(C(execution_action_server_name, **params))
        #planning_4.add_child(SyncConfiguration2('update robot configuration',
        #                                         self.god_map.unsafe_get_data(identifier.robot_group_name)))
        planning_4.add_child(LogTrajPlugin('log'))
        if self.god_map.get_data(identifier.collision_checker) is not None:
            planning_4.add_child(CollisionChecker('collision checker'))
        planning_4.add_child(ControllerPlugin('controller'))
        planning_4.add_child(KinSimPlugin('kin sim'))

        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            planning_4.add_child(LogDebugExpressionsPlugin('log lba'))
        # planning_4.add_plugin(WiggleCancel('wiggle'))
        # planning_4.add_plugin(LoopDetector('loop detector'))
        planning_4.add_child(GoalReached('goal reached'))
        planning_4.add_child(TimePlugin('time'))
        if self.god_map.get_data(identifier.MaxTrajectoryLength_enabled):
            kwargs = self.god_map.get_data(identifier.MaxTrajectoryLength)
            planning_4.add_child(MaxTrajectoryLength('traj length check', **kwargs))
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
