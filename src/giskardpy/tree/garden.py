from typing import Type, TypeVar, Union, Generic

import py_trees
import pydot
import rospy
from py_trees import Behaviour, Chooser, common, Blackboard, Composite
from py_trees import Selector, Sequence
from py_trees_ros.trees import BehaviourTree
from sortedcontainers import SortedList

import giskardpy
from giskard_msgs.msg import MoveAction, MoveFeedback
from giskardpy import identifier
from giskardpy.data_types import order_map
from giskardpy.god_map import GodMap
from giskardpy.model.world import WorldTree
from giskardpy.tree.behaviors.append_zero_velocity import AppendZeroVelocity
from giskardpy.tree.behaviors.cleanup import CleanUp
from giskardpy.tree.behaviors.collision_checker import CollisionChecker
from giskardpy.tree.behaviors.collision_marker import CollisionMarker
from giskardpy.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy.tree.behaviors.commands_remaining import CommandsRemaining
from giskardpy.tree.behaviors.exception_to_execute import ExceptionToExecute
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.goal_reached import GoalReachedPlugin
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.kinematic_sim import KinSimPlugin
from giskardpy.tree.behaviors.log_debug_expressions import LogDebugExpressionsPlugin
from giskardpy.tree.behaviors.log_trajectory import LogTrajPlugin
from giskardpy.tree.behaviors.loop_detector import LoopDetector
from giskardpy.tree.behaviors.max_trajectory_length import MaxTrajectoryLength
from giskardpy.tree.behaviors.plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.plugin_if import IF
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.behaviors.set_cmd import SetCmd
from giskardpy.tree.behaviors.set_error_code import SetErrorCode
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_configuration2 import SyncConfiguration2
from giskardpy.tree.behaviors.sync_localization import SyncTfFrames
from giskardpy.tree.behaviors.tf_publisher import TFPublisher
from giskardpy.tree.behaviors.time import TimePlugin
from giskardpy.tree.behaviors.update_constraints import GoalToConstraints
from giskardpy.tree.behaviors.visualization import VisualizationBehavior
from giskardpy.tree.behaviors.world_updater import WorldUpdater
from giskardpy.tree.composites.async_composite import PluginBehavior
from giskardpy.tree.composites.better_parallel import ParallelPolicy, Parallel
from giskardpy.utils import logging
from giskardpy.utils.time_collector import TimeCollector
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
        if isinstance(self.node, PluginBehavior):
            self.node.remove_plugin(manager_node.node.name)
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
        if isinstance(self.node, PluginBehavior):
            self.node.add_plugin(manager_node.node)
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
        if isinstance(self.node, PluginBehavior):
            self.enabled_children.add(manager_node)
            self.node.add_plugin(manager_node.node)
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
        if isinstance(self.node, PluginBehavior):
            if manager_node in self.enabled_children:
                self.enabled_children.remove(manager_node)
                self.node.remove_plugin(manager_node.node.name)
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


class TreeManager:
    god_map: GodMap

    @profile
    def __init__(self, god_map, tree=None):
        self.god_map = god_map
        self.action_server_name = self.god_map.get_data(identifier.action_server_name)
        world = WorldTree(self.god_map)
        world.delete_all_but_robot()

        collision_checker = self.god_map.get_data(identifier.collision_checker)
        if collision_checker == 'bpb':
            logging.loginfo('Using bpb for collision checking.')
            from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
            collision_scene = BetterPyBulletSyncer(world)
        elif collision_checker == 'pybullet':
            logging.loginfo('Using pybullet for collision checking.')
            from giskardpy.model.pybullet_syncer import PyBulletSyncer
            collision_scene = PyBulletSyncer(world)
        else:
            logging.logwarn('Unknown collision checker {}. Collision avoidance is disabled'.format(collision_checker))
            from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
            collision_scene = CollisionWorldSynchronizer(world)
            self.god_map.set_data(identifier.collision_checker, None)
        self.god_map.set_data(identifier.collision_scene, collision_scene)

        if tree is None:
            self.tree = BehaviourTree(self.grow_giskard())
            self.setup()
        else:
            self.tree = tree
        self.tree_nodes = {}
        collision_scene.reset_collision_blacklist()

        self.__init_map(self.tree.root, None, 0)
        self.render()

    @classmethod
    @profile
    def from_param_server(cls):
        god_map = GodMap.init_from_paramserver(rospy.get_name())
        god_map.set_data(identifier.timer_collector, TimeCollector(god_map))
        blackboard = Blackboard
        blackboard.god_map = god_map
        mode = god_map.get_data(identifier.control_mode)
        if mode == 'OpenLoop':
            self = OpenLoop(god_map)
        elif mode == 'ClosedLoop':
            self = ClosedLoop(god_map)
        else:
            raise KeyError('Robot interface mode \'{}\' is not supported.'.format(mode))

        god_map.set_data(identifier.tree_manager, self)
        return self

    def live(self):
        sleeper = rospy.Rate(1 / self.god_map.get_data(identifier.tree_tick_rate))
        logging.loginfo('giskard is ready')
        while not rospy.is_shutdown():
            try:
                self.tick()
                sleeper.sleep()
            except KeyboardInterrupt:
                break

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
        if isinstance(node, PluginBehavior):
            children = node.get_plugins()
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
            raise ValueError('node with that name already exists')
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

    def render(self, profile=None):
        path = self.god_map.get_data(identifier.data_folder) + 'tree'
        create_path(path)
        render_dot_tree(self.tree.root, name=path, profile=profile)


def render_dot_tree(root, visibility_level=common.VisibilityLevel.DETAIL, name=None, profile=None):
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
    graph = generate_pydot_graph(root, visibility_level, profile)
    filename_wo_extension = root.name.lower().replace(" ", "_") if name is None else name
    logging.loginfo("Writing %s.dot/svg/png" % filename_wo_extension)
    graph.write(filename_wo_extension + '.dot')
    graph.write_png(filename_wo_extension + '.png')
    graph.write_svg(filename_wo_extension + '.svg')


def generate_pydot_graph(root, visibility_level, profile=None):
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

    def add_edges(root, root_dot_name, visibility_level, profile):
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
                color = 'black'
                if (isinstance(c, GiskardBehavior) or (hasattr(c, 'original')
                                                       and isinstance(c.original, GiskardBehavior))) \
                        and not isinstance(c, PluginBehavior) and profile is not None:
                    if hasattr(c, 'original'):
                        file_name = str(c.original.__class__).split('.')[-2]
                    else:
                        file_name = str(c.__class__).split('.')[-2]
                    if file_name in profile:
                        max_time = max(profile[file_name].values(), key=lambda x: 0 if x == 'n/a' else x)
                        if max_time > 1:
                            color = 'red'
                        proposed_dot_name += '\n' + '\n'.join(
                            ['{}= {}'.format(k, v) for k, v in sorted(profile[file_name].items())])

                while proposed_dot_name in names:
                    proposed_dot_name = proposed_dot_name + "*"
                names.append(proposed_dot_name)
                node = pydot.Node(proposed_dot_name, shape=node_shape, style="filled", fillcolor=node_colour,
                                  fontsize=fontsize, fontcolor=node_font_colour, color=color)
                graph.add_node(node)
                edge = pydot.Edge(root_dot_name, proposed_dot_name)
                graph.add_edge(edge)
                if (isinstance(c, PluginBehavior) and c.get_plugins() != []) or \
                        (isinstance(c, Behaviour) and c.children != []):
                    add_edges(c, proposed_dot_name, visibility_level, profile)

    add_edges(root, root.name, visibility_level, profile)
    return graph


class OpenLoop(TreeManager):
    def grow_giskard(self):
        root = Sequence('Giskard')
        root.add_child(self.grow_wait_for_goal())
        root.add_child(CleanUp('cleanup'))
        root.add_child(self.grow_process_goal())
        root.add_child(self.grow_follow_joint_trajectory_execution())
        root.add_child(SendResult('send result', self.action_server_name, MoveAction))
        return root

    def grow_wait_for_goal(self):
        wait_for_goal = Sequence('wait for goal')
        wait_for_goal.add_child(self.grow_sync_branch())
        wait_for_goal.add_child(GoalReceived('has goal',
                                             self.action_server_name,
                                             MoveAction))
        return wait_for_goal

    def grow_sync_branch(self):
        sync = Sequence('Synchronize')
        sync.add_child(WorldUpdater('update world'))
        sync.add_child(running_is_success(SyncConfiguration)('update robot configuration',
                                                             self.god_map.unsafe_get_data(identifier.robot_group_name)))
        sync.add_child(SyncTfFrames('update robot localization',
                                    **self.god_map.unsafe_get_data(identifier.SyncTfFrames)))
        sync.add_child(TFPublisher('publish tf', **self.god_map.get_data(identifier.TFPublisher)))
        sync.add_child(CollisionSceneUpdater('update collision scene'))
        sync.add_child(running_is_success(VisualizationBehavior)('visualize collision scene'))
        return sync

    def grow_process_goal(self):
        process_move_cmd = success_is_failure(Sequence)('Process move commands')
        process_move_cmd.add_child(SetCmd('set move cmd', self.action_server_name))
        process_move_cmd.add_child(self.grow_planning())
        process_move_cmd.add_child(SetErrorCode('set error code', 'Planning'))
        process_move_goal = failure_is_success(Selector)('Process goal')
        process_move_goal.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                        self.action_server_name,
                                                                        MoveFeedback.PLANNING))
        process_move_goal.add_child(process_move_cmd)
        process_move_goal.add_child(ExceptionToExecute('clear exception'))
        process_move_goal.add_child(failure_is_running(CommandsRemaining)('commands remaining?'))
        return process_move_goal

    def grow_planning(self):
        planning = failure_is_success(Sequence)('planning')
        planning.add_child(IF('command set?', identifier.next_move_goal))
        planning.add_child(GoalToConstraints('update constraints', self.action_server_name))
        planning.add_child(self.grow_planning2())
        # planning.add_child(planning_1)
        # planning.add_child(SetErrorCode('set error code'))
        if self.god_map.get_data(identifier.PlotTrajectory_enabled):
            kwargs = self.god_map.get_data(identifier.PlotTrajectory)
            planning.add_child(PlotTrajectory('plot trajectory', **kwargs))
        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            kwargs = self.god_map.get_data(identifier.PlotDebugTrajectory)
            planning.add_child(PlotDebugExpressions('plot debug expressions', **kwargs))
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
        planning_3 = Sequence('planning III', sleep=0)
        planning_3.add_child(self.grow_planning4())
        planning_3.add_child(running_is_success(TimePlugin)('time for zero velocity'))
        planning_3.add_child(AppendZeroVelocity('append zero velocity'))
        planning_3.add_child(running_is_success(LogTrajPlugin)('log zero velocity'))
        if self.god_map.get_data(identifier.enable_VisualizationBehavior) \
                and not self.god_map.get_data(identifier.VisualizationBehavior_in_planning_loop):
            planning_3.add_child(running_is_success(VisualizationBehavior)('visualization', ensure_publish=True))
        if self.god_map.get_data(identifier.enable_CPIMarker) \
                and self.god_map.get_data(identifier.collision_checker) is not None \
                and not self.god_map.get_data(identifier.CPIMarker_in_planning_loop):
            planning_3.add_child(running_is_success(CollisionMarker)('collision marker'))
        return planning_3

    def grow_planning4(self):
        planning_4 = PluginBehavior('planning IIII')
        if self.god_map.get_data(identifier.collision_checker) is not None:
            planning_4.add_plugin(CollisionChecker('collision checker'))
        if self.god_map.get_data(identifier.VisualizationBehavior_in_planning_loop):
            planning_4.add_plugin(VisualizationBehavior('visualization'))
        if self.god_map.get_data(identifier.CPIMarker_in_planning_loop):
            planning_4.add_plugin(CollisionMarker('cpi marker'))
        planning_4.add_plugin(ControllerPlugin('controller'))
        planning_4.add_plugin(KinSimPlugin('kin sim'))
        planning_4.add_plugin(LogTrajPlugin('log'))
        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            planning_4.add_plugin(LogDebugExpressionsPlugin('log lba'))
        # planning_4.add_plugin(WiggleCancel('wiggle'))
        planning_4.add_plugin(LoopDetector('loop detector'))
        planning_4.add_plugin(GoalReachedPlugin('goal reached'))
        planning_4.add_plugin(TimePlugin('time'))
        if self.god_map.get_data(identifier.MaxTrajectoryLength_enabled):
            kwargs = self.god_map.get_data(identifier.MaxTrajectoryLength)
            planning_4.add_plugin(MaxTrajectoryLength('traj length check', **kwargs))
        return planning_4

    def grow_follow_joint_trajectory_execution(self):
        execution_action_server = Parallel('execution action servers',
                                           policy=ParallelPolicy.SuccessOnAll(synchronise=True))
        action_servers = self.god_map.get_data(identifier.robot_interface)
        behaviors = get_all_classes_in_package(giskardpy.tree.behaviors)
        for i, (execution_action_server_name, params) in enumerate(action_servers.items()):
            C = behaviors[params['plugin']]
            del params['plugin']
            execution_action_server.add_child(C(execution_action_server_name, **params))

        execute_canceled = Sequence('execute canceled')
        execute_canceled.add_child(GoalCanceled('goal canceled', self.action_server_name))
        execute_canceled.add_child(SetErrorCode('set error code', 'Execution'))

        publish_result = failure_is_success(Selector)('monitor execution')
        publish_result.add_child(success_is_failure(PublishFeedback)('publish feedback',
                                                                     self.god_map.get_data(
                                                                         identifier.action_server_name),
                                                                     MoveFeedback.EXECUTION))
        publish_result.add_child(execute_canceled)
        publish_result.add_child(execution_action_server)
        publish_result.add_child(SetErrorCode('set error code', 'Execution'))

        move_robot = failure_is_success(Sequence)('move robot')
        move_robot.add_child(IF('execute?', identifier.execute))
        move_robot.add_child(publish_result)
        return move_robot


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
        planning_3.add_child(self.grow_planning4())
        return planning_3

    def grow_planning4(self):
        planning_4 = PluginBehavior('planning IIII')
        action_servers = self.god_map.get_data(identifier.robot_interface)
        behaviors = get_all_classes_in_package(giskardpy.tree.behaviors)
        for i, (execution_action_server_name, params) in enumerate(action_servers.items()):
            C = behaviors[params['plugin']]
            del params['plugin']
            planning_4.add_plugin(C(execution_action_server_name, **params))
        planning_4.add_plugin(SyncConfiguration2('update robot configuration',
                                                 self.god_map.unsafe_get_data(identifier.robot_group_name)))
        planning_4.add_plugin(LogTrajPlugin('log'))
        if self.god_map.get_data(identifier.collision_checker) is not None:
            planning_4.add_plugin(CollisionChecker('collision checker'))
        planning_4.add_plugin(ControllerPlugin('controller'))
        planning_4.add_plugin(KinSimPlugin('kin sim'))

        if self.god_map.get_data(identifier.PlotDebugTrajectory_enabled):
            planning_4.add_plugin(LogDebugExpressionsPlugin('log lba'))
        # planning_4.add_plugin(WiggleCancel('wiggle'))
        # planning_4.add_plugin(LoopDetector('loop detector'))
        planning_4.add_plugin(GoalReachedPlugin('goal reached'))
        planning_4.add_plugin(TimePlugin('time'))
        if self.god_map.get_data(identifier.MaxTrajectoryLength_enabled):
            kwargs = self.god_map.get_data(identifier.MaxTrajectoryLength)
            planning_4.add_plugin(MaxTrajectoryLength('traj length check', **kwargs))
        return planning_4


# def sanity_check(god_map):
#     check_velocity_limits_reachable(god_map)


def sanity_check_derivatives(god_map):
    weights = god_map.get_data(identifier.joint_weights)
    limits = god_map.get_data(identifier.joint_limits)
    check_derivatives(weights, 'Weights')
    check_derivatives(limits, 'Limits')
    if len(weights) != len(limits):
        raise AttributeError('Weights and limits are not defined for the same number of derivatives')


def check_derivatives(entries, name):
    """
    :type entries: dict
    """
    allowed_derivates = list(order_map.values())[1:]
    for weight in entries:
        if weight not in allowed_derivates:
            raise AttributeError(
                '{} set for unknown derivative: {} not in {}'.format(name, weight, list(allowed_derivates)))
    weight_ids = [order_map.inverse[x] for x in entries]
    if max(weight_ids) != len(weight_ids):
        raise AttributeError(
            '{} for {} set, but some of the previous derivatives are missing'.format(name, order_map[max(weight_ids)]))

# def check_velocity_limits_reachable(god_map):
#     # TODO a more general version of this
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
