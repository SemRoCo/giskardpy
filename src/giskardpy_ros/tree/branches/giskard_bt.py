from collections import defaultdict
from copy import deepcopy
from typing import Any, Type

import numpy as np
import pydot
import rospy
from py_trees_ros.trees import BehaviourTree
from py_trees import Chooser, common, Composite, Behaviour
from py_trees import Selector, Sequence
from giskard_msgs.msg import MoveAction
from giskardpy.exceptions import GiskardException
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.send_result import SendResult
from giskardpy.tree.branches.clean_up_control_loop import CleanupControlLoop
from giskardpy.tree.branches.control_loop import ControlLoop
from giskardpy.tree.branches.post_processing import PostProcessing
from giskardpy.tree.branches.prepare_control_loop import PrepareControlLoop
from giskardpy.tree.branches.send_trajectories import ExecuteTraj
from giskardpy.tree.branches.wait_for_goal import WaitForGoal
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.composites.better_parallel import Parallel
from giskardpy.tree.control_modes import ControlModes
from giskardpy.tree.decorators import failure_is_success
from giskardpy.utils import logging
from giskardpy.utils.decorators import toggle_on, toggle_off
from giskardpy.utils.utils import create_path


def behavior_is_instance_of(obj: Any, type_: Type) -> bool:
    return isinstance(obj, type_) or hasattr(obj, 'original') and isinstance(obj.original, type_)


class GiskardBT(BehaviourTree):
    tick_rate: float = 0.05
    control_mode: ControlModes
    wait_for_goal: WaitForGoal
    prepare_control_loop: PrepareControlLoop
    post_processing: PostProcessing
    cleanup_control_loop: CleanupControlLoop
    control_loop_branch: ControlLoop
    root: Sequence
    execute_traj: ExecuteTraj

    def __init__(self, control_mode: ControlModes):
        god_map.tree = self
        self.control_mode = control_mode
        if control_mode not in ControlModes:
            raise AttributeError(f'Control mode {control_mode} doesn\'t exist.')
        self.root = Sequence('Giskard')
        self.wait_for_goal = WaitForGoal()
        self.prepare_control_loop = failure_is_success(PrepareControlLoop)()
        if self.is_closed_loop():
            max_hz = god_map.behavior_tree_config.control_loop_max_hz
        else:
            max_hz = god_map.behavior_tree_config.simulation_max_hz
        self.control_loop_branch = failure_is_success(ControlLoop)(max_hz=max_hz)
        if self.is_closed_loop():
            self.control_loop_branch.add_closed_loop_behaviors()
        else:
            self.control_loop_branch.add_projection_behaviors()

        self.post_processing = failure_is_success(PostProcessing)()
        self.cleanup_control_loop = CleanupControlLoop()
        if self.is_open_loop():
            self.execute_traj = failure_is_success(ExecuteTraj)()

        self.root.add_child(self.wait_for_goal)
        self.root.add_child(self.prepare_control_loop)
        self.root.add_child(self.control_loop_branch)
        self.root.add_child(self.cleanup_control_loop)
        self.root.add_child(self.post_processing)
        self.root.add_child(SendResult(god_map.move_action_server))
        super().__init__(self.root)
        self.switch_to_execution()

    def is_closed_loop(self):
        return self.control_mode == self.control_mode.close_loop

    def is_standalone(self):
        return self.control_mode == self.control_mode.standalone

    def is_open_loop(self):
        return self.control_mode == self.control_mode.open_loop

    @toggle_on('visualization_mode')
    def turn_on_visualization(self):
        self.wait_for_goal.publish_state.add_visualization_marker_behavior()
        self.control_loop_branch.publish_state.add_visualization_marker_behavior()

    @toggle_off('visualization_mode')
    def turn_off_visualization(self):
        self.wait_for_goal.publish_state.remove_visualization_marker_behavior()
        self.control_loop_branch.publish_state.remove_visualization_marker_behavior()

    @toggle_on('projection_mode')
    def switch_to_projection(self):
        if self.is_open_loop():
            self.root.remove_child(self.execute_traj)
        elif self.is_closed_loop():
            self.control_loop_branch.switch_to_projection()
        self.cleanup_control_loop.add_reset_world_state()

    @toggle_off('projection_mode')
    def switch_to_execution(self):
        if self.is_open_loop():
            self.root.insert_child(self.execute_traj, -2)
        elif self.is_closed_loop():
            self.control_loop_branch.switch_to_closed_loop()
        self.cleanup_control_loop.remove_reset_world_state()

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

    def kill_all_services(self):
        self.blackboard_exchange.get_blackboard_variables_srv.shutdown()
        self.blackboard_exchange.open_blackboard_watcher_srv.shutdown()
        self.blackboard_exchange.close_blackboard_watcher_srv.shutdown()
        # for value in god_map.tree_nodes.values():
        #     node = value.node
        #     for attribute_name, attribute in vars(node).items():
        #         if isinstance(attribute, rospy.Service):
        #             attribute.shutdown(reason='life is pain')

    def render(self):
        path = god_map.giskard.tmp_folder + 'tree'
        create_path(path)
        render_dot_tree(self.root, name=path)


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
    logging.loginfo(f"Writing {filename_wo_extension}.dot/svg/png")
    # graph.write(filename_wo_extension + '.dot')
    graph.write_png(filename_wo_extension + '.png')
    # graph.write_svg(filename_wo_extension + '.svg')


time_function_names = ['__init__', 'setup', 'initialise', 'update']


def get_original_node(node: Behaviour) -> Behaviour:
    if hasattr(node, 'original'):
        return node.original
    return node


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
        if hasattr(node, 'original'):
            node = node.original

        if isinstance(node, Chooser):
            attributes = ('doubleoctagon', 'cyan', 'black')  # octagon
        elif isinstance(node, Selector):
            attributes = ('octagon', 'cyan', 'black')  # octagon
        elif isinstance(node, Sequence):
            attributes = ('box', 'orange', 'black')
        elif isinstance(node, Parallel):
            attributes = ('note', 'gold', 'black')
        elif isinstance(node, AsyncBehavior):
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
    add_children_stats_to_parent(root)

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
                original_c = get_original_node(c)

                # %% add run time stats to proposed dot name
                function_name_padding = 20
                entry_name_padding = 8
                number_padding = function_name_padding - entry_name_padding
                if hasattr(original_c, '__times'):
                    time_dict = original_c.__times
                else:
                    time_dict = {}
                for function_name in time_function_names:
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

                proposed_dot_name = f'"{proposed_dot_name}"'
                names.append(proposed_dot_name)
                original_c = pydot.Node(proposed_dot_name, shape=node_shape, style="filled", fillcolor=node_colour,
                                        fontsize=fontsize, fontcolor=node_font_colour, color=color, fontname=fontname)
                graph.add_node(original_c)
                edge = pydot.Edge(root_dot_name, proposed_dot_name)
                graph.add_edge(edge)
                if (hasattr(c, 'children') and c.children != []) or (hasattr(c, '_children') and c._children != []):
                    add_edges(c, proposed_dot_name, visibility_level)

    add_edges(root, root.name, visibility_level)
    return graph


def add_children_stats_to_parent(parent: Composite) -> None:
    if ((hasattr(parent, 'children') and parent.children != [])
            or (hasattr(parent, '_children') and parent._children != [])):
        children = parent.children
        names2 = [c.name for c in children]
        for name, child in zip(names2, children):
            original_child = get_original_node(child)
            if isinstance(original_child, Composite):
                add_children_stats_to_parent(original_child)

            if not hasattr(parent, '__times'):
                setattr(parent, '__times', defaultdict(list))

            if hasattr(original_child, '__times'):
                time_dict = original_child.__times
            else:
                time_dict = {}
            for function_name in time_function_names:
                if function_name in time_dict:
                    if function_name not in parent.__times:
                        parent.__times = deepcopy(time_dict)
                    else:
                        for i, (v1, v2) in enumerate(zip(parent.__times[function_name], time_dict[function_name])):
                            if i > len(parent.__times[function_name]):
                                parent.__times[function_name].append(v2)
                            else:
                                parent.__times[function_name][i] = v1 + v2
