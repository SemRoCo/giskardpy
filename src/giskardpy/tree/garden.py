from __future__ import annotations

from enum import Enum
from typing import Type, TypeVar, Union, Dict, List, Optional, Any

from abc import ABC
from collections import defaultdict
from copy import copy

import numpy as np
import py_trees
import pydot
import rospy
from py_trees import Chooser, common
from py_trees import Selector, Sequence
from sortedcontainers import SortedList

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.branches.giskard_bt import GiskardBT
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.tree.composites.better_parallel import Parallel
from giskardpy.tree.control_modes import ControlModes
from giskardpy.utils import logging
from giskardpy.utils.utils import create_path


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
