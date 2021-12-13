import pydot
import rospy
from py_trees import Behaviour, Chooser, common, Selector, Sequence
from py_trees_ros.trees import BehaviourTree
from sortedcontainers import SortedList

from giskardpy import identifier
from giskardpy.god_map import GodMap
from giskardpy.tree.async_composite import PluginBehavior
from giskardpy.tree.better_parallel import Parallel
from giskardpy.utils import logging
from giskardpy.utils.utils import create_path


class TreeManager(object):
    class ManagerNode(object):
        def __init__(self, node, parent, position):
            """
            :param node: the behavior that is represented by this ManagerNode
            :type node: py_trees.behaviour.Behaviour
            :param parent: the parent of the behavior that is represented by this ManagerNode
            :type parent: py_trees.behaviour.Behaviour
            :param position: the position of the node in the list of children of the parent
            :type position: int
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
            :type manager_node: TreeManager.ManagerNode
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

    def __init__(self, god_map: GodMap, tree=None):
        self.god_map = god_map
        self.action_server_name = self.god_map.get_data(identifier.action_server_name)
        self.god_map.set_data(identifier.tree_manager, self)
        if tree is None:
            self.tree = BehaviourTree(self.grow_giskard())
            self.setup()
        else:
            self.tree = tree
        self.tree_nodes = {}
        self.__init_map(self.tree.root, None, 0)
        self.render()

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
        manager_node = TreeManager.ManagerNode(node=node, parent=parent, position=idx)
        if parent is not None:
            parent.enabled_children.add(manager_node)
        if isinstance(node, PluginBehavior):
            children = node.get_plugins()
            for child_name in children:
                child_node = TreeManager.ManagerNode(node=children[child_name], parent=manager_node, position=0)
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
        tree_node = TreeManager.ManagerNode(node=node, parent=parent, position=position)
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
        path = self.god_map.get_data(identifier.data_folder) + 'tree'
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
