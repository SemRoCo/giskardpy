from py_trees import Sequence, Selector, BehaviourTree, Blackboard
from py_trees.meta import failure_is_success, success_is_failure
from py_trees_ros.trees import BehaviourTree
import rospy
from collections import namedtuple
from collections import defaultdict
from giskard_msgs.srv import DisableNode, DisableNodeResponse
from giskardpy.plugin import PluginBehavior
from giskardpy import logging
from sortedcontainers import SortedList

class TreeManager():

    class ManagerNode():
        def __init__(self, node, parent, position):
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
            if manager_node in self.enabled_children:
                self.disabled_children.add(manager_node)
                self.enabled_children.remove(manager_node)
                if isinstance(self.node, PluginBehavior):
                    self.node.remove_plugin(manager_node.node.name)
                else:
                    self.node.remove_child(manager_node.node)
                return True
            else:
                logging.loginfo("node is not in the list of enabled children")
            return False

        def enable_child(self, manager_node):
            if manager_node in self.disabled_children:
                self.disabled_children.remove(manager_node)
                self.enabled_children.add(manager_node)
                if isinstance(self.node, PluginBehavior):
                    return self.node.add_plugin(manager_node.node)
                else:
                    idx = self.enabled_children.index(manager_node)
                    self.node.insert_child(manager_node.node, idx)
                return True
            logging.loginfo("node is not in the list of disabled children")
            return False

        def add_child(self, manager_node):
            if isinstance(self.node, PluginBehavior):
                self.enabled_children.add(manager_node)
                return self.node.add_plugin(manager_node.node)
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
                    for c in list(self.disabled_children.islice(start=idx)):
                        c.position += 1
                    idx = self.enabled_children.bisect_left(manager_node)
                    for c in list(self.enabled_children.islice(start=idx)):
                        c.position += 1
                self.node.insert_child(manager_node.node, idx)
                self.enabled_children.add(manager_node)
                return True

        def remove_child(self, manager_node):
            if isinstance(self.node, PluginBehavior):
                if manager_node in self.enabled_children:
                    self.enabled_children.remove(manager_node)
                    self.node.remove_plugin(manager_node.node.name)
                    return True
                if manager_node in self.disabled_children:
                    self.disabled_children.remove(manager_node)
                    return True
                return False
            else:
                if manager_node in self.enabled_children:
                    self.enabled_children.remove(manager_node)
                    self.node.remove_child(manager_node.node)
                elif(manager_node in self.disabled_children):
                    self.disabled_children.remove(manager_node)
                else:
                    logging.loginfo("no such node")
                    return False
                idx = self.disabled_children.bisect_right(manager_node)
                for c in list(self.disabled_children.islice(start=idx)):
                    c.position -= 1
                idx = self.enabled_children.bisect_right(manager_node)
                for c in list(self.enabled_children.islice(start=idx)):
                    c.position -= 1
                return True

    def __init__(self, tree):
        self.tree = tree
        self.tree_nodes = {}
        self.init_map(tree.root, None, 0)

    def init_map(self, node, parent, idx):
        manager_node = TreeManager.ManagerNode(node=node, parent=parent, position=idx)
        if parent != None:
            parent.enabled_children.add(manager_node)
        if isinstance(node, PluginBehavior):
            children = node.get_plugins()
            for child_name in children.keys():
                child_node = TreeManager.ManagerNode(node=children[child_name], parent=manager_node, position=0)
                self.tree_nodes[child_name] = child_node
                manager_node.enabled_children.add(child_node)
        self.tree_nodes[node.name] = manager_node
        for idx, child in enumerate(node.children):
            self.init_map(child, manager_node, idx)


    def disable_node(self, node_name):
        if node_name in self.tree_nodes.keys():
            t = self.tree_nodes[node_name]
            if t.parent != None:
                return t.parent.disable_child(t)
            else:
                logging.logwarn('cannot disable root node')
                return False
        logging.loginfo("no node with that name")
        return False


    def enable_node(self, node_name):
        if node_name in self.tree_nodes.keys():
            t = self.tree_nodes[node_name]
            if t.parent != None:
                return t.parent.enable_child(t)
            else:
                logging.loginfo('root node')
                return False
        logging.loginfo('there is no node with that name')
        return False


    def insert_node(self, node, parent_name, position=-1): #todo: plugin behavior in pluginbehavior does not work
        if node.name in self.tree_nodes.keys():
            print("node with that name already exists")
            return False
        try:
            parent = self.tree_nodes[parent_name]
        except:
            logging.loginfo("parent does not exist")
            return False

        return parent.add_child(TreeManager.ManagerNode(node=node, parent=parent, position=position))

    def remove_node(self, node_name):
        if node_name in self.tree_nodes.keys():
            node = self.tree_nodes[node_name]
            parent = node.parent
            del self.tree_nodes[node_name]
            return parent.remove_child(node)
        else:
            logging.loginfo('there is no node with that name')
            return False

