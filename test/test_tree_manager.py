import pytest
from py_trees.composites import Sequence, Selector
from giskardpy.plugin import PluginBehavior
from py_trees.meta import failure_is_success, success_is_failure
from py_trees import display, Blackboard
from py_trees_ros.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from giskardpy.tree_manager import TreeManager
from giskardpy.god_map import GodMap


class TestBehavior(Behaviour):
    def __init__(self, name):
        super(TestBehavior, self).__init__(name)



@pytest.fixture()
def blackboard():
    god_map = GodMap()
    blackboard = Blackboard
    blackboard.god_map = god_map
    return blackboard

@pytest.fixture()
def tree(blackboard):
    # ----------------------------------------------
    wait_for_goal = Sequence(u'wait for goal')
    wait_for_goal.add_child(TestBehavior(u'tf'))
    wait_for_goal.add_child(TestBehavior(u'js1'))
    wait_for_goal.add_child(TestBehavior(u'pybullet updater'))
    wait_for_goal.add_child(TestBehavior(u'has goal'))
    wait_for_goal.add_child(TestBehavior(u'js2'))
    # ----------------------------------------------
    planning_3 = PluginBehavior(u'planning III', sleep=0)
    planning_3.add_plugin(TestBehavior(u'coll'))
    planning_3.add_plugin(TestBehavior(u'controller'))
    planning_3.add_plugin(TestBehavior(u'kin sim'))
    planning_3.add_plugin(TestBehavior(u'log'))
    planning_3.add_plugin(TestBehavior(u'goal reached'))
    planning_3.add_plugin(TestBehavior(u'wiggle'))
    planning_3.add_plugin(TestBehavior(u'time'))
    # ----------------------------------------------
    publish_result = failure_is_success(Selector)(u'monitor execution')
    publish_result.add_child(TestBehavior(u'goal canceled'))
    publish_result.add_child(TestBehavior(u'send traj'))
    # ----------------------------------------------
    # ----------------------------------------------
    planning_2 = failure_is_success(Selector)(u'planning II')
    planning_2.add_child(TestBehavior(u'goal canceled'))
    planning_2.add_child(success_is_failure(TestBehavior)(u'visualization'))
    planning_2.add_child(success_is_failure(TestBehavior)(u'cpi marker'))
    planning_2.add_child(planning_3)
    # ----------------------------------------------
    move_robot = failure_is_success(Sequence)(u'move robot')
    move_robot.add_child(TestBehavior(u'execute?'))
    move_robot.add_child(publish_result)
    # ----------------------------------------------
    # ----------------------------------------------
    planning_1 = success_is_failure(Sequence)(u'planning I')
    planning_1.add_child(TestBehavior(u'update constraints'))
    planning_1.add_child(planning_2)
    # ----------------------------------------------
    # ----------------------------------------------
    process_move_goal = failure_is_success(Selector)(u'process move goal')
    process_move_goal.add_child(planning_1)
    process_move_goal.add_child(TestBehavior(u'set move goal'))
    # ----------------------------------------------
    #
    post_processing = failure_is_success(Sequence)(u'post processing')
    post_processing.add_child(TestBehavior(u'wiggle_cancel_final_detection'))
    post_processing.add_child(TestBehavior(u'post_processing'))
    # ----------------------------------------------
    # ----------------------------------------------
    root = Sequence(u'root')
    root.add_child(wait_for_goal)
    root.add_child(TestBehavior(u'cleanup'))
    root.add_child(process_move_goal)
    root.add_child(TestBehavior(u'plot trajectory'))
    root.add_child(post_processing)
    root.add_child(move_robot)
    root.add_child(TestBehavior(u'send result'))

    tree = BehaviourTree(root)
    return tree


@pytest.fixture()
def tree_manager(tree):
    return TreeManager(tree)





class TestTreeManager():

    def test_disable_enable_vis(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert (tree_manager.disable_node('visualization'))
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert (tree_manager.enable_node('visualization'))
        assert display.ascii_tree(tree.root) == ascii_tree
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]

    def test_disable_node_twice(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert (tree_manager.disable_node('visualization'))
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        ascii_tree2 = display.ascii_tree(tree.root)
        assert not (tree_manager.disable_node('visualization'))
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert display.ascii_tree(tree.root) == ascii_tree2
        assert (tree_manager.enable_node('visualization'))
        assert display.ascii_tree(tree.root) == ascii_tree
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]

    def test_enable_node_twice(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert not (tree_manager.enable_node('visualization'))
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert ascii_tree == display.ascii_tree(tree.root)

    def test_enable_disable_multiple_nodes(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'cpi marker' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'cpi marker' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'wait for goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'wait for goal' in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        assert (tree_manager.disable_node('visualization'))
        assert (tree_manager.disable_node('cpi marker'))
        assert (tree_manager.disable_node('wait for goal'))
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'cpi marker' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'cpi marker' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'cpi marker' not in display.ascii_tree(tree.root)
        assert 'wait for goal' in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'wait for goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        assert 'wait for goal' not in display.ascii_tree(tree.root)
        assert (tree_manager.enable_node('cpi marker'))
        assert (tree_manager.enable_node('visualization'))
        assert (tree_manager.enable_node('wait for goal'))
        assert display.ascii_tree(tree.root) == ascii_tree
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'cpi marker' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'cpi marker' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'wait for goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'wait for goal' in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]

    def test_remove_enabled_node(self, tree_manager, tree):
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert (tree_manager.remove_node('visualization'))
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert not (tree_manager.enable_node('visualization'))
        assert display.ascii_tree(tree.root) == """root
[-] wait for goal
    --> tf
    --> js1
    --> pybullet updater
    --> has goal
    --> js2
--> cleanup
--> process move goal
    --> planning I
        --> update constraints
        --> planning II
            --> goal canceled
            --> cpi marker
            --> planning III
    --> set move goal
--> plot trajectory
--> post processing
    --> wiggle_cancel_final_detection
    --> post_processing
--> move robot
    --> execute?
    --> monitor execution
        --> goal canceled
        --> send traj
--> send result
"""
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]

    def test_remove_disabled_node(self, tree_manager, tree):
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert (tree_manager.disable_node('visualization'))
        assert (tree_manager.remove_node('visualization'))
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert not (tree_manager.enable_node('visualization'))
        assert display.ascii_tree(tree.root) == """root
[-] wait for goal
    --> tf
    --> js1
    --> pybullet updater
    --> has goal
    --> js2
--> cleanup
--> process move goal
    --> planning I
        --> update constraints
        --> planning II
            --> goal canceled
            --> cpi marker
            --> planning III
    --> set move goal
--> plot trajectory
--> post processing
    --> wiggle_cancel_final_detection
    --> post_processing
--> move robot
    --> execute?
    --> monitor execution
        --> goal canceled
        --> send traj
--> send result
"""
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]

    def test_add_node(self, tree_manager, tree):
        visualization = tree_manager.tree_nodes['visualization'].node
        assert (tree_manager.remove_node('visualization'))
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert not (tree_manager.enable_node('visualization'))
        assert tree_manager.insert_node(visualization, 'wait for goal', 3)
        assert 'visualization' in display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['wait for goal'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['wait for goal'].enabled_children]

    def test_remove_node_twice(self, tree_manager, tree):
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert (tree_manager.remove_node('visualization'))
        assert not (tree_manager.remove_node('visualization'))

    def add_node_twice(self):
        visualization = tree_manager.tree_nodes['visualization'].node
        assert not tree_manager.insert_node(visualization, 'wait for goal', 3)
        assert display.ascii_tree(tree.root) == """root
[-] wait for goal
    --> tf
    --> js1
    --> pybullet updater
    --> has goal
    --> js2
--> cleanup
--> process move goal
    --> planning I
        --> update constraints
        --> planning II
            --> goal canceled
            --> visualization
            --> cpi marker
            --> planning III
    --> set move goal
--> plot trajectory
--> post processing
    --> wiggle_cancel_final_detection
    --> post_processing
--> move robot
    --> execute?
    --> monitor execution
        --> goal canceled
        --> send traj
--> send result
"""


    def test_same_order_add_node(self, tree_manager, tree):
        visualization = tree_manager.tree_nodes['visualization'].node
        assert (tree_manager.remove_node('visualization'))
        assert (tree_manager.disable_node('cpi marker'))
        assert (tree_manager.disable_node('update constraints'))
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'cpi marker' not in display.ascii_tree(tree.root)
        assert 'update constraints' not in display.ascii_tree(tree.root)
        assert (tree_manager.insert_node(visualization, 'planning I', 1))
        assert (tree_manager.enable_node('cpi marker'))
        assert (tree_manager.enable_node('update constraints'))
        assert display.ascii_tree(tree.root) == """root
[-] wait for goal
    --> tf
    --> js1
    --> pybullet updater
    --> has goal
    --> js2
--> cleanup
--> process move goal
    --> planning I
        --> update constraints
        --> visualization
        --> planning II
            --> goal canceled
            --> cpi marker
            --> planning III
    --> set move goal
--> plot trajectory
--> post processing
    --> wiggle_cancel_final_detection
    --> post_processing
--> move robot
    --> execute?
    --> monitor execution
        --> goal canceled
        --> send traj
--> send result
"""


    def test_disable_plugin_behavior_node(self, tree_manager, tree):
        planningIII = tree_manager.tree_nodes['planning III'].node
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert (tree_manager.disable_node('coll'))
        assert 'coll' not in [x for x in planningIII.get_plugins()]

    def test_enable_plugin_behavior_node(self, tree_manager, tree):
        planningIII = tree_manager.tree_nodes['planning III'].node
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert (tree_manager.disable_node('coll'))
        assert 'coll' not in [x for x in planningIII.get_plugins()]
        assert (tree_manager.enable_node('coll'))
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]

    def test_disable_child_of_disabled_node(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'process move goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'process move goal' in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        assert (tree_manager.disable_node('process move goal'))
        assert 'process move goal' not in display.ascii_tree(tree.root)
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'process move goal' in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'process move goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        assert 'process move goal' not in display.ascii_tree(tree.root)
        assert (tree_manager.disable_node('visualization'))
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert tree_manager.enable_node('process move goal')
        assert 'process move goal' in display.ascii_tree(tree.root)
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert (tree_manager.enable_node('visualization'))
        assert ascii_tree == display.ascii_tree(tree.root)


    def test_remove_and_readd_plugin_behavior_node(self, tree_manager, tree):
        planningIII = tree_manager.tree_nodes['planning III'].node
        coll = tree_manager.tree_nodes['coll'].node
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert tree_manager.remove_node('coll')
        assert 'coll' not in [x for x in planningIII.get_plugins()]
        assert not (tree_manager.enable_node('coll'))
        assert 'coll' not in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert tree_manager.insert_node(coll, 'planning III')
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]


    def test_enable_child_of_disabled_node(self, tree_manager, tree):
        assert (tree_manager.disable_node('visualization'))
        assert (tree_manager.disable_node('planning II'))
        assert (tree_manager.enable_node('visualization'))
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'planning II' not in display.ascii_tree(tree.root)

    def test_everything(self, tree_manager, tree):
        update_constraints = tree_manager.tree_nodes['update constraints'].node
        cpi_marker = tree_manager.tree_nodes['cpi marker'].node
        visualization = tree_manager.tree_nodes['visualization'].node
        js1 = tree_manager.tree_nodes['js1'].node
        post_processing = tree_manager.tree_nodes['post processing'].node
        assert tree_manager.remove_node('visualization')
        assert tree_manager.remove_node('cpi marker')
        assert tree_manager.remove_node('update constraints')
        assert tree_manager.remove_node('post processing')
        assert tree_manager.remove_node('js1')
        assert tree_manager.disable_node('pybullet updater')
        assert tree_manager.remove_node('pybullet updater')
        assert tree_manager.insert_node(visualization, 'wait for goal', 1)
        assert tree_manager.insert_node(cpi_marker, 'wait for goal', 3)
        assert tree_manager.insert_node(js1, 'wait for goal')
        assert tree_manager.insert_node(update_constraints, 'wait for goal', 1)
        assert tree_manager.insert_node(post_processing, 'wait for goal', 4)
        assert display.ascii_tree(tree.root) == """root
[-] wait for goal
    --> tf
    --> update constraints
    --> visualization
    --> has goal
    --> post processing
        --> wiggle_cancel_final_detection
        --> post_processing
    --> cpi marker
    --> js2
    --> js1
--> cleanup
--> process move goal
    --> planning I
        --> planning II
            --> goal canceled
            --> planning III
    --> set move goal
--> plot trajectory
--> move robot
    --> execute?
    --> monitor execution
        --> goal canceled
        --> send traj
--> send result
"""