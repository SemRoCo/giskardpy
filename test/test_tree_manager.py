import pytest
from py_trees.composites import Sequence, Selector
from giskardpy.plugins.plugin import PluginBehavior
from py_trees.meta import failure_is_success, success_is_failure
from py_trees import display, Blackboard
from py_trees_ros.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from giskardpy.tree_manager import TreeManager
from giskardpy.god_map import GodMap
from giskardpy.utils import logging
from giskardpy import identifier

import roslaunch
import rospy
from giskardpy.utils.tfwrapper import init as tf_init
from utils_for_tests import PR2
from giskardpy.model.pybullet_wrapper import stop_pybullet
from giskardpy.plugins.plugin import GiskardBehavior


default_pose = {u'r_elbow_flex_joint': -0.15,
                u'r_forearm_roll_joint': 0,
                u'r_shoulder_lift_joint': 0,
                u'r_shoulder_pan_joint': 0,
                u'r_upper_arm_roll_joint': 0,
                u'r_wrist_flex_joint': -0.10001,
                u'r_wrist_roll_joint': 0,

                u'l_elbow_flex_joint': -0.15,
                u'l_forearm_roll_joint': 0,
                u'l_shoulder_lift_joint': 0,
                u'l_shoulder_pan_joint': 0,
                u'l_upper_arm_roll_joint': 0,
                u'l_wrist_flex_joint': -0.10001,
                u'l_wrist_roll_joint': 0,

                u'torso_lift_joint': 0.2,
                u'head_pan_joint': 0,
                u'head_tilt_joint': 0,
                }

pocky_pose = {u'r_elbow_flex_joint': -1.29610152504,
              u'r_forearm_roll_joint': -0.0301682323805,
              u'r_shoulder_lift_joint': 1.20324921318,
              u'r_shoulder_pan_joint': -0.73456435706,
              u'r_upper_arm_roll_joint': -0.70790051778,
              u'r_wrist_flex_joint': -0.10001,
              u'r_wrist_roll_joint': 0.258268529825,

              u'l_elbow_flex_joint': -1.29610152504,
              u'l_forearm_roll_joint': 0.0301682323805,
              u'l_shoulder_lift_joint': 1.20324921318,
              u'l_shoulder_pan_joint': 0.73456435706,
              u'l_upper_arm_roll_joint': 0.70790051778,
              u'l_wrist_flex_joint': -0.1001,
              u'l_wrist_roll_joint': -0.258268529825,

              u'torso_lift_joint': 0.2,
              u'head_pan_joint': 0,
              u'head_tilt_joint': 0,
              }


@pytest.fixture(scope='module')
def ros():
    try:
        logging.loginfo(u'deleting tmp test folder')
    except Exception:
        pass
    logging.loginfo(u'init ros')
    rospy.init_node(u'tests')
    tf_init(60)
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    rospy.set_param('/joint_trajectory_splitter/state_topics',
                    ['/whole_body_controller/base/state',
                     '/whole_body_controller/body/state'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/whole_body_controller/base/follow_joint_trajectory',
                     '/whole_body_controller/body/follow_joint_trajectory'])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter')
    joint_trajectory_splitter = launch.launch(node)
    yield
    joint_trajectory_splitter.stop()
    rospy.delete_param('/joint_trajectory_splitter/state_topics')
    rospy.delete_param('/joint_trajectory_splitter/client_topics')
    logging.loginfo(u'shutdown ros')
    rospy.signal_shutdown(u'die')
    try:
        logging.loginfo(u'deleting tmp test folder')
    except Exception:
        pass


@pytest.fixture(scope='function')
def giskard(ros):
    c = PR2()
    yield c
    tree_manager = c.get_god_map().get_data(identifier.tree_manager)
    tree_manager.get_node('pybullet updater').srv_update_world.shutdown()
    tree_manager.get_node('pybullet updater').get_object_names.shutdown()
    tree_manager.get_node('pybullet updater').get_object_info.shutdown()
    tree_manager.get_node('pybullet updater').get_attached_objects.shutdown()
    tree_manager.get_node('pybullet updater').update_rviz_markers.shutdown()
    tree_manager.get_node('coll').srv_activate_rendering.shutdown()
    c.tree.blackboard_exchange.get_blackboard_variables_srv.shutdown()
    c.tree.blackboard_exchange.open_blackboard_watcher_srv.shutdown()
    c.tree.blackboard_exchange.close_blackboard_watcher_srv.shutdown()
    c.tear_down()
    c.reset_base()
    stop_pybullet()

@pytest.fixture()
def zero_pose(giskard):
    """
    :type giskard: PR2
    """
    giskard.allow_all_collisions()
    giskard.send_and_check_joint_goal(default_pose)
    return giskard


class TestBehavior(Behaviour):
    def __init__(self, name):
        super(TestBehavior, self).__init__(name)

class RemovesItselfBehavior(GiskardBehavior):
    def __init__(self, name):
        super(RemovesItselfBehavior, self).__init__(name)
        self.tree_manager = self.get_god_map().get_data(identifier.tree_manager)

    def update(self):
        self.tree_manager.remove_node(self.name)
        return Status.RUNNING

class GiskardTestBehavior(GiskardBehavior):
    def __init__(self, name, return_status = Status.SUCCESS):
        super(GiskardTestBehavior, self).__init__(name)
        self.tree_manager = self.get_god_map().get_data(identifier.tree_manager)
        self.executed = False
        self.return_status = return_status

    def update(self):
        self.executed = True
        return self.return_status

class ReplaceBehavior(GiskardBehavior):
    def __init__(self, name, new_behavior, old_behavior_name, parent_name, position, return_status = Status.SUCCESS):
        super(ReplaceBehavior, self).__init__(name)
        self.tree_manager = self.get_god_map().get_data(identifier.tree_manager)
        self.new_behavior = new_behavior
        self.parent_name = parent_name
        self.position = position
        self.old_behavior_name = old_behavior_name
        self.return_status = return_status
        self.replaced=False

    def update(self):
        if not self.replaced:
            self.tree_manager.remove_node(self.old_behavior_name)
            self.tree_manager.insert_node(self.new_behavior, self.parent_name, self.position)
            self.replaced=True
        return self.return_status




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
        tree_manager.disable_node('visualization')
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        tree_manager.enable_node('visualization')
        assert display.ascii_tree(tree.root) == ascii_tree
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]

    def test_disable_node_twice(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        tree_manager.disable_node('visualization')
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        ascii_tree2 = display.ascii_tree(tree.root)
        with pytest.raises(ValueError):
            tree_manager.disable_node('visualization')
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert display.ascii_tree(tree.root) == ascii_tree2
        tree_manager.enable_node('visualization')
        assert display.ascii_tree(tree.root) == ascii_tree
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]

    def test_enable_node_twice(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        with pytest.raises(ValueError):
            tree_manager.enable_node('visualization')
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
        tree_manager.disable_node('visualization')
        tree_manager.disable_node('cpi marker')
        tree_manager.disable_node('wait for goal')
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'cpi marker' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'cpi marker' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'cpi marker' not in display.ascii_tree(tree.root)
        assert 'wait for goal' in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'wait for goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        assert 'wait for goal' not in display.ascii_tree(tree.root)
        tree_manager.enable_node('cpi marker')
        tree_manager.enable_node('visualization')
        tree_manager.enable_node('wait for goal')
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
        tree_manager.remove_node('visualization')
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        with pytest.raises(KeyError):
            tree_manager.enable_node('visualization')
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
        tree_manager.disable_node('visualization')
        tree_manager.remove_node('visualization')
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        with pytest.raises(KeyError):
            tree_manager.enable_node('visualization')
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
        tree_manager.remove_node('visualization')
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        assert 'visualization' not in display.ascii_tree(tree.root)
        with pytest.raises(KeyError):
            tree_manager.enable_node('visualization')
        tree_manager.insert_node(visualization, 'wait for goal', 3)
        assert 'visualization'in tree_manager.tree_nodes.keys()
        assert 'visualization' in display.ascii_tree(tree.root)
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['wait for goal'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['wait for goal'].enabled_children]

    def test_remove_node_twice(self, tree_manager, tree):
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        tree_manager.remove_node('visualization')
        with pytest.raises(KeyError):
            tree_manager.remove_node('visualization')

    def test_add_node_twice(self, tree_manager, tree):
        visualization = tree_manager.tree_nodes['visualization'].node
        with pytest.raises(ValueError):
            tree_manager.insert_node(visualization, 'wait for goal', 3)
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
        tree_manager.remove_node('visualization')
        tree_manager.disable_node('cpi marker')
        tree_manager.disable_node('update constraints')
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'cpi marker' not in display.ascii_tree(tree.root)
        assert 'update constraints' not in display.ascii_tree(tree.root)
        tree_manager.insert_node(visualization, 'planning I', 1)
        tree_manager.enable_node('cpi marker')
        tree_manager.enable_node('update constraints')
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
        tree_manager.disable_node('coll')
        assert 'coll' not in [x for x in planningIII.get_plugins()]

    def test_enable_plugin_behavior_node(self, tree_manager, tree):
        planningIII = tree_manager.tree_nodes['planning III'].node
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        tree_manager.disable_node('coll')
        assert 'coll' not in [x for x in planningIII.get_plugins()]
        tree_manager.enable_node('coll')
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]

    def test_disable_child_of_disabled_node(self, tree_manager, tree):
        ascii_tree = display.ascii_tree(tree.root)
        assert 'process move goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'process move goal' in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        tree_manager.disable_node('process move goal')
        assert 'process move goal' not in display.ascii_tree(tree.root)
        assert 'visualization' not in display.ascii_tree(tree.root)
        assert 'process move goal' in [x.node.name for x in tree_manager.tree_nodes['root'].disabled_children]
        assert 'process move goal' not in [x.node.name for x in tree_manager.tree_nodes['root'].enabled_children]
        assert 'process move goal' not in display.ascii_tree(tree.root)
        tree_manager.disable_node('visualization')
        assert 'visualization' in [x.node.name for x in tree_manager.tree_nodes['planning II'].disabled_children]
        assert 'visualization' not in [x.node.name for x in tree_manager.tree_nodes['planning II'].enabled_children]
        tree_manager.enable_node('process move goal')
        assert 'process move goal' in display.ascii_tree(tree.root)
        assert 'visualization' not in display.ascii_tree(tree.root)
        tree_manager.enable_node('visualization')
        assert ascii_tree == display.ascii_tree(tree.root)


    def test_remove_and_readd_plugin_behavior_node(self, tree_manager, tree):
        planningIII = tree_manager.tree_nodes['planning III'].node
        coll = tree_manager.tree_nodes['coll'].node
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        tree_manager.remove_node('coll')
        assert 'coll' not in [x for x in planningIII.get_plugins()]
        with pytest.raises(KeyError):
            tree_manager.enable_node('coll')
        assert 'coll' not in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        tree_manager.insert_node(coll, 'planning III')
        assert 'coll' in [x for x in planningIII.get_plugins()]
        assert 'coll' not in [x.node.name for x in tree_manager.tree_nodes['planning III'].disabled_children]
        assert 'coll' in [x.node.name for x in tree_manager.tree_nodes['planning III'].enabled_children]


    def test_enable_child_of_disabled_node(self, tree_manager, tree):
        tree_manager.disable_node('visualization')
        tree_manager.disable_node('planning II')
        tree_manager.enable_node('visualization')
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
        tree_manager.remove_node('visualization')
        tree_manager.remove_node('cpi marker')
        tree_manager.remove_node('update constraints')
        tree_manager.remove_node('post processing')
        tree_manager.remove_node('js1')
        tree_manager.disable_node('pybullet updater')
        tree_manager.remove_node('pybullet updater')
        tree_manager.insert_node(visualization, 'wait for goal', 1)
        tree_manager.insert_node(cpi_marker, 'wait for goal', 3)
        tree_manager.insert_node(js1, 'wait for goal')
        tree_manager.insert_node(update_constraints, 'wait for goal', 1)
        tree_manager.insert_node(post_processing, 'wait for goal', 4)
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



class TestTreeManagerGiskardIntegration():
    def test_disable_enable_vis_before_execution(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        tree_manager.disable_node('visualization')
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)
        tree_manager.enable_node('visualization')
        zero_pose.send_and_check_joint_goal(default_pose)

    def test_disable_enable_move_robot_before_execution(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        tree_manager.disable_node('move robot')
        tree_manager.enable_node('move robot')
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_enable_disable_plugin_behavior_node_before_execution(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        tree_manager.disable_node('coll')
        tree_manager.enable_node('coll')
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_remove_plugin_behavior_node_during_execution(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        planningIII = tree_manager.tree_nodes['planning III'].node
        removes_itself = RemovesItselfBehavior('removesitself')
        tree_manager.insert_node(removes_itself, 'planning III')
        assert 'removesitself' in [x for x in planningIII.get_plugins()]
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)
        assert 'removesitself' not in [x for x in planningIII.get_plugins()]

    def test_replace_plugin_behavior_node_during_execution_same_thread(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        planningIII = tree_manager.tree_nodes['planning III'].node
        test_behavior = GiskardTestBehavior('test_behavior', Status.RUNNING)
        replaces_itself = ReplaceBehavior('replace', test_behavior, 'replace', 'planning III', 3, Status.RUNNING)
        tree_manager.insert_node(replaces_itself, 'planning III', 3)
        assert 'replace' in [x for x in planningIII.get_plugins()]
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)
        assert 'replace' not in [x for x in planningIII.get_plugins()]
        assert 'test_behavior' in [x for x in planningIII.get_plugins()]
        assert test_behavior.executed

    def test_replace_plugin_behavior_node_during_execution_different_thread(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        planningIII = tree_manager.tree_nodes['planning III'].node
        wait_for_goal = tree_manager.tree_nodes['wait for goal'].node
        test_behavior1 = GiskardTestBehavior('test_behavior1', Status.RUNNING)
        test_behavior2 = GiskardTestBehavior('test_behavior2', Status.RUNNING)
        replace_behavior = ReplaceBehavior('replace', test_behavior2, 'test_behavior1', 'planning III', 3, Status.SUCCESS)
        tree_manager.insert_node(test_behavior1, 'planning III', 3)
        tree_manager.insert_node(replace_behavior, 'wait for goal', 3)
        assert 'replace' in [x.name for x in wait_for_goal.children]
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)
        assert 'test_behavior1' not in [x for x in planningIII.get_plugins()]
        assert 'test_behavior2' in [x for x in planningIII.get_plugins()]
        assert test_behavior2.executed

    def test_replace_node_during_execution(self, zero_pose):
        tree_manager = zero_pose.get_god_map().get_data(identifier.tree_manager)
        planningII = tree_manager.tree_nodes['planning II'].node
        test_behavior = GiskardTestBehavior('test_behavior', Status.FAILURE)
        replaces_itself = ReplaceBehavior('replace', test_behavior, 'replace', 'planning II', 3, Status.FAILURE)
        tree_manager.insert_node(replaces_itself, 'planning II', 3)
        assert 'replace' in [x.name for x in planningII.children]
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)
        assert 'replace' not in [x.name for x in planningII.children]
        assert 'test_behavior' in [x.name for x in planningII.children]
        assert test_behavior.executed
