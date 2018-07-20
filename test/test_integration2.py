from multiprocessing import Queue
from threading import Thread
import numpy as np
import pytest
import rospy

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveActionResult, CollisionEntry, MoveActionGoal, MoveResult
from giskard_msgs.srv import UpdateWorldResponse

from giskardpy.python_interface import GiskardWrapper
from giskardpy.tfwrapper import transform_pose, lookup_transform
from giskardpy.utils import to_list
from ros_trajectory_controller_main import giskard_pm

# scopes = ['module', 'class', 'function']
default_joint_state = {u'r_elbow_flex_joint': -1.29610152504,
                       u'r_forearm_roll_joint': -0.0301682323805,
                       u'r_gripper_joint': 2.22044604925e-16,
                       u'r_shoulder_lift_joint': 1.20324921318,
                       u'r_shoulder_pan_joint': -0.73456435706,
                       u'r_upper_arm_roll_joint': -0.70790051778,
                       u'r_wrist_flex_joint': -0.10001,
                       u'r_wrist_roll_joint': 0.258268529825,

                       u'l_elbow_flex_joint': -1.29610152504,
                       u'l_forearm_roll_joint': 0.0301682323805,
                       u'l_gripper_joint': 2.22044604925e-16,
                       u'l_shoulder_lift_joint': 1.20324921318,
                       u'l_shoulder_pan_joint': 0.73456435706,
                       u'l_upper_arm_roll_joint': 0.70790051778,
                       u'l_wrist_flex_joint': -0.1001,
                       u'l_wrist_roll_joint': -0.258268529825,

                       u'torso_lift_joint': 0.2,
                       u'head_pan_joint': 0,
                       u'head_tilt_joint': 0}


class Context(object):
    def __init__(self):
        rospy.set_param(u'~interactive_marker_chains', [])
        rospy.set_param(u'~enable_gui', False)
        rospy.set_param(u'~map_frame', u'map')
        rospy.set_param(u'~joint_convergence_threshold', 0.002)
        rospy.set_param(u'~wiggle_precision_threshold', 7)
        rospy.set_param(u'~sample_period', 0.1)
        rospy.set_param(u'~default_joint_vel_limit', 0.5)
        rospy.set_param(u'~default_collision_avoidance_distance', 0.05)
        rospy.set_param(u'~fill_velocity_values', False)
        rospy.set_param(u'~nWSR', u'None')
        rospy.set_param(u'~root_link', u'base_footprint')
        rospy.set_param(u'~enable_collision_marker', False)
        rospy.set_param(u'~enable_self_collision', False)
        rospy.set_param(u'~path_to_data_folder', u'../data/pr2/')
        rospy.set_param(u'~collision_time_threshold', 15)
        rospy.set_param(u'~max_traj_length', 30)
        self.sub_result = rospy.Subscriber(u'/qp_controller/command/result', MoveActionResult, self.cb, queue_size=100)

        self.pm = giskard_pm()
        self.pm.start_plugins()
        self.wrapper = GiskardWrapper(ns='tests')
        self.results = Queue(100)
        self.robot = self.pm._plugins[u'fk'].get_robot()
        controlled_joints = self.pm._plugins[u'controlled joints'].controlled_joints
        self.joint_limits = {joint_name: self.robot.get_joint_lower_upper_limit(joint_name) for joint_name in
                             controlled_joints if self.robot.is_joint_controllable(joint_name)}

        self.default_root = u'base_link'
        self.r_tip = u'r_gripper_tool_frame'
        self.l_tip = u'l_gripper_tool_frame'
        self.map = u'map'
        self.simple_base_pose_pub = rospy.Publisher(u'/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.sleep(0.5)

    def cb(self, msg):
        self.results.put(msg.result)

    def loop_once(self):
        self.pm.update()

    def send_fake_goal(self):
        goal = MoveActionGoal()
        goal.goal = self.wrapper._get_goal()

        t1 = Thread(target=self.pm._plugins[u'action server']._as.action_server.internal_goal_callback, args=(goal,))
        t1.start()
        while self.results.empty():
            self.loop_once()
        t1.join()
        result = self.results.get()
        return result

    def move_pr2_base(self, goal_pose):
        self.simple_base_pose_pub.publish(goal_pose)

    def reset_pr2_base(self):
        p = PoseStamped()
        p.header.frame_id = self.map
        p.pose.orientation.w = 1
        self.move_pr2_base(p)

    def get_allow_l_gripper(self, body_b='box'):
        links = [u'l_gripper_l_finger_tip_link', u'l_gripper_r_finger_tip_link', u'l_gripper_l_finger_link',
                 u'l_gripper_r_finger_link', u'l_gripper_r_finger_link', u'l_gripper_palm_link']
        return [CollisionEntry(CollisionEntry.ALLOW_COLLISION, 0, link, body_b, '') for link in links]

    def add_box(self, name='box', position=(1.2, 0, 0.5)):
        r = self.wrapper.add_box(name=name, position=position)
        assert r.error_codes == UpdateWorldResponse.SUCCESS

    def tear_down(self):
        print('stopping plugins')
        self.pm.stop()

    def set_and_check_cart_goal(self, root, tip, goal_pose):
        goal_in_base = transform_pose('base_footprint', goal_pose)
        self.wrapper.set_cart_goal(root, tip, goal_pose)
        self.send_fake_goal()
        current_pose = lookup_transform('base_footprint', tip)
        np.testing.assert_array_almost_equal(to_list(goal_in_base.pose.position),
                                             to_list(current_pose.pose.position), decimal=4)

        np.testing.assert_array_almost_equal(to_list(goal_in_base.pose.orientation),
                                             to_list(current_pose.pose.orientation), decimal=2)


@pytest.fixture(scope='module')
def ros(request):
    print('init ros')
    rospy.init_node('tests')

    def kill_ros():
        print('shutdown ros')
        rospy.signal_shutdown('die')

    request.addfinalizer(kill_ros)


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = Context()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def reseted_giskard(giskard):
    """
    :type giskard: Context
    """
    print('resetting giskard')
    giskard.wrapper.clear_world()
    giskard.wrapper.set_joint_goal(default_joint_state)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
    giskard.wrapper.set_collision_entries([collision_entry])
    assert giskard.send_fake_goal().error_code == MoveResult.SUCCESS
    giskard.reset_pr2_base()
    return giskard



class TestPM(object):
    def test1(self, reseted_giskard):
        """
        :type reseted_giskard: Context
        """
        reseted_giskard.add_box()
        p = PoseStamped()
        p.header.frame_id = reseted_giskard.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = 'box'
        reseted_giskard.wrapper.set_collision_entries([collision_entry])

        reseted_giskard.set_and_check_cart_goal(reseted_giskard.default_root, reseted_giskard.r_tip, p)
        # result = reseted_giskard.send_fake_goal()
        # self.assertEqual(result.error_code, MoveResult.SUCCESS)
        print('test1')

    def test2(self, reseted_giskard):
        print('test2')
