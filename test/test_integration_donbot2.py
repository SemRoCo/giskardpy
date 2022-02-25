import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult, CollisionEntry
from giskardpy.data_types import PrefixName
from giskardpy.goals.goal import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from utils_for_tests import compare_poses, Donbot2

# TODO roslaunch iai_donbot_sim ros_control_sim.launch


floor_detection_js = {
    'ur5_shoulder_pan_joint': -1.63407260576,
    'ur5_shoulder_lift_joint': -1.4751423041,
    'ur5_elbow_joint': 0.677300930023,
    'ur5_wrist_1_joint': -2.12363607088,
    'ur5_wrist_2_joint': -1.50967580477,
    'ur5_wrist_3_joint': 1.55717146397,
}

self_collision_js = {
    'ur5_shoulder_pan_joint': -1.57,
    'ur5_shoulder_lift_joint': -1.35,
    'ur5_elbow_joint': 2.4,
    'ur5_wrist_1_joint': 0.66,
    'ur5_wrist_2_joint': 1.57,
    'ur5_wrist_3_joint': 1.28191862405e-15,
}


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = Donbot2()
    rospy.sleep(1.0)
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: Donbot2
    """
    for robot_name in giskard.robot_names:
        giskard.open_gripper(robot_name)
    giskard.clear_world()
    for robot_name in giskard.robot_names:
        giskard.reset_base(robot_name)
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position = Point(0, 1, 0)
    p.pose.orientation = Quaternion(0, 0, 0, 1)
    giskard.move_base(giskard.robot_names[1], p)
    return giskard

@pytest.fixture()
def better_pose(resetted_giskard):
    """
    :type resetted_giskard: Donbot2
    :rtype: Donbot2
    """
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.better_pose, prefix=robot_name)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard

@pytest.fixture()
def self_collision_pose(resetted_giskard):
    """
    :type resetted_giskard: Donbot2
    :rtype: Donbot2
    """
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(self_collision_js, prefix=robot_name)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type resetted_giskard: Donbot2
    """
    resetted_giskard.allow_all_collisions()
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.default_pose, prefix=robot_name)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def fake_table_setup(zero_pose):
    """
    :type zero_pose: Donbot2
    :rtype: Donbot2
    """
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 0.9
    p.pose.position.y = 0
    p.pose.position.z = 0.2
    p.pose.orientation.w = 1
    zero_pose.add_box(name='box', size=[1, 1, 1], pose=p)
    return zero_pose


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: Donbot2
        """
        zero_pose.allow_self_collision()
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(floor_detection_js, prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_joint_movement1a(self, zero_pose):
        """
        :type zero_pose: Donbot2
        """
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(0, -1, 0)
        p.pose.orientation = Quaternion(0, 0, 1.0, 0)
        zero_pose.move_base(zero_pose.robot_names[1], p)
        zero_pose.avoid_all_collisions()
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(floor_detection_js, prefix=robot_name)
        zero_pose.plan_and_execute()

class TestConstraints(object):
    def test_pointing(self, better_pose):
        """
        :type better_pose: Donbot2
        """
        for robot_name in better_pose.robot_names:
            tip = str(PrefixName('rs_camera_link', robot_name))
            goal_point = tf.lookup_point('map', str(PrefixName('base_footprint', robot_name)))
            better_pose.set_pointing_goal(tip, goal_point, prefix=robot_name)
        better_pose.plan_and_execute()


class TestCartGoals(object):

    def test_cart_goal2a(self, zero_pose):
        """
        :type zero_pose: Donbot2
        """
        for robot_name in zero_pose.robot_names:
            js = {
                'ur5_shoulder_pan_joint': 3.141554832458496,
                'ur5_shoulder_lift_joint': -1.3695076147662562,
                'ur5_elbow_joint': 0.5105495452880859,
                'ur5_wrist_1_joint': -0.7200177351581019,
                'ur5_wrist_2_joint': -0.22007495561708623,
                'ur5_wrist_3_joint': 0,
            }
            zero_pose.set_joint_goal(js, prefix=robot_name, check=False)
        zero_pose.plan_and_execute()
        for robot_name in zero_pose.robot_names:
            p = PoseStamped()
            p.header.frame_id = zero_pose.camera_tip
            p.pose.position = Point(0, 1, 0)
            p.pose.orientation.w = 1
            zero_pose.set_translation_goal(p, zero_pose.camera_tip,
                                           root_link='ur5_shoulder_link', prefix=robot_name,
                                           weight=WEIGHT_BELOW_CA)
            zero_pose.set_rotation_goal(p, zero_pose.camera_tip,
                                        root_link='ur5_shoulder_link', prefix=robot_name,
                                        weight=WEIGHT_ABOVE_CA)
        zero_pose.plan_and_execute()

    def test_cart_goal2b(self, zero_pose):
        """
        :type zero_pose: Donbot2
        """
        for robot_name in zero_pose.robot_names:
            js = {
                'ur5_shoulder_pan_joint': 3.141554832458496,
                'ur5_shoulder_lift_joint': -1.3695076147662562,
                'ur5_elbow_joint': 0.5105495452880859,
                'ur5_wrist_1_joint': -0.7200177351581019,
                'ur5_wrist_2_joint': -0.22007495561708623,
                'ur5_wrist_3_joint': 0,
            }
            zero_pose.set_joint_goal(js, prefix=robot_name, check=False)
        zero_pose.plan_and_execute()
        for robot_name in zero_pose.robot_names:
            p = PoseStamped()
            p.header.frame_id = str(zero_pose.camera_tips[robot_name])
            p.pose.position = Point(0, 1, 0)
            p.pose.orientation.w = 1
            zero_pose.set_translation_goal(p, str(zero_pose.camera_tips[robot_name]),
                                           root_link=str(PrefixName('ur5_shoulder_link', robot_name)),
                                           weight=WEIGHT_BELOW_CA)
            zero_pose.set_rotation_goal(p, str(zero_pose.camera_tips[robot_name]),
                                        root_link=str(PrefixName('ur5_shoulder_link', robot_name)),
                                        weight=WEIGHT_ABOVE_CA)
        zero_pose.plan_and_execute()