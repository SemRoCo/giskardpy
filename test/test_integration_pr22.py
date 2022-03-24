from __future__ import division

import itertools
import re
from copy import deepcopy

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from numpy import pi
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from giskardpy import identifier
from giskardpy.data_types import PrefixName
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.identifier import fk_pose
from giskardpy.python_interface import DEFAULT_WORLD_TIMEOUT
from giskardpy.utils import logging
from utils_for_tests import PR22, compare_poses, compare_points, compare_orientations, publish_marker_vector, \
    JointGoalChecker

# TODO roslaunch iai_pr2_sim ros_control_sim_with_base.launch
# TODO roslaunch iai_kitchen upload_kitchen_obj.launch

# scopes = ['module', 'class', 'function']
pocky_pose = {'r_elbow_flex_joint': -1.29610152504,
              'r_forearm_roll_joint': -0.0301682323805,
              'r_shoulder_lift_joint': 1.20324921318,
              'r_shoulder_pan_joint': -0.73456435706,
              'r_upper_arm_roll_joint': -0.70790051778,
              'r_wrist_flex_joint': -0.10001,
              'r_wrist_roll_joint': 0.258268529825,

              'l_elbow_flex_joint': -1.29610152504,
              'l_forearm_roll_joint': 0.0301682323805,
              'l_shoulder_lift_joint': 1.20324921318,
              'l_shoulder_pan_joint': 0.73456435706,
              'l_upper_arm_roll_joint': 0.70790051778,
              'l_wrist_flex_joint': -0.1001,
              'l_wrist_roll_joint': -0.258268529825,

              'torso_lift_joint': 0.2,
              'head_pan_joint': 0,
              'head_tilt_joint': 0,
              }

pick_up_pose = {
    'head_pan_joint': -2.46056758502e-16,
    'head_tilt_joint': -1.97371778181e-16,
    'l_elbow_flex_joint': -0.962150355946,
    'l_forearm_roll_joint': 1.44894622393,
    'l_shoulder_lift_joint': -0.273579583084,
    'l_shoulder_pan_joint': 0.0695426768038,
    'l_upper_arm_roll_joint': 1.3591238067,
    'l_wrist_flex_joint': -1.9004529902,
    'l_wrist_roll_joint': 2.23732576003,
    'r_elbow_flex_joint': -2.1207193579,
    'r_forearm_roll_joint': 1.76628402882,
    'r_shoulder_lift_joint': -0.256729037039,
    'r_shoulder_pan_joint': -1.71258744959,
    'r_upper_arm_roll_joint': -1.46335011257,
    'r_wrist_flex_joint': -0.100010762609,
    'r_wrist_roll_joint': 0.0509923457388,
    'torso_lift_joint': 0.261791330751,
}

folder_name = 'tmp_data/'


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR22()
    request.addfinalizer(c.tear_down)
    return c

@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: PR22
    """
    logging.loginfo(u'resetting giskard')
    for robot_name in giskard.robot_names:
        giskard.open_l_gripper(robot_name)
        giskard.open_r_gripper(robot_name)
    giskard.clear_world()
    for robot_name in giskard.robot_names:
        giskard.reset_base(robot_name)
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position = Point(0, 2, 0)
    p.pose.orientation = Quaternion(0, 0, 0, 1)
    giskard.move_base(p, giskard.robot_names[1])
    return giskard

@pytest.fixture()
def kitchen_setup(resetted_giskard):
    """
    :type resetted_giskard: PR22
    :return:
    """
    resetted_giskard.allow_all_collisions()
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.better_pose, prefix=robot_name)
    resetted_giskard.plan_and_execute()
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                              set_js_topic=u'/kitchen/cram_joint_states')
    js = {str(k): 0.0 for k in resetted_giskard.world.groups[object_name].movable_joints}
    resetted_giskard.set_kitchen_js(js)
    return resetted_giskard

@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type resetted_giskard: PR22
    """
    resetted_giskard.allow_all_collisions()
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.default_pose, prefix=robot_name)
    resetted_giskard.plan_and_execute()
    return resetted_giskard

@pytest.fixture()
def pocky_pose_setup(resetted_giskard):
    """
    :type resetted_giskard: PR22
    """
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def box_setup(pocky_pose_setup):
    """
    :type pocky_pose_setup: PR22
    :rtype: PR22
    """
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(name='box', size=[1, 1, 1], pose=p)
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(pocky_pose_setup):
    """
    :type pocky_pose_setup: PR22
    :rtype: PR22
    """
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.3
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(name='box', size=[1, 1, 1], pose=p)
    return pocky_pose_setup


class TestFk(object):
    def test_fk(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        for robot_name in zero_pose.robot_names:
            for root, tip in itertools.product(zero_pose.world.groups[robot_name].link_names, repeat=2):
                fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
                fk2 = tf.lookup_pose(str(root), str(tip))
                compare_poses(fk1.pose, fk2.pose)

    def test_fk_attached(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        pocky = 'box'
        for robot_name in zero_pose.robot_names:
            ps = PoseStamped()
            ps.header.frame_id = str(PrefixName(zero_pose.r_tips[robot_name], robot_name))
            ps.pose.position.x = 0.05
            ps.pose.orientation.x = 1.0
            zero_pose.attach_box(robot_name + pocky, [0.1, 0.02, 0.02],
                                 str(PrefixName(zero_pose.r_tips[robot_name], robot_name)),
                                 ps)
            for root, tip in itertools.product(zero_pose.world.groups[robot_name].link_names, [robot_name + pocky]):
                fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
                fk2 = tf.lookup_pose(str(root), str(tip))
                compare_poses(fk1.pose, fk2.pose)


class TestJointGoals(object):
    def test_joint_movement1a(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_all_collisions()
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(pocky_pose, prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_joint_movement1b(self, zero_pose):
        """
        Move one robot closer to the other one, such that they collide if both are going naively in the pocky pose.

        :type zero_pose: PR22
        """
        zero_pose.avoid_all_collisions()
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(0, 1, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.move_base(p, zero_pose.robot_names[1])
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(pocky_pose, prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision()
        js = dict(list(pocky_pose.items())[:3])
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(js, prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision()
        js = {'r_wrist_roll_joint': -pi,
              'l_wrist_roll_joint': -2.1 * pi, }
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(js, prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_continuous_joint2(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision()
        js = dict()
        js.update({'{}/r_wrist_roll_joint'.format(zero_pose.robot_names[i-1]): -pi * i
                   for i in range(1, len(zero_pose.robot_names)+1)})
        js.update({'{}/l_wrist_roll_joint'.format(zero_pose.robot_names[i-1]): -2.1 * pi * i
                   for i in range(1, len(zero_pose.robot_names)+1)})
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1_with_prefix(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision()
        js = {'torso_lift_joint': 0.1}
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(js, prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1_without_prefix(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision()
        js = {'{}/torso_lift_joint'.format(robot_name): 0.1 for robot_name in zero_pose.robot_names}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint2(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision()
        js = {'{}/torso_lift_joint'.format(zero_pose.robot_names[i-1]): 0.1 * i for i in range(1, len(zero_pose.robot_names)+1)}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_hard_joint_limits(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        for robot_name in zero_pose.robot_names:
            zero_pose.allow_self_collision()
            r_elbow_flex_joint_name = PrefixName('r_elbow_flex_joint', zero_pose.tf_prefix[robot_name])
            torso_lift_joint_name = PrefixName('torso_lift_joint', zero_pose.tf_prefix[robot_name])
            head_pan_joint_name = PrefixName('head_pan_joint', zero_pose.tf_prefix[robot_name])
            robot = zero_pose.world.groups[robot_name]

            r_elbow_flex_joint_limits = robot.get_joint_position_limits(r_elbow_flex_joint_name)
            torso_lift_joint_limits = robot.get_joint_position_limits(torso_lift_joint_name)
            head_pan_joint_limits = robot.get_joint_position_limits(head_pan_joint_name)

            goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[0] - 0.2,
                       'torso_lift_joint': torso_lift_joint_limits[0] - 0.2,
                       'head_pan_joint': head_pan_joint_limits[0] - 0.2}

            zero_pose.set_joint_goal(goal_js, prefix=robot_name, check=False)
            zero_pose.plan_and_execute()

            goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[1] + 0.2,
                       'torso_lift_joint': torso_lift_joint_limits[1] + 0.2,
                       'head_pan_joint': head_pan_joint_limits[1] + 0.2}

            zero_pose.set_joint_goal(goal_js, prefix=robot_name, check=False)
        zero_pose.plan_and_execute()


class TestConstraints(object):

    def test_CartesianPosition_robot_independent_links(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        expecteds = list()
        new_poses = list()
        tip = zero_pose.r_tips[zero_pose.robot_names[0]]
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = str(PrefixName(tip, zero_pose.robot_names[0]))
        p.pose.orientation.w = 1

        expecteds.append(tf.transform_pose('map', p))

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('CartesianPosition',
                                root_link=str(PrefixName(zero_pose.r_tips[zero_pose.robot_names[0]], zero_pose.robot_names[0])),
                                tip_link=str(PrefixName(zero_pose.r_tips[zero_pose.robot_names[1]], zero_pose.robot_names[1])),
                                goal=p)

        zero_pose.plan_and_execute()
        for robot_name in zero_pose.robot_names:
            tip = zero_pose.r_tips[robot_name]
            new_poses.append(tf.lookup_pose('map', str(PrefixName(tip, robot_name))))
        [compare_points(expected.pose.position, new_pose.pose.position) for (expected, new_pose) in zip(expecteds, new_poses)]

    def test_CartesianPose(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        expecteds = list()
        new_poses = list()
        for robot_name in zero_pose.robot_names:
            tip = zero_pose.r_tips[robot_name]
            p = PoseStamped()
            p.header.stamp = rospy.get_rostime()
            p.header.frame_id = str(PrefixName(tip, robot_name))
            p.pose.position = Point(-0.4, -0.2, -0.3)
            p.pose.orientation = Quaternion(0, 0, 1, 0)

            expecteds.append(tf.transform_pose('map', p))

            zero_pose.allow_all_collisions()
            zero_pose.set_json_goal('CartesianPose',
                                    root_link=zero_pose.default_roots[robot_name].short_name,
                                    tip_link=tip,
                                    goal=p,
                                    prefix=robot_name)
        zero_pose.plan_and_execute()
        for robot_name in zero_pose.robot_names:
            tip = zero_pose.r_tips[robot_name]
            new_poses.append(tf.lookup_pose('map', str(PrefixName(tip, robot_name))))
        [compare_points(expected.pose.position, new_pose.pose.position) for (expected, new_pose) in zip(expecteds, new_poses)]


class TestCartGoals(object):
    def test_move_base_with_prefix(self, zero_pose):
        """
        :type zero_pose: PR222
        """
        for robot_name in zero_pose.robot_names:
            map_T_odom = PoseStamped()
            map_T_odom.pose.position.x = 1
            map_T_odom.pose.position.y = 1
            map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
            zero_pose.set_localization(map_T_odom, robot_name)

            base_goal = PoseStamped()
            base_goal.header.frame_id = 'map'
            base_goal.pose.position.x = 1
            base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
            zero_pose.set_cart_goal(base_goal, 'base_footprint', prefix=robot_name)
        zero_pose.plan_and_execute()

    def test_move_base_without_prefix(self, zero_pose):
        """
        :type zero_pose: PR222
        """
        for robot_name in zero_pose.robot_names:
            map_T_odom = PoseStamped()
            map_T_odom.pose.position.x = 1
            map_T_odom.pose.position.y = 1
            map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
            zero_pose.set_localization(map_T_odom, robot_name)

        for robot_name in zero_pose.robot_names:
            base_goal = PoseStamped()
            base_goal.header.frame_id = 'map'
            base_goal.pose.position.x = 1
            base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
            zero_pose.set_cart_goal(base_goal, '{}/base_footprint'.format(robot_name))
        zero_pose.plan_and_execute()

    def test_move_base_with_offset(self, zero_pose):
        """
        :type zero_pose: PR222
        """
        for i in range(0, len(zero_pose.robot_names)):
            map_T_odom = PoseStamped()
            map_T_odom.header.frame_id = 'map'
            map_T_odom.pose.position.x = i + 1
            map_T_odom.pose.position.y = i + 1
            map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
            zero_pose.set_localization(map_T_odom, zero_pose.robot_names[i])
            zero_pose.wait_heartbeats()

        for i in range(0, len(zero_pose.robot_names)):
            base_goal = PoseStamped()
            base_goal.header.frame_id = 'map'
            base_goal.pose.position.x = i + 1
            base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
            zero_pose.set_cart_goal(base_goal, '{}/base_footprint'.format(zero_pose.robot_names[i]))
        zero_pose.plan_and_execute()

    def test_rotate_gripper(self, zero_pose):
        """
        :type zero_pose: PR222
        """
        for robot_name in zero_pose.robot_names:
            r_goal = PoseStamped()
            r_goal.header.frame_id = zero_pose.r_tips[robot_name]
            r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [1, 0, 0]))
            zero_pose.set_cart_goal(r_goal, zero_pose.r_tips[robot_name], prefix=robot_name)
            zero_pose.plan_and_execute()

# import pytest
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_movement1'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_attached_collision2'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_self_collision'])
# pytest.main(['-s', __file__ + '::TestWayPoints::test_waypoints2'])
