from __future__ import division

import itertools

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis, rotation_from_matrix, quaternion_matrix

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.srv import UpdateWorldResponse
from giskardpy.configs.pr2_twice import PR22
from giskardpy.data_types import PrefixName
from giskardpy.identifier import fk_pose
from giskardpy.utils import logging
from giskardpy.utils.math import compare_points
from giskardpy.utils.utils import position_dict_to_joint_states
from iai_naive_kinematics_sim.srv import SetJointState, UpdateTransform, SetJointStateRequest, UpdateTransformRequest
from utils_for_tests import compare_poses, GiskardTestWrapper

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


class PR22TestWrapper(GiskardTestWrapper):
    default_pose = {'r_elbow_flex_joint': -0.15,
                    'r_forearm_roll_joint': 0,
                    'r_shoulder_lift_joint': 0,
                    'r_shoulder_pan_joint': 0,
                    'r_upper_arm_roll_joint': 0,
                    'r_wrist_flex_joint': -0.10001,
                    'r_wrist_roll_joint': 0,
                    'l_elbow_flex_joint': -0.15,
                    'l_forearm_roll_joint': 0,
                    'l_shoulder_lift_joint': 0,
                    'l_shoulder_pan_joint': 0,
                    'l_upper_arm_roll_joint': 0,
                    'l_wrist_flex_joint': -0.10001,
                    'l_wrist_roll_joint': 0,
                    'torso_lift_joint': 0.2,
                    'head_pan_joint': 0,
                    'head_tilt_joint': 0}

    better_pose = {'r_shoulder_pan_joint': -1.7125,
                   'r_shoulder_lift_joint': -0.25672,
                   'r_upper_arm_roll_joint': -1.46335,
                   'r_elbow_flex_joint': -2.12,
                   'r_forearm_roll_joint': 1.76632,
                   'r_wrist_flex_joint': -0.10001,
                   'r_wrist_roll_joint': 0.05106,
                   'l_shoulder_pan_joint': 1.9652,
                   'l_shoulder_lift_joint': - 0.26499,
                   'l_upper_arm_roll_joint': 1.3837,
                   'l_elbow_flex_joint': -2.12,
                   'l_forearm_roll_joint': 16.99,
                   'l_wrist_flex_joint': - 0.10001,
                   'l_wrist_roll_joint': 0,
                   'torso_lift_joint': 0.2,
                   'head_pan_joint': 0,
                   'head_tilt_joint': 0,
                   }

    def __init__(self):
        self.r_tips = dict()
        self.l_tips = dict()
        self.r_grippers = dict()
        self.l_grippers = dict()
        self.set_localization_srvs = dict()
        self.set_bases = dict()
        self.default_roots = dict()
        self.tf_prefix = dict()
        self.robot_names = ['pr2_a', 'pr2_b']
        self.odom_roots = {}
        super().__init__(PR22)
        for robot_name in self.robot_names:
            self.odom_roots[robot_name] = 'odom_combined'
            self.r_tips[robot_name] = 'r_gripper_tool_frame'
            self.l_tips[robot_name] = 'l_gripper_tool_frame'
            self.tf_prefix[robot_name] = robot_name.replace('/', '')
            self.r_grippers[robot_name] = rospy.ServiceProxy(
                '/{}/r_gripper_simulator/set_joint_states'.format(robot_name), SetJointState)
            self.l_grippers[robot_name] = rospy.ServiceProxy(
                '/{}/l_gripper_simulator/set_joint_states'.format(robot_name), SetJointState)
            self.set_localization_srvs[robot_name] = rospy.ServiceProxy(
                '/{}/map_odom_transform_publisher/update_map_odom_transform'.format(robot_name),
                UpdateTransform)
            self.set_bases[robot_name] = rospy.ServiceProxy('/{}/base_simulator/set_joint_states'.format(robot_name),
                                                            SetJointState)
            self.default_roots[robot_name] = self.world.groups[robot_name].root_link_name

    def move_base(self, goal_pose, robot_name):
        self.teleport_base(goal_pose, robot_name)

    def open_r_gripper(self, robot_name):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'r_gripper_l_finger_joint', u'r_gripper_r_finger_joint', u'r_gripper_l_finger_tip_joint',
                          u'r_gripper_r_finger_tip_joint']
        sjs.state.position = [0.54, 0.54, 0.54, 0.54]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.r_grippers[robot_name].call(sjs)

    def close_r_gripper(self, robot_name):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'r_gripper_l_finger_joint', u'r_gripper_r_finger_joint', u'r_gripper_l_finger_tip_joint',
                          u'r_gripper_r_finger_tip_joint']
        sjs.state.position = [0, 0, 0, 0]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.r_grippers[robot_name].call(sjs)

    def open_l_gripper(self, robot_name):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'l_gripper_l_finger_joint', u'l_gripper_r_finger_joint', u'l_gripper_l_finger_tip_joint',
                          u'l_gripper_r_finger_tip_joint']
        sjs.state.position = [0.54, 0.54, 0.54, 0.54]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.l_grippers[robot_name].call(sjs)

    def close_l_gripper(self, robot_name):
        sjs = SetJointStateRequest()
        sjs.state.name = [u'l_gripper_l_finger_joint', u'l_gripper_r_finger_joint', u'l_gripper_l_finger_tip_joint',
                          u'l_gripper_r_finger_tip_joint']
        sjs.state.position = [0, 0, 0, 0]
        sjs.state.velocity = [0, 0, 0, 0]
        sjs.state.effort = [0, 0, 0, 0]
        self.l_grippers[robot_name].call(sjs)

    def clear_world(self):
        return_val = super(GiskardTestWrapper, self).clear_world()
        assert return_val.error_codes == UpdateWorldResponse.SUCCESS
        assert len(self.world.groups) == 2
        assert len(self.world.robot_names) == 2
        assert self.original_number_of_links == len(self.world.links)

    def teleport_base(self, goal_pose, robot_name):
        goal_pose = tf.transform_pose(str(self.default_roots[robot_name]), goal_pose)
        js = {'odom_x_joint': goal_pose.pose.position.x,
              'odom_y_joint': goal_pose.pose.position.y,
              'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                      goal_pose.pose.orientation.y,
                                                                      goal_pose.pose.orientation.z,
                                                                      goal_pose.pose.orientation.w]))[0]}
        goal = SetJointStateRequest()
        goal.state = position_dict_to_joint_states(js)
        self.set_bases[robot_name].call(goal)
        rospy.sleep(0.5)

    def set_localization(self, map_T_odom, robot_name):
        """
        :type map_T_odom: PoseStamped
        """
        req = UpdateTransformRequest()
        req.transform.translation = map_T_odom.pose.position
        req.transform.rotation = map_T_odom.pose.orientation
        assert self.set_localization_srvs[robot_name](req).success
        self.wait_heartbeats(10)
        p2 = self.world.compute_fk_pose(self.world.root_link_name, self.world.groups[robot_name].root_link_name)
        compare_poses(p2.pose, map_T_odom.pose)

    def reset_base(self, robot_name):
        p = PoseStamped()
        p.header.frame_id = self.world.root_link_name
        p.pose.orientation.w = 1
        self.set_localization(p, robot_name)
        self.wait_heartbeats()
        self.teleport_base(p, robot_name)


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR22TestWrapper()
    request.addfinalizer(c.tear_down)
    return c

@pytest.fixture()
def resetted_giskard(giskard) -> PR22TestWrapper:
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
def kitchen_setup(resetted_giskard) -> PR22TestWrapper:
    resetted_giskard.allow_all_collisions()
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.better_pose, group_name=robot_name)
    resetted_giskard.plan_and_execute()
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states',
                              set_js_topic=u'/kitchen/cram_joint_states')
    js = {str(k): 0.0 for k in resetted_giskard.world.groups[object_name].movable_joint_names}
    resetted_giskard.set_kitchen_js(js)
    return resetted_giskard

@pytest.fixture()
def zero_pose(resetted_giskard) -> PR22TestWrapper:
    resetted_giskard.allow_all_collisions()
    for robot_name in resetted_giskard.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.default_pose, group_name=robot_name)
    resetted_giskard.plan_and_execute()
    return resetted_giskard

@pytest.fixture()
def pocky_pose_setup(resetted_giskard) -> PR22TestWrapper:
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


class TestFk(object):
    def test_fk(self, zero_pose):
        for robot_name in zero_pose.robot_names:
            for root, tip in itertools.product(zero_pose.world.groups[robot_name].link_names_as_set, repeat=2):
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
            zero_pose.add_box(robot_name + pocky, (0.1, 0.02, 0.02), ps,
                              parent_link=zero_pose.r_tips[robot_name],
                              parent_link_group=robot_name)
            for root, tip in itertools.product(zero_pose.world.groups[robot_name].link_names_as_set, [robot_name + pocky]):
                fk1 = zero_pose.god_map.get_data(fk_pose + [(root, tip)])
                fk2 = tf.lookup_pose(str(root), str(tip))
                compare_poses(fk1.pose, fk2.pose)


class TestJointGoals(object):
    def test_joint_movement1a(self, zero_pose: PR22TestWrapper):
        zero_pose.allow_all_collisions()
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(pocky_pose, group_name=robot_name)
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
            zero_pose.set_joint_goal(pocky_pose, group_name=robot_name)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision(zero_pose.robot_names[0])
        zero_pose.allow_self_collision(zero_pose.robot_names[1])
        js = dict(list(pocky_pose.items())[:3])
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(js, group_name=robot_name)
        zero_pose.plan_and_execute()

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision(zero_pose.robot_names[0])
        zero_pose.allow_self_collision(zero_pose.robot_names[1])
        js = {'r_wrist_roll_joint': -pi,
              'l_wrist_roll_joint': -2.1 * pi, }
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(js, group_name=robot_name)
        zero_pose.plan_and_execute()

    def test_continuous_joint2(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision(zero_pose.robot_names[0])
        zero_pose.allow_self_collision(zero_pose.robot_names[1])
        js = dict()
        js.update({'{}/r_wrist_roll_joint'.format(zero_pose.robot_names[i-1]): -pi * i
                   for i in range(1, len(zero_pose.robot_names)+1)})
        js.update({'{}/l_wrist_roll_joint'.format(zero_pose.robot_names[i-1]): -2.1 * pi * i
                   for i in range(1, len(zero_pose.robot_names)+1)})
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1_with_group_name(self, zero_pose: PR22TestWrapper):
        zero_pose.allow_self_collision(zero_pose.robot_names[0])
        zero_pose.allow_self_collision(zero_pose.robot_names[1])
        js = {'torso_lift_joint': 0.1}
        for robot_name in zero_pose.robot_names:
            zero_pose.set_joint_goal(js, group_name=robot_name)
        zero_pose.plan_and_execute()

    def test_prismatic_joint1_without_group_name(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision(zero_pose.robot_names[0])
        zero_pose.allow_self_collision(zero_pose.robot_names[1])
        js = {'{}/torso_lift_joint'.format(robot_name): 0.1 for robot_name in zero_pose.robot_names}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_prismatic_joint2(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        zero_pose.allow_self_collision(zero_pose.robot_names[0])
        zero_pose.allow_self_collision(zero_pose.robot_names[1])
        js = {'{}/torso_lift_joint'.format(zero_pose.robot_names[i-1]): 0.1 * i for i in range(1, len(zero_pose.robot_names)+1)}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_hard_joint_limits(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        for robot_name in zero_pose.robot_names:
            zero_pose.allow_self_collision(zero_pose.robot_names[0])
            zero_pose.allow_self_collision(zero_pose.robot_names[1])
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

            zero_pose.set_joint_goal(goal_js, group_name=robot_name, check=False)
            zero_pose.plan_and_execute()

            goal_js = {'r_elbow_flex_joint': r_elbow_flex_joint_limits[1] + 0.2,
                       'torso_lift_joint': torso_lift_joint_limits[1] + 0.2,
                       'head_pan_joint': head_pan_joint_limits[1] + 0.2}

            zero_pose.set_joint_goal(goal_js, group_name=robot_name, check=False)
        zero_pose.plan_and_execute()


class TestConstraints(object):

    def test_CartesianPosition(self, zero_pose):
        """
        :type zero_pose: PR22
        """
        expecteds = list()
        new_poses = list()
        tip = zero_pose.r_tips[zero_pose.robot_names[1]]
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = str(PrefixName(tip, zero_pose.robot_names[0]))
        p.pose.orientation.w = 1

        expecteds.append(tf.transform_pose('map', p))

        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal('CartesianPosition',
                                root_link=tip,
                                root_group=zero_pose.robot_names[0],
                                tip_link=tip,
                                tip_group=zero_pose.robot_names[1],
                                goal_point=p)

        zero_pose.plan_and_execute()
        new_poses.append(tf.lookup_pose('map', str(PrefixName(tip, zero_pose.tf_prefix[zero_pose.robot_names[0]]))))
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
                                    goal_pose=p,
                                    root_group=robot_name,
                                    tip_group=robot_name)
        zero_pose.plan_and_execute()
        for robot_name in zero_pose.robot_names:
            tip = zero_pose.r_tips[robot_name]
            new_poses.append(tf.lookup_pose('map', str(PrefixName(tip, zero_pose.tf_prefix[robot_name]))))
        [compare_points(expected.pose.position, new_pose.pose.position) for (expected, new_pose) in zip(expecteds, new_poses)]


class TestCartGoals(object):
    def test_move_base(self, zero_pose):
        """
        :type zero_pose: PR222
        """
        zero_pose.allow_all_collisions()
        for robot_name in zero_pose.robot_names:
            map_T_odom = PoseStamped()
            map_T_odom.pose.position.x = 1
            map_T_odom.pose.position.y = 1
            map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
            zero_pose.set_localization(map_T_odom, robot_name)
            zero_pose.wait_heartbeats()

        for robot_name in zero_pose.robot_names:
            base_goal = PoseStamped()
            base_goal.header.frame_id = 'map'
            base_goal.pose.position.x = 1
            base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
            zero_pose.set_cart_goal(base_goal, 'base_footprint', tip_group=robot_name, root_group=robot_name)
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
            base_goal.pose.position.x = i + 2
            base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
            zero_pose.set_cart_goal(base_goal, 'base_footprint', tip_group=zero_pose.robot_names[i],
                                    root_group=zero_pose.robot_names[i])
        zero_pose.plan_and_execute()

    def test_rotate_gripper(self, zero_pose):
        """
        :type zero_pose: PR222
        """
        for robot_name in zero_pose.robot_names:
            r_goal = PoseStamped()
            r_goal.header.frame_id = str(PrefixName(zero_pose.r_tips[robot_name], robot_name))
            r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [1, 0, 0]))
            zero_pose.set_cart_goal(r_goal, zero_pose.r_tips[robot_name], tip_group=robot_name, root_group=robot_name)
            zero_pose.plan_and_execute()

# import pytest
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_movement1'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_attached_collision2'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_self_collision'])
# pytest.main(['-s', __file__ + '::TestWayPoints::test_waypoints2'])
