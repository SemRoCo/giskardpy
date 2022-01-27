import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult, CollisionEntry
from giskardpy.goals.goal import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from utils_for_tests import Donbot, compare_poses

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
    c = Donbot()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def self_collision_pose(resetted_giskard):
    """
    :type resetted_giskard: Donbot
    :rtype: Donbot
    """
    resetted_giskard.set_joint_goal(self_collision_js)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def fake_table_setup(zero_pose):
    """
    :type zero_pose: Donbot
    :rtype: Donbot
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
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(floor_detection_js)
        zero_pose.plan_and_execute()

    def test_joint_movement_gaya(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js1 = {"ur5_shoulder_pan_joint": 1.475476861000061,
               "ur5_shoulder_lift_joint": -1.664506737385885,
               "ur5_elbow_joint": -2.0976365248309534,
               "ur5_wrist_1_joint": 0.6524184942245483,
               "ur5_wrist_2_joint": 1.7044463157653809,
               "ur5_wrist_3_joint": -1.5686963240252894}
        js2 = {
            "ur5_shoulder_pan_joint": 4.112661838531494,
            "ur5_shoulder_lift_joint": - 1.6648781935321253,
            "ur5_elbow_joint": - 1.4145501295672815,
            "ur5_wrist_1_joint": - 1.608563248311178,
            "ur5_wrist_2_joint": 1.5707963267948966,
            "ur5_wrist_3_joint": - 1.6503928343402308
        }
        zero_pose.set_joint_goal(js2)
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(js1)
        zero_pose.plan_and_execute()

    def test_empty_joint_goal(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal({
            'ur5_shoulder_pan_joint': -0.15841275850404912,
            'ur5_shoulder_lift_joint': -2.2956998983966272,
            'ur5_elbow_joint': 2.240689277648926,
            'ur5_wrist_1_joint': -2.608211342488424,
            'ur5_wrist_2_joint': -2.7356796900378626,
            'ur5_wrist_3_joint': -2.5249870459186,
        })
        zero_pose.set_joint_goal({})
        zero_pose.plan_and_execute()

    def test_joint_movement2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {
            'ur5_shoulder_pan_joint': -1.5438225905,
            'ur5_shoulder_lift_joint': -1.20804578463,
            'ur5_elbow_joint': -2.21223670641,
            'ur5_wrist_1_joint': -1.5827181975,
            'ur5_wrist_2_joint': -4.71748859087,
            'ur5_wrist_3_joint': -1.57543737093,
        }
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        js2 = {
            'ur5_shoulder_pan_joint': -np.pi / 2,
            'ur5_shoulder_lift_joint': -np.pi / 2,
            'ur5_elbow_joint': -2.3,
            'ur5_wrist_1_joint': -np.pi / 2,
            'ur5_wrist_2_joint': 0,
            'ur5_wrist_3_joint': -np.pi / 2,
        }
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(js2)
        zero_pose.plan_and_execute()

    def test_joint_movement3(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {
            'odom_x_joint': 1,
            'odom_y_joint': 1,
            'odom_z_joint': 1,
            'ur5_shoulder_pan_joint': -1.5438225905,
            'ur5_shoulder_lift_joint': -1.20804578463,
            'ur5_elbow_joint': -2.21223670641,
            'ur5_wrist_1_joint': -1.5827181975,
            'ur5_wrist_2_joint': -4.71748859087,
            'ur5_wrist_3_joint': -1.57543737093,
        }
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        js = dict(list(floor_detection_js.items())[:3])
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()


class TestConstraints(object):
    def test_pointing(self, better_pose):
        """
        :type better_pose: Donbot
        """
        tip = 'rs_camera_link'
        goal_point = tf.lookup_point('map', 'base_footprint')
        better_pose.set_pointing_goal(tip, goal_point)
        better_pose.plan_and_execute()

        goal_point = tf.lookup_point('map', tip)
        better_pose.set_pointing_goal(tip, goal_point, root_link=tip)


class TestCartGoals(object):
    def test_cart_goal_1eef(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.gripper_tip
        p.pose.position = Point(0, -0.1, 0)
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [1, 0, 0]))
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.gripper_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_cart_goal2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {
            'ur5_shoulder_pan_joint': 3.141554832458496,
            'ur5_shoulder_lift_joint': -1.3695076147662562,
            'ur5_elbow_joint': 0.5105495452880859,
            'ur5_wrist_1_joint': -0.7200177351581019,
            'ur5_wrist_2_joint': -0.22007495561708623,
            'ur5_wrist_3_joint': 0,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()
        p = PoseStamped()
        p.header.frame_id = 'camera_link'
        p.pose.position = Point(0, 1, 0)
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_translation_goal(p, zero_pose.camera_tip, 'ur5_shoulder_link', weight=WEIGHT_BELOW_CA)
        zero_pose.set_rotation_goal(p, zero_pose.camera_tip, 'ur5_shoulder_link', weight=WEIGHT_ABOVE_CA)
        zero_pose.plan_and_execute()

    def test_endless_wiggling1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        start_pose = {
            'ur5_elbow_joint': 2.14547738764,
            'ur5_shoulder_lift_joint': -1.177280122,
            'ur5_shoulder_pan_joint': -1.8550731481,
            'ur5_wrist_1_joint': -3.70994178242,
            'ur5_wrist_2_joint': -1.30010203311,
            'ur5_wrist_3_joint': 1.45079807832,
        }

        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(start_pose)
        zero_pose.plan_and_execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'base_link'
        goal_pose.pose.position.x = -0.512
        goal_pose.pose.position.y = -1.036126
        goal_pose.pose.position.z = 0.605
        goal_pose.pose.orientation.x = -0.007
        goal_pose.pose.orientation.y = -0.684
        goal_pose.pose.orientation.z = 0.729
        goal_pose.pose.orientation.w = 0

        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(goal_pose, zero_pose.camera_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_endless_wiggling2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'base_link'
        goal_pose.pose.position.x = 0.212
        goal_pose.pose.position.y = -0.314
        goal_pose.pose.position.z = 0.873
        goal_pose.pose.orientation.x = 0.004
        goal_pose.pose.orientation.y = 0.02
        goal_pose.pose.orientation.z = 0.435
        goal_pose.pose.orientation.w = .9

        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(goal_pose, zero_pose.gripper_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_elbow_singularity(self, better_pose):
        """
        :type better_pose: Donbot
        """
        hand_goal = PoseStamped()
        hand_goal.header.frame_id = better_pose.gripper_tip
        hand_goal.pose.position.z = 1
        hand_goal.pose.orientation.w = 1
        better_pose.set_cart_goal(hand_goal, better_pose.gripper_tip, 'base_footprint', check=False)
        better_pose.plan_and_execute()
        hand_goal = PoseStamped()
        hand_goal.header.frame_id = better_pose.gripper_tip
        hand_goal.pose.position.z = -0.2
        hand_goal.pose.orientation.w = 1
        better_pose.set_cart_goal(hand_goal, better_pose.gripper_tip, 'base_footprint')
        better_pose.plan_and_execute()
        pass

    def test_elbow_singularity2(self, zero_pose):
        """
        :type better_pose: Donbot
        """
        tip = 'ur5_wrist_1_link'
        hand_goal = PoseStamped()
        hand_goal.header.frame_id = tip
        hand_goal.pose.position.x = 0.5
        hand_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(hand_goal, tip, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        hand_goal = PoseStamped()
        hand_goal.header.frame_id = tip
        hand_goal.pose.position.x = -0.6
        hand_goal.pose.orientation.w = 1
        zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.set_cart_goal(hand_goal, tip, 'base_footprint', weight=WEIGHT_BELOW_CA/2)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_base_driving(self, zero_pose):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        zero_pose.teleport_base(p)
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position.y = 1
        p.pose.orientation.w = 1
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute()

    def test_shoulder_singularity(self, better_pose):
        """
        :type better_pose: Donbot
        """
        hand_goal = PoseStamped()
        hand_goal.header.frame_id = 'ur5_base_link'
        hand_goal.pose.position.x = 0.05
        hand_goal.pose.position.y = -0.2
        hand_goal.pose.position.z = 0.4
        hand_goal.pose.orientation = Quaternion(*quaternion_from_matrix(
            [
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        ))
        better_pose.allow_all_collisions()
        better_pose.set_cart_goal(hand_goal, 'ur5_wrist_2_link', 'base_footprint', weight=WEIGHT_BELOW_CA)
        better_pose.plan_and_execute()
        hand_goal.pose.position.y = 0.05
        # better_pose.allow_all_collisions()
        better_pose.set_cart_goal(hand_goal, 'ur5_wrist_2_link', 'base_footprint', weight=WEIGHT_BELOW_CA)
        better_pose.plan_and_execute()
        pass


class TestCollisionAvoidanceGoals(object):
    # kernprof -lv py.test -s test/test_integration_donbot.py::TestCollisionAvoidanceGoals::test_place_in_shelf

    def test_attach_box(self, better_pose):
        """
        :type zero_pose: PR2
        """
        pocky = 'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0.05
        p.pose.orientation.w = 1
        better_pose.attach_box(pocky, [0.1, 0.02, 0.02], better_pose.gripper_tip, p)

    def test_avoid_collision(self, better_pose):
        """
        :type zero_pose: Donbot
        """
        box = 'box'
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.y = -0.75
        p.pose.position.z = 0.5
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        better_pose.add_box(box, [1, 0.5, 2], p)
        better_pose.plan_and_execute()

    def test_avoid_collision2(self, better_pose):
        """
        :type box_setup: PR2
        """
        # FIXME check if out of collision at the
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.z = 0.08
        p.pose.orientation.w = 1
        better_pose.attach_box(name='box',
                               size=[0.05, 0.05, 0.2],
                               parent_link=better_pose.gripper_tip,
                               pose=p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.15
        p.pose.position.y = -0.04
        p.pose.orientation.w = 1
        better_pose.add_box('br', [0.2, 0.01, 0.1], p)

        better_pose.allow_self_collision()
        better_pose.plan_and_execute()
        better_pose.check_cpi_geq(['box'], 0.025)

    def test_avoid_collision3(self, better_pose):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.z = 0.08
        p.pose.orientation.w = 1
        better_pose.attach_box(name='box',
                               size=[0.05, 0.05, 0.2],
                               parent_link=better_pose.gripper_tip,
                               pose=p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.15
        p.pose.position.y = 0.04
        p.pose.orientation.w = 1
        better_pose.add_box('bl', [0.2, 0.01, 0.1], p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.15
        p.pose.position.y = -0.04
        p.pose.orientation.w = 1
        better_pose.add_box('br', [0.2, 0.01, 0.1], p)

        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position = Point(0, 0, -0.15)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        better_pose.set_cart_goal(p, better_pose.gripper_tip, better_pose.default_root)

        better_pose.plan_and_execute()
        # TODO check traj length?
        better_pose.check_cpi_geq(['box'], 0.045)

    def test_avoid_collision4(self, better_pose):
        """
        :type better_pose: Donbot
        """
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.z = 0.08
        p.pose.orientation.w = 1
        better_pose.attach_cylinder(name='cylinder',
                                    height=0.3,
                                    radius=0.025,
                                    parent_link=better_pose.gripper_tip,
                                    pose=p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.25
        p.pose.position.y = 0.04
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        better_pose.add_cylinder('fdown', height=0.2, radius=0.01, pose=p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.25
        p.pose.position.y = -0.07
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        better_pose.add_cylinder('fup', height=0.2, radius=0.01, pose=p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.15
        p.pose.position.y = 0.07
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        better_pose.add_cylinder('bdown', height=0.2, radius=0.01, pose=p)
        p = PoseStamped()
        p.header.frame_id = better_pose.gripper_tip
        p.pose.position.x = 0
        p.pose.position.z = 0.15
        p.pose.position.y = -0.04
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 1, 0]))
        better_pose.add_cylinder('bup', height=0.2, radius=0.01, pose=p)

        eef_goal = PoseStamped()
        eef_goal.header.frame_id = 'cylinder'
        eef_goal.pose.position.z -= 0.2
        eef_goal.pose.orientation.w = 1
        better_pose.set_cart_goal(eef_goal, 'cylinder', weight=WEIGHT_BELOW_CA)
        better_pose.plan_and_execute()

    def test_allow_self_collision2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        goal_js = {
            'ur5_shoulder_lift_joint': .5,
        }
        zero_pose.set_joint_goal(goal_js, check=False)
        zero_pose.plan_and_execute()

        arm_goal = PoseStamped()
        arm_goal.header.frame_id = zero_pose.gripper_tip
        arm_goal.pose.position.y = -.1
        arm_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(arm_goal, zero_pose.gripper_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_avoid_self_collision(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        goal_js = {
            'ur5_shoulder_lift_joint': .5,
        }
        # zero_pose.wrapper.set_self_collision_distance(0.025)
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(goal_js)
        zero_pose.plan_and_execute()

        arm_goal = PoseStamped()
        arm_goal.header.frame_id = zero_pose.gripper_tip
        arm_goal.pose.position.y = -.1
        arm_goal.pose.orientation.w = 1
        # zero_pose.wrapper.set_self_collision_distance(0.025)
        zero_pose.set_cart_goal(arm_goal, zero_pose.gripper_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_avoid_self_collision2(self, self_collision_pose):
        self_collision_pose.plan_and_execute()
        map_T_root = tf.lookup_pose('map', 'base_footprint')
        expected_pose = Pose()
        expected_pose.orientation.w = 1
        compare_poses(map_T_root.pose, expected_pose)

    def test_unknown_body_b(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = 'asdf'
        zero_pose.set_collision_entries([ce])
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.UNKNOWN_OBJECT])
        zero_pose.plan_and_execute()
