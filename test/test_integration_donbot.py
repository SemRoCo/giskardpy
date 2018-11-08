import shutil
import rospkg
from multiprocessing import Queue
from threading import Thread
import numpy as np
import pytest
import rospy
from numpy import pi
from angles import normalize_angle, normalize_angle_positive, shortest_angular_distance
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveActionResult, CollisionEntry, MoveActionGoal, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from transforms3d.quaternions import axangle2quat

from giskardpy.symengine_wrappers import quaternion_from_axis_angle
from giskardpy.test_utils import Donbot, Donbot
from giskardpy.tfwrapper import transform_pose, lookup_transform, init as tf_init
from giskardpy.utils import msg_to_list

# TODO roslaunch iai_donbot_sim ros_control_sim.launch


default_pose = {
    u'ur5_elbow_joint': 0.0,
    u'ur5_shoulder_lift_joint': 0.0,
    u'ur5_shoulder_pan_joint': 0.0,
    u'ur5_wrist_1_joint': 0.0,
    u'ur5_wrist_2_joint': 0.0,
    u'ur5_wrist_3_joint': 0.0
}

floor_detection_pose = {
    u'ur5_shoulder_pan_joint': -1.63407260576,
    u'ur5_shoulder_lift_joint': -1.4751423041,
    u'ur5_elbow_joint': 0.677300930023,
    u'ur5_wrist_1_joint': -2.12363607088,
    u'ur5_wrist_2_joint': -1.50967580477,
    u'ur5_wrist_3_joint': 1.55717146397,
}


@pytest.fixture(scope=u'module')
def ros(request):
    print(u'init ros')
    # shutil.rmtree(u'../data/')
    rospy.init_node(u'tests')
    tf_init(60)

    def kill_ros():
        print(u'shutdown ros')
        rospy.signal_shutdown(u'die')

    request.addfinalizer(kill_ros)


@pytest.fixture(scope=u'module')
def giskard(request, ros):
    c = Donbot()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: Donbot
    """
    print(u'resetting giskard')
    giskard.clear_world()
    giskard.reset_base()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type giskard: Donbot
    """
    resetted_giskard.set_joint_goal(default_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_goal()
    return resetted_giskard


#
# @pytest.fixture()
# def pocky_pose_setup(resetted_giskard):
#     resetted_giskard.set_joint_goal(pocky_pose)
#     resetted_giskard.allow_all_collisions()
#     resetted_giskard.send_and_check_goal()
#     return resetted_giskard


@pytest.fixture()
def box_setup(pocky_pose_setup):
    """
    :type pocky_pose_setup: Donbot
    :rtype: Donbot
    """
    pocky_pose_setup.add_box(position=[1.2, 0, 0.5])
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(zero_pose):
    """
    :type zero_pose: Donbot
    :rtype: Donbot
    """
    zero_pose.add_box(position=[.9, 0, 0.2])
    return zero_pose


@pytest.fixture()
def kitchen_setup(zero_pose):
    object_name = u'kitchen'
    zero_pose.add_urdf(object_name,
                       rospy.get_param(u'kitchen_description'),
                       u'/kitchen/joint_states',
                       lookup_transform(u'map', u'iai_kitchen/world'))
    return zero_pose


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(floor_detection_pose)

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        js = dict(pocky_pose.items()[:3])
        zero_pose.send_and_check_joint_goal(js)

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        js = {u'r_wrist_roll_joint': -pi,
              u'l_wrist_roll_joint': 3.5 * pi, }
        zero_pose.send_and_check_joint_goal(js)

    def test_undefined_type(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.UNDEFINED
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE

    def test_empty_goal(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.PLAN_AND_EXECUTE
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE


class TestCartGoals(object):
    def test_cart_goal_1eef(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.gripper_tip
        p.pose.position = Point(0, -0.1, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.gripper_tip, p)

    def test_endless_wiggling1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        start_pose = {
            u'ur5_elbow_joint': 2.14547738764,
            u'ur5_shoulder_lift_joint': -1.177280122,
            u'ur5_shoulder_pan_joint': -1.8550731481,
            u'ur5_wrist_1_joint': -3.70994178242,
            u'ur5_wrist_2_joint': -1.30010203311,
            u'ur5_wrist_3_joint': 1.45079807832,
        }

        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(start_pose)

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = u'base_link'
        goal_pose.pose.position.x = -0.512
        goal_pose.pose.position.y = -1.036126
        goal_pose.pose.position.z = 0.605
        goal_pose.pose.orientation.x = -0.007
        goal_pose.pose.orientation.y = -0.684
        goal_pose.pose.orientation.z = 0.729
        goal_pose.pose.orientation.w = 0

        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.camera_tip, goal_pose)

#     def test_cart_goal_2eef(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         # FIXME? eef don't move at the same time
#         r_goal = PoseStamped()
#         r_goal.header.frame_id = zero_pose.r_tip
#         r_goal.header.stamp = rospy.get_rostime()
#         r_goal.pose.position = Point(-0.1, 0, 0)
#         r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
#         zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, r_goal)
#         l_goal = PoseStamped()
#         l_goal.header.frame_id = zero_pose.l_tip
#         l_goal.header.stamp = rospy.get_rostime()
#         l_goal.pose.position = Point(-0.05, 0, 0)
#         l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
#         zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, l_goal)
#         zero_pose.allow_self_collision()
#         zero_pose.send_and_check_goal()
#         zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
#         zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)
#
#     def test_weird_wiggling(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#
#         # FIXME get rid of wiggling
#         goal_js = {
#             u'l_upper_arm_roll_joint': 1.63487737202,
#             u'l_shoulder_pan_joint': 1.36222920328,
#             u'l_shoulder_lift_joint': 0.229120778526,
#             u'l_forearm_roll_joint': 13.7578920265,
#             u'l_elbow_flex_joint': -1.48141189643,
#             u'l_wrist_flex_joint': -1.22662876066,
#             u'l_wrist_roll_joint': -53.6150824007,
#         }
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(goal_js)
#
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = -0.1
#         p.pose.orientation.w = 1
#         zero_pose.allow_all_collisions()
#         zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
#
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.2
#         p.pose.orientation.w = 1
#         zero_pose.allow_all_collisions()
#         zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
#
#     def test_hot_init_failed(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         r_goal = PoseStamped()
#         r_goal.header.frame_id = zero_pose.r_tip
#         r_goal.header.stamp = rospy.get_rostime()
#         r_goal.pose.position = Point(-0.0, 0, 0)
#         r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
#         zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, r_goal)
#         l_goal = PoseStamped()
#         l_goal.header.frame_id = zero_pose.l_tip
#         l_goal.header.stamp = rospy.get_rostime()
#         l_goal.pose.position = Point(-0.0, 0, 0)
#         l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
#         zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, l_goal)
#         zero_pose.allow_self_collision()
#         zero_pose.send_and_check_goal()
#         zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
#         zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)
#
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(default_pose)
#
#         goal_js = {
#             u'r_upper_arm_roll_joint': -0.0812729778068,
#             u'r_shoulder_pan_joint': -1.20939684714,
#             u'r_shoulder_lift_joint': 0.135095147908,
#             u'r_forearm_roll_joint': -1.50201448056,
#             u'r_elbow_flex_joint': -0.404527363115,
#             u'r_wrist_flex_joint': -1.11738043795,
#             u'r_wrist_roll_joint': 8.0946050982,
#         }
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(goal_js)
#
#     def test_endless_wiggling(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         goal_js = {
#             u'r_upper_arm_roll_joint': -0.0812729778068,
#             u'r_shoulder_pan_joint': -1.20939684714,
#             u'r_shoulder_lift_joint': 0.135095147908,
#             u'r_forearm_roll_joint': -1.50201448056,
#             u'r_elbow_flex_joint': -0.404527363115,
#             u'r_wrist_flex_joint': -1.11738043795,
#             u'r_wrist_roll_joint': 8.0946050982,
#         }
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(goal_js)
#
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.r_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.5
#         p.pose.orientation.w = 1
#         # self.giskard.allow_all_collisions()
#         zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
#         zero_pose.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)
#
#     def test_root_link_not_equal_chain_root(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         p = PoseStamped()
#         p.header.stamp = rospy.get_rostime()
#         p.header.frame_id = u'base_footprint'
#         p.pose.position.x = 0.8
#         p.pose.position.y = -0.5
#         p.pose.position.z = 1
#         p.pose.orientation.w = 1
#         zero_pose.allow_self_collision()
#         zero_pose.set_cart_goal(u'torso_lift_link', zero_pose.r_tip, p)
#         zero_pose.send_and_check_goal()
#
#
#
# class TestCollisionAvoidanceGoals(object):
#     def test_add_box(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         object_name = u'muh'
#         zero_pose.add_box(object_name, position=[1.2, 0, 1.6])
#
#     def test_add_remove_sphere(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         object_name = u'muh'
#         zero_pose.add_sphere(object_name, position=[1.2, 0, 1.6])
#         zero_pose.remove_object(object_name)
#
#     def test_add_remove_cylinder(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         object_name = u'muh'
#         zero_pose.add_cylinder(object_name, position=[1.2, 0, 1.6])
#         zero_pose.remove_object(object_name)
#
#     def test_add_urdf_body(self, kitchen_setup):
#         """
#         :type kitchen_setup: Donbot
#         """
#         kitchen_setup.remove_object(u'kitchen')
#
#     def test_attach_box(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         pocky = u'http://muh#pocky'
#         zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])
#
#     def test_attach_existing_box(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         pocky = u'http://muh#pocky'
#         zero_pose.add_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])
#         zero_pose.attach_box(pocky, frame_id=zero_pose.r_tip)
#
#     def test_attach_to_nonexistant_robot_link(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         pocky = u'http://muh#pocky'
#         zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], u'', [0.05, 0, 0],
#                              expected_response=UpdateWorldResponse.CORRUPT_SHAPE_ERROR)
#
#     def test_add_remove_object(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         object_name = u'muh'
#         zero_pose.add_box(object_name, position=[1.2, 0, 1.6])
#         zero_pose.remove_object(object_name)
#
#     def test_invalid_update_world(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         req = UpdateWorldRequest(42, WorldBody(), True, PoseStamped())
#         assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION
#
#     def test_missing_body_error(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         zero_pose.remove_object(u'muh', expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)
#
#     def test_corrupt_shape_error(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         req = UpdateWorldRequest(UpdateWorldRequest.ADD, WorldBody(type=WorldBody.PRIMITIVE_BODY,
#                                                                    shape=SolidPrimitive(type=42)), True, PoseStamped())
#         assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR
#
#     def test_unsupported_options(self, kitchen_setup):
#         """
#         :type kitchen_setup: Donbot
#         """
#         wb = WorldBody()
#         pose = PoseStamped()
#         pose.header.stamp = rospy.Time.now()
#         pose.header.frame_id = str(u'map')
#         pose.pose.position = Point()
#         pose.pose.orientation = Quaternion(w=1)
#         wb.type = WorldBody.URDF_BODY
#
#         req = UpdateWorldRequest(UpdateWorldRequest.ADD, wb, True, pose)
#         assert kitchen_setup.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.UNSUPPORTED_OPTIONS
#
#     def test_link_b_set_but_body_b_not(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         ce = CollisionEntry()
#         ce.type = CollisionEntry.AVOID_COLLISION
#         ce.link_bs = [u'asdf']
#         box_setup.add_collision_entries([ce])
#         box_setup.send_and_check_goal(MoveResult.INSOLVABLE)
#
#     def test_unknown_robot_link(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         ce = CollisionEntry()
#         ce.type = CollisionEntry.AVOID_COLLISION
#         ce.robot_links = [u'asdf']
#         box_setup.add_collision_entries([ce])
#         box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)
#
#     def test_unknown_body_b(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         ce = CollisionEntry()
#         ce.type = CollisionEntry.AVOID_COLLISION
#         ce.body_b = u'asdf'
#         box_setup.add_collision_entries([ce])
#         box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)
#
#     def test_unknown_link_b(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         ce = CollisionEntry()
#         ce.type = CollisionEntry.AVOID_COLLISION
#         ce.body_b = u'box'
#         ce.link_bs = [u'asdf']
#         box_setup.add_collision_entries([ce])
#         box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)
#
#     def test_base_link_in_collision(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         zero_pose.allow_self_collision()
#         zero_pose.add_box(position=[0, 0, -0.2])
#         zero_pose.send_and_check_joint_goal(pocky_pose)
#
#     def test_unknown_object1(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.pose.position = Point(0.1, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#         box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.AVOID_COLLISION
#         collision_entry.min_dist = 0.05
#         collision_entry.body_b = u'muh'
#         box_setup.add_collision_entries([collision_entry])
#
#         box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)
#
#     def test_allow_self_collision(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.1)
#         zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.1)
#
#     def test_allow_self_collision2(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         goal_js = {
#             u'l_elbow_flex_joint': -1.43286344265,
#             u'l_forearm_roll_joint': 1.26465060073,
#             u'l_shoulder_lift_joint': 0.47990329056,
#             u'l_shoulder_pan_joint': 0.281272240139,
#             u'l_upper_arm_roll_joint': 0.528415402668,
#             u'l_wrist_flex_joint': -1.18811419869,
#             u'l_wrist_roll_joint': 2.26884630124,
#         }
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(goal_js)
#
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.2
#         p.pose.orientation.w = 1
#         zero_pose.allow_self_collision()
#         zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
#         zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
#         zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
#         zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)
#
#     def test_allow_self_collision3(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         goal_js = {
#             u'l_elbow_flex_joint': -1.43286344265,
#             u'l_forearm_roll_joint': 1.26465060073,
#             u'l_shoulder_lift_joint': 0.47990329056,
#             u'l_shoulder_pan_joint': 0.281272240139,
#             u'l_upper_arm_roll_joint': 0.528415402668,
#             u'l_wrist_flex_joint': -1.18811419869,
#             u'l_wrist_roll_joint': 2.26884630124,
#         }
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(goal_js)
#
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.18
#         p.pose.position.z = 0.02
#         p.pose.orientation.w = 1
#
#         ces = []
#         ces.append(CollisionEntry(type=CollisionEntry.ALLOW_COLLISION,
#                                   robot_links=zero_pose.get_l_gripper_links(),
#                                   body_b=u'pr2',
#                                   link_bs=zero_pose.get_r_forearm_links()))
#         zero_pose.add_collision_entries(ces)
#
#         zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
#         zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
#         zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
#         zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)
#
#     def test_avoid_self_collision(self, zero_pose):
#         """
#         :type zero_pose: Donbot
#         """
#         goal_js = {
#             u'l_elbow_flex_joint': -1.43286344265,
#             u'l_forearm_roll_joint': 1.26465060073,
#             u'l_shoulder_lift_joint': 0.47990329056,
#             u'l_shoulder_pan_joint': 0.281272240139,
#             u'l_upper_arm_roll_joint': 0.528415402668,
#             u'l_wrist_flex_joint': -1.18811419869,
#             u'l_wrist_roll_joint': 2.26884630124,
#         }
#         zero_pose.allow_all_collisions()
#         zero_pose.send_and_check_joint_goal(goal_js)
#
#         p = PoseStamped()
#         p.header.frame_id = zero_pose.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.2
#         p.pose.orientation.w = 1
#         zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
#         zero_pose.send_goal()
#         zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)
#
#     def test_avoid_collision(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         ce = CollisionEntry()
#         ce.type = CollisionEntry.AVOID_COLLISION
#         ce.body_b = u'box'
#         ce.min_dist = 0.05
#         box_setup.add_collision_entries([ce])
#         box_setup.send_and_check_goal(MoveResult.SUCCESS)
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
#         box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)
#
#     def test_allow_collision(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position = Point(0.15, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.ALLOW_COLLISION
#         collision_entry.body_b = u'box'
#         collision_entry.link_bs = [u'base']
#         box_setup.wrapper.set_collision_entries([collision_entry])
#
#         box_setup.allow_self_collision()
#         box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
#         box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.0)
#
#     def test_avoid_collision2(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.pose.position = Point(0.1, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#         box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#
#         # box_setup.wrapper.avoid_collision()
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.AVOID_COLLISION
#         collision_entry.min_dist = 0.05
#         collision_entry.body_b = u'box'
#         box_setup.add_collision_entries([collision_entry])
#
#         box_setup.send_and_check_goal()
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
#         box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)
#
#     def test_avoid_collision_with_far_object(self, pocky_pose_setup):
#         """
#         :type pocky_pose_setup: Donbot
#         """
#         pocky_pose_setup.add_box(position=[25, 25, 25])
#         p = PoseStamped()
#         p.header.frame_id = pocky_pose_setup.r_tip
#         p.pose.position = Point(0.1, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#         pocky_pose_setup.set_cart_goal(pocky_pose_setup.default_root, pocky_pose_setup.r_tip, p)
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.AVOID_COLLISION
#         collision_entry.min_dist = 0.05
#         collision_entry.body_b = u'box'
#         pocky_pose_setup.add_collision_entries([collision_entry])
#
#         pocky_pose_setup.send_and_check_goal()
#         pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_l_gripper_links(), 0.048)
#         pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_r_gripper_links(), 0.048)
#
#     def test_avoid_all_collision(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.pose.position = Point(0.1, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#         box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
#         collision_entry.min_dist = 0.05
#         box_setup.add_collision_entries([collision_entry])
#
#         box_setup.send_and_check_goal()
#
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
#         box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)
#
#     def test_get_out_of_collision(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.pose.position = Point(0.15, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#         box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
#         collision_entry.min_dist = 0.05
#         box_setup.add_collision_entries([collision_entry])
#
#         box_setup.send_and_check_goal()
#
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.pose.position = Point(0.1, 0, 0)
#         p.pose.orientation = Quaternion(0, 0, 0, 1)
#         box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#
#         collision_entry = CollisionEntry()
#         collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
#         collision_entry.min_dist = 0.05
#         box_setup.add_collision_entries([collision_entry])
#
#         box_setup.send_and_check_goal()
#
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
#         box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)
#
#     def test_allow_collision_gripper(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         ces = box_setup.get_allow_l_gripper(u'box')
#         box_setup.add_collision_entries(ces)
#         p = PoseStamped()
#         p.header.frame_id = box_setup.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.11
#         p.pose.orientation.w = 1
#         box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.l_tip, p)
#         # box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
#         box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)
#
#     def test_attached_collision1(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         attached_link_name = u'pocky'
#         box_setup.attach_box(attached_link_name, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0])
#         box_setup.attach_box(attached_link_name, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0],
#                              expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = -0.11
#         p.pose.orientation.w = 1
#         box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
#         box_setup.check_cpi_geq([attached_link_name], 0.048)
#         box_setup.remove_object(attached_link_name)
#
#     def test_attached_collision_avoidance(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         pocky = 'http://muh#pocky'
#         box_setup.attach_box(pocky, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0])
#
#         ces = []
#         ce = CollisionEntry()
#         ce.type = CollisionEntry.ALLOW_COLLISION
#         ce.robot_links = [pocky]
#         ce.body_b = 'box'
#         ces.append(ce)
#         box_setup.add_collision_entries(ces)
#
#         p = PoseStamped()
#         p.header.frame_id = box_setup.r_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.y = -0.11
#         p.pose.orientation.w = 1
#         box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
#
#
#     def test_avoid_collision_gripper(self, box_setup):
#         """
#         :type box_setup: Donbot
#         """
#         box_setup.allow_all_collisions()
#         ces = box_setup.get_l_gripper_collision_entries(u'box', 0.05, CollisionEntry.AVOID_COLLISION)
#         box_setup.add_collision_entries(ces)
#         p = PoseStamped()
#         p.header.frame_id = box_setup.l_tip
#         p.header.stamp = rospy.get_rostime()
#         p.pose.position.x = 0.
#         p.pose.orientation.w = 1
#         box_setup.set_cart_goal(box_setup.default_root, box_setup.l_tip, p)
#         box_setup.send_goal()
#         box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.049)
