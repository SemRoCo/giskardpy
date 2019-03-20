# import shutil
# import rospkg
# from multiprocessing import Queue
# from threading import Thread
import numpy as np
import shutil
from itertools import combinations

import pytest
import rospy
from numpy import pi
# from angles import normalize_angle, normalize_angle_positive, shortest_angular_distance
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import CollisionEntry, MoveActionGoal, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
# from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
# from transforms3d.quaternions import axangle2quat

# from giskardpy.python_interface import GiskardWrapper
# from giskardpy.symengine_wrappers import quaternion_from_axis_angle
from giskardpy.identifier import fk_identifier
from utils_for_tests import PR2, compare_poses
from giskardpy.tfwrapper import lookup_transform, init as tf_init, lookup_pose

# from giskardpy.utils import msg_to_list

# TODO roslaunch iai_pr2_sim ros_control_sim.launch
# TODO roslaunch iai_kitchen upload_kitchen_obj.launch

# scopes = ['module', 'class', 'function']
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
              u'head_tilt_joint': 0}

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
                u'head_tilt_joint': 0}

gaya_pose = {u'r_shoulder_pan_joint': -1.7125,
             u'r_shoulder_lift_joint': -0.25672,
             u'r_upper_arm_roll_joint': -1.46335,
             u'r_elbow_flex_joint': -2.12216,
             u'r_forearm_roll_joint': 1.76632,
             u'r_wrist_flex_joint': -0.10001,
             u'r_wrist_roll_joint': 0.05106,
             u'l_shoulder_pan_joint': 1.9652,
             u'l_shoulder_lift_joint': - 0.26499,
             u'l_upper_arm_roll_joint': 1.3837,
             u'l_elbow_flex_joint': - 2.1224,
             u'l_forearm_roll_joint': 16.99,
             u'l_wrist_flex_joint': - 0.10001,
             u'l_wrist_roll_joint': 0}

pick_up_pose = {
    u'head_pan_joint': -2.46056758502e-16,
    u'head_tilt_joint': -1.97371778181e-16,
    u'l_elbow_flex_joint': -0.962150355946,
    u'l_forearm_roll_joint': 1.44894622393,
    u'l_shoulder_lift_joint': -0.273579583084,
    u'l_shoulder_pan_joint': 0.0695426768038,
    u'l_upper_arm_roll_joint': 1.3591238067,
    u'l_wrist_flex_joint': -1.9004529902,
    u'l_wrist_roll_joint': 2.23732576003,
    u'r_elbow_flex_joint': -2.1207193579,
    u'r_forearm_roll_joint': 1.76628402882,
    u'r_shoulder_lift_joint': -0.256729037039,
    u'r_shoulder_pan_joint': -1.71258744959,
    u'r_upper_arm_roll_joint': -1.46335011257,
    u'r_wrist_flex_joint': -0.100010762609,
    u'r_wrist_roll_joint': 0.0509923457388,
    u'torso_lift_joint': 0.321791330751,
}

folder_name = u'tmp_data/'


@pytest.fixture(scope=u'module')
def ros(request):
    try:
        print(u'deleting tmp test folder')
        # shutil.rmtree(folder_name)
    except Exception:
        pass

    print(u'init ros')
    rospy.init_node(u'tests')
    tf_init(60)

    def kill_ros():
        print(u'shutdown ros')
        rospy.signal_shutdown(u'die')
        try:
            print(u'deleting tmp test folder')
            # shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_ros)


@pytest.fixture(scope=u'module')
def giskard(request, ros):
    c = PR2()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: PR2
    """
    print(u'resetting giskard')
    giskard.clear_world()
    giskard.reset_base()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type giskard: PR2
    """
    resetted_giskard.set_joint_goal(default_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_goal()
    return resetted_giskard


@pytest.fixture()
def pocky_pose_setup(resetted_giskard):
    resetted_giskard.set_joint_goal(pocky_pose)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_goal()
    return resetted_giskard


@pytest.fixture()
def box_setup(pocky_pose_setup):
    """
    :type pocky_pose_setup: PR2
    :rtype: PR2
    """
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(pose=p)
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(zero_pose):
    """
    :type zero_pose: PR2
    :rtype: PR2
    """
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 0.9
    p.pose.position.y = 0
    p.pose.position.z = 0.2
    p.pose.orientation.w = 1
    zero_pose.add_box(pose=p)
    return zero_pose


@pytest.fixture()
def kitchen_setup(resetted_giskard):
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_joint_goal(gaya_pose)
    object_name = u'kitchen'
    resetted_giskard.add_urdf(object_name,
                              rospy.get_param(u'kitchen_description'),
                              u'/kitchen/joint_states',
                              lookup_pose(u'map', u'iai_kitchen/world'))
    return resetted_giskard

class TestFk(object):
    def test_fk1(self, zero_pose):
        root = zero_pose.get_robot().get_root()
        for link in zero_pose.get_robot().get_link_names():
            fk1 = zero_pose.get_god_map().safe_get_data(fk_identifier+[(root, link)])
            fk2 = lookup_pose(root, link)
            np.testing.assert_almost_equal(fk1.pose.position.x, fk2.pose.position.x)
            np.testing.assert_almost_equal(fk1.pose.position.y, fk2.pose.position.y)
            np.testing.assert_almost_equal(fk1.pose.position.z, fk2.pose.position.z)
            np.testing.assert_almost_equal(fk1.pose.orientation.x, fk2.pose.orientation.x)
            np.testing.assert_almost_equal(fk1.pose.orientation.y, fk2.pose.orientation.y)
            np.testing.assert_almost_equal(fk1.pose.orientation.z, fk2.pose.orientation.z)
            np.testing.assert_almost_equal(fk1.pose.orientation.w, fk2.pose.orientation.w)

class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = dict(pocky_pose.items()[:3])
        zero_pose.send_and_check_joint_goal(js)

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        js = {u'r_wrist_roll_joint': -pi,
              u'l_wrist_roll_joint': 3.5 * pi, }
        zero_pose.send_and_check_joint_goal(js)

    def test_undefined_type(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.UNDEFINED
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE

    def test_empty_goal(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.PLAN_AND_EXECUTE
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE


class TestCartGoals(object):
    def test_cart_goal_1eef(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)

    def test_cart_goal_1eef2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.default_root
        p.pose.position = Point(0.599, -0.009, 0.983)
        p.pose.orientation = Quaternion(0.524, -0.495, 0.487, -0.494)
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)

    def test_cart_goal_1eef3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        self.test_cart_goal_1eef(zero_pose)
        self.test_cart_goal_1eef2(zero_pose)

    def test_cart_goal_2eef(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.header.stamp = rospy.get_rostime()
        r_goal.pose.position = Point(-0.1, 0, 0)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, r_goal)
        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.l_tip
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.pose.position = Point(-0.05, 0, 0)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, l_goal)
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_weird_wiggling(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_upper_arm_roll_joint': 1.63487737202,
            u'l_shoulder_pan_joint': 1.36222920328,
            u'l_shoulder_lift_joint': 0.229120778526,
            u'l_forearm_roll_joint': 13.7578920265,
            u'l_elbow_flex_joint': -1.48141189643,
            u'l_wrist_flex_joint': -1.22662876066,
            u'l_wrist_roll_joint': -53.6150824007,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1
        # zero_pose.allow_all_collisions()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        # zero_pose.allow_all_collisions()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)

    def test_hot_init_failed(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.r_tip
        r_goal.header.stamp = rospy.get_rostime()
        r_goal.pose.position = Point(-0.0, 0, 0)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, r_goal)
        l_goal = PoseStamped()
        l_goal.header.frame_id = zero_pose.l_tip
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.pose.position = Point(-0.0, 0, 0)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, l_goal)
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(default_pose)

        goal_js = {
            u'r_upper_arm_roll_joint': -0.0812729778068,
            u'r_shoulder_pan_joint': -1.20939684714,
            u'r_shoulder_lift_joint': 0.135095147908,
            u'r_forearm_roll_joint': -1.50201448056,
            u'r_elbow_flex_joint': -0.404527363115,
            u'r_wrist_flex_joint': -1.11738043795,
            u'r_wrist_roll_joint': 8.0946050982,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

    # def test_endless_wiggling(self, zero_pose):
    #     """
    #     :type zero_pose: PR2
    #     """
    #     #FIXME
    #     goal_js = {
    #         u'r_upper_arm_roll_joint': -0.0812729778068,
    #         u'r_shoulder_pan_joint': -1.20939684714,
    #         u'r_shoulder_lift_joint': 0.135095147908,
    #         u'r_forearm_roll_joint': -1.50201448056,
    #         u'r_elbow_flex_joint': -0.404527363115,
    #         u'r_wrist_flex_joint': -1.11738043795,
    #         u'r_wrist_roll_joint': 8.0946050982,
    #     }
    #     zero_pose.allow_all_collisions()
    #     zero_pose.send_and_check_joint_goal(goal_js)
    #
    #     p = PoseStamped()
    #     p.header.frame_id = zero_pose.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position.x = 0.5
    #     p.pose.orientation.w = 1
    #     # self.giskard.allow_all_collisions()
    #     zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
    #     zero_pose.send_and_check_goal()

    def test_root_link_not_equal_chain_root(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = u'base_footprint'
        p.pose.position.x = 0.8
        p.pose.position.y = -0.5
        p.pose.position.z = 1
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(u'torso_lift_link', zero_pose.r_tip, p)
        zero_pose.send_and_check_goal()

    # def test_waypoints(self, zero_pose):
    #     """
    #     :type zero_pose: PR2
    #     """
    # FIXME
    #     p = PoseStamped()
    #     p.header.frame_id = zero_pose.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(-0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
    #
    #     zero_pose.add_waypoint()
    #     p = PoseStamped()
    #     p.header.frame_id = zero_pose.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(-0.1, 0, -0.1)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
    #
    #     zero_pose.add_waypoint()
    #     p = PoseStamped()
    #     p.header.frame_id = zero_pose.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(0.2, 0, 0.1)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
    #
    #     zero_pose.send_and_check_goal()


class TestCollisionAvoidanceGoals(object):
    def test_add_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.2, 0, 1.6)
        p.pose.orientation = Quaternion(0.0, 0.0, 0.47942554, 0.87758256)
        zero_pose.add_box(object_name, pose=p)
        m = zero_pose.get_world().get_object(object_name).as_marker_msg()
        compare_poses(m.pose, p.pose)

    def test_add_remove_sphere(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_sphere(object_name, pose=p)
        zero_pose.remove_object(object_name)

    def test_add_remove_cylinder(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_cylinder(object_name, pose=p)
        zero_pose.remove_object(object_name)

    def test_add_urdf_body(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        kitchen_setup.remove_object(u'kitchen')

    def test_attach_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])

    # def test_attach_box_as_eef(self, zero_pose):
    #     """
    #     :type zero_pose: PR2
    #     """
    #     # FIXME works but goal cant be checked
    #     pocky = u'http://muh#pocky'
    #     zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0], [1, 0, 0, 0])
    #     p = PoseStamped()
    #     p.header.frame_id = zero_pose.r_tip
    #     p.pose.orientation.w = 1
    #     # rospy.sleep(5)
    #     for i in range(20):
    #         zero_pose.loop_once()
    #         rospy.sleep(0.1)
    #
    #     # zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
    #     zero_pose.set_and_check_cart_goal(zero_pose.default_root, pocky, p)

    def test_attach_remove_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])
        zero_pose.detach_object(pocky)
        zero_pose.remove_object(pocky)

    def test_attach_existing_box(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(0.05, 0, 0)
        p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, [0.1, 0.02, 0.02], pose=p)
        zero_pose.attach_existing(pocky, frame_id=zero_pose.r_tip)
        relative_pose = zero_pose.get_robot().get_fk(zero_pose.r_tip, pocky).pose
        compare_poses(p.pose, relative_pose)

    def test_attach_existing_box2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        old_p = PoseStamped()
        old_p.header.frame_id = zero_pose.r_tip
        old_p.pose.position = Point(0.05, 0, 0)
        old_p.pose.orientation = Quaternion(0., 0., 0.47942554, 0.87758256)
        zero_pose.add_box(pocky, [0.1, 0.02, 0.02], pose=old_p)
        zero_pose.attach_existing(pocky, frame_id=zero_pose.r_tip)
        relative_pose = zero_pose.get_robot().get_fk(zero_pose.r_tip, pocky).pose
        compare_poses(old_p.pose, relative_pose)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1.0
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
        p.header.frame_id = u'map'
        p.pose.position.y = -1
        p.pose.orientation = Quaternion(0, 0, 0.47942554, 0.87758256)
        zero_pose.move_base(p)

        zero_pose.detach_object(pocky)
        # compare_poses(old_p.pose, new_pose)



    def test_attach_to_nonexistant_robot_link(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], u'', [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

    def test_add_remove_object(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        object_name = u'muh'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 1.2
        p.pose.position.y = 0
        p.pose.position.z = 1.6
        p.pose.orientation.w = 1
        zero_pose.add_box(object_name, pose=p)
        zero_pose.remove_object(object_name)
        # FIXME marker does not get removed

    def test_invalid_update_world(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        req = UpdateWorldRequest(42, WorldBody(), True, PoseStamped())
        assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION

    def test_missing_body_error(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.remove_object(u'muh', expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

    def test_corrupt_shape_error(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, WorldBody(type=WorldBody.PRIMITIVE_BODY,
                                                                   shape=SolidPrimitive(type=42)), True, PoseStamped())
        assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR

    def test_unsupported_options(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        """
        wb = WorldBody()
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(u'map')
        pose.pose.position = Point()
        pose.pose.orientation = Quaternion(w=1)
        wb.type = WorldBody.URDF_BODY

        req = UpdateWorldRequest(UpdateWorldRequest.ADD, wb, True, pose)
        assert kitchen_setup.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.UNSUPPORTED_OPTIONS

    def test_link_b_set_but_body_b_not(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_bs = [u'asdf']
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.INSOLVABLE)

    def test_unknown_robot_link(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [u'asdf']
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_unknown_body_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'asdf'
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_unknown_link_b(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        ce.link_bs = [u'asdf']
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_base_link_in_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.allow_self_collision()
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = -0.2
        p.pose.orientation.w = 1
        zero_pose.add_box(pose=p)
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_unknown_object1(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'muh'
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    # def test_interrupt1(self, box_setup):
    #     """
    #     :type box_setup: Context
    #     """
    #     # FIXME
    #     box_setup.add_box()
    #     p = PoseStamped()
    #     p.header.frame_id = box_setup.r_tip
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 0.5
    #     box_setup.set_collision_entries([collision_entry])
    #
    #     box_setup.send_goal(wait=False)
    #     rospy.sleep(.5)
    # box_setup.interrupt()
    # result = self.giskard.get_result()
    # self.assertEqual(result.error_code, MoveResult.INTERRUPTED)

    def test_allow_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.1)
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.1)

    def test_allow_self_collision2(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_allow_self_collision3(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.18
        p.pose.position.z = 0.02
        p.pose.orientation.w = 1

        ces = []
        ces.append(CollisionEntry(type=CollisionEntry.ALLOW_COLLISION,
                                  robot_links=zero_pose.get_l_gripper_links(),
                                  body_b=u'pr2',
                                  link_bs=zero_pose.get_r_forearm_links()))
        zero_pose.add_collision_entries(ces)

        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_avoid_self_collision(self, zero_pose):
        """
        :type zero_pose: PR2
        """
        goal_js = {
            u'l_elbow_flex_joint': -1.43286344265,
            u'l_forearm_roll_joint': 1.26465060073,
            u'l_shoulder_lift_joint': 0.47990329056,
            u'l_shoulder_pan_joint': 0.281272240139,
            u'l_upper_arm_roll_joint': 0.528415402668,
            u'l_wrist_flex_joint': -1.18811419869,
            u'l_wrist_roll_joint': 2.26884630124,
        }
        zero_pose.allow_all_collisions()
        zero_pose.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
        zero_pose.send_goal()
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)

    def test_avoid_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        ce.min_dist = 0.05
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.SUCCESS)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_allow_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = u'box'
        collision_entry.link_bs = [u'box']
        box_setup.wrapper.set_collision_entries([collision_entry])

        box_setup.allow_self_collision()
        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.0)

    def test_avoid_collision2(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        # box_setup.wrapper.avoid_collision()

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'box'
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_avoid_collision_with_far_object(self, pocky_pose_setup):
        """
        :type pocky_pose_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position.x = 25
        p.pose.position.y = 25
        p.pose.position.z = 25
        p.pose.orientation.w = 1
        pocky_pose_setup.add_box(pose=p)
        p = PoseStamped()
        p.header.frame_id = pocky_pose_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        pocky_pose_setup.set_cart_goal(pocky_pose_setup.default_root, pocky_pose_setup.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'box'
        pocky_pose_setup.add_collision_entries([collision_entry])

        pocky_pose_setup.send_and_check_goal()
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_l_gripper_links(), 0.048)
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_r_gripper_links(), 0.048)

    def test_avoid_all_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_get_out_of_collision(self, box_setup):
        """
        :type box_setup: PR2
        """
        # FIXME fails because weight of unused joints is set to 1 instead of 0
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal()

        # p = PoseStamped()
        # p.header.frame_id = box_setup.r_tip
        # p.pose.position = Point(0.1, 0, 0)
        # p.pose.orientation = Quaternion(0, 0, 0, 1)
        # box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        box_setup.add_collision_entries([collision_entry])

        box_setup.send_and_check_goal()

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_allow_collision_gripper(self, box_setup):
        """
        :type box_setup: PR2
        """
        ces = box_setup.get_allow_l_gripper(u'box')
        box_setup.add_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.11
        p.pose.orientation.w = 1
        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.l_tip, p)
        box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_attached_collision1(self, box_setup):
        """
        :type box_setup: PR2
        """
        attached_link_name = u'pocky'
        box_setup.attach_box(attached_link_name, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0])
        box_setup.attach_box(attached_link_name, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.11
        p.pose.orientation.w = 1
        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)
        box_setup.check_cpi_geq([attached_link_name], 0.048)
        box_setup.detach_object(attached_link_name)

    def test_attached_self_collision(self, zero_pose):
        """
        :type box_setup: PR2
        """

        collision_pose = {
            u'l_elbow_flex_joint': - 1.1343683863086362,
            u'l_forearm_roll_joint': 7.517553513504836,
            u'l_shoulder_lift_joint': 0.5726770101613905,
            u'l_shoulder_pan_joint': 0.1592669164939349,
            u'l_upper_arm_roll_joint': 0.5532568387077381,
            u'l_wrist_flex_joint': - 1.215660155912625,
            u'l_wrist_roll_joint': 4.249300323527076,
            u'r_forearm_roll_joint': 0.0,
            u'r_shoulder_lift_joint': 0.0,
            u'r_shoulder_pan_joint': 0.0,
            u'r_upper_arm_roll_joint': 0.0,
            u'r_wrist_flex_joint': 0.0,
            u'r_wrist_roll_joint': 0.0,
            u'r_elbow_flex_joint': 0.0,
            u'torso_lift_joint': 0.2}

        zero_pose.set_joint_goal(collision_pose)
        zero_pose.send_goal()

        # zero_pose.avoid_all_collisions()

        attached_link_name = u'pocky'
        zero_pose.attach_box(attached_link_name, [0.08, 0.02, 0.02], zero_pose.l_tip, [0.04, 0, 0])

        zero_pose.set_joint_goal({u'torso_lift_joint': 0.2})

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.z = 0.20
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
        zero_pose.send_goal()

        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.048)
        zero_pose.check_cpi_geq([attached_link_name], 0.048)
        zero_pose.detach_object(attached_link_name)

    def test_attached_collision_allow(self, box_setup):
        """
        :type box_setup: PR2
        """
        pocky = u'http://muh#pocky'
        box_setup.attach_box(pocky, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0])

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [pocky]
        ce.body_b = u'box'
        ces.append(ce)
        box_setup.add_collision_entries(ces)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.y = -0.11
        p.pose.orientation.w = 1
        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)

    # def test_collision_during_planning1(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    # FIXME feature not implemented
    #     # FIXME sometimes says endless wiggle detected
    #     p = PoseStamped()
    #     p.header.frame_id = box_setup.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 1
    #     box_setup.add_collision_entries([collision_entry])
    #     box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
    #     box_setup.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)

    # def test_collision_during_planning2(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    # FIXME feature not implemented
    #     # FIXME sometimes says endless wiggle detected
    #     p = PoseStamped()
    #     p.header.frame_id = box_setup.r_tip
    #     p.header.stamp = rospy.get_rostime()
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 1
    #     box_setup.add_collision_entries([collision_entry])
    #     box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
    #     box_setup.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)
    #
    #     box_setup.set_joint_goal(pocky_pose)
    #     box_setup.allow_all_collisions()
    #     box_setup.send_and_check_goal()

    def test_avoid_collision_gripper(self, box_setup):
        """
        :type box_setup: PR2
        """
        box_setup.allow_all_collisions()
        ces = box_setup.get_l_gripper_collision_entries(u'box', 0.05, CollisionEntry.AVOID_COLLISION)
        box_setup.add_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.
        p.pose.orientation.w = 1
        box_setup.set_cart_goal(box_setup.default_root, box_setup.l_tip, p)
        box_setup.send_goal()
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.049)

    # def test_grasping(self, fake_table_setup):
    #     """
    #     :type fake_table_setup: PR2
    #     """
    #     pocky = u'http://muh#pocky'
    #     fake_table_setup.add_box(pocky, [0.02, 0.02, 0.1], u'map', [.5, -0.1, .77])
    #     goal_pose = PoseStamped()
    #     goal_pose.header.frame_id = u'map'
    #     goal_pose.pose.position = Point(.5, -0.13, .77)
    #     goal_pose.pose.orientation = Quaternion(*quaternion_from_axis_angle([0,0,1], np.pi/2))
    #     fake_table_setup.avoid_all_collisions(0.025)
    #     # fake_table_setup.allow_all_collisions()
    #     fake_table_setup.set_and_check_cart_goal(fake_table_setup.default_root, fake_table_setup.r_tip, goal_pose)
    #     pass

    #
    # def test_end_state_collision(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    #     # TODO endstate impossible as long as we check for path collision?
    #     pass

    # def test_filled_vel_values(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    #     pass
    #
    # def test_undefined_goal(self, box_setup):
    #     """
    #     :type box_setup: PR2
    #     """
    #     pass

    # TODO test plan only

    # TODO test translation and orientation goal in different frame

    def test_pick_and_place(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        :return:
        """

        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position = Point(0.760, 0.480, 0.000)
        base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.230, 0.973)
        kitchen_setup.move_pr2_base(base_pose)
        attached_link_name = u'edekabowl'
        p = PoseStamped()
        p.header.frame_id = u'map'
        p.pose.position = Point(1.39985, 0.799920, 0.888)
        p.pose.orientation = Quaternion(-0.0037, -0.00476, 0.3921, 0.9198)
        kitchen_setup.add_box(attached_link_name, [.145, .145, .072], pose=p)

        pick_pose = PoseStamped()
        pick_pose.header.frame_id = u'base_footprint'
        pick_pose.pose.position = Point(0.649, -0.023, 0.918)
        pick_pose.pose.orientation = Quaternion(0.407, 0.574, -0.408, 0.582)

        # pregrasp
        pick_pose.pose.position.z += 0.2
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, pick_pose)

        # grasp
        pick_pose.pose.position.z -= 0.2
        kitchen_setup.avoid_collision(kitchen_setup.get_l_gripper_links(), u'kitchen', [], 0)
        kitchen_setup.allow_collision(kitchen_setup.get_l_gripper_links(), attached_link_name, [])
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, pick_pose)
        kitchen_setup.attach_existing(attached_link_name, frame_id=kitchen_setup.l_tip)

        # post grasp
        pick_pose.pose.position.z += 0.2
        kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, pick_pose)
        # kitchen_setup.remove_object(attached_link_name)
        kitchen_setup.send_and_check_joint_goal(gaya_pose)

        # place============================
        base_pose.pose.position = Point(-0.200, 1.120, 0.000)
        base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.994, -0.105)
        kitchen_setup.move_pr2_base(base_pose)

        # pre place
        place_pose = PoseStamped()
        place_pose.header.frame_id = u'base_footprint'
        place_pose.pose.position = Point(0.587, 0.068, 0.920)
        place_pose.pose.orientation = Quaternion(0.703, -0.074, -0.703, -0.074)
        place_pose.pose.position.z += 0.2
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, place_pose)

        # place
        place_pose.pose.position.z -= 0.19
        kitchen_setup.avoid_all_collisions(0.)
        kitchen_setup.set_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, place_pose)
        kitchen_setup.send_goal()
        rospy.sleep(1)

        # post place
        kitchen_setup.detach_object(attached_link_name)
        place_pose.pose.position.z += 0.2
        kitchen_setup.avoid_all_collisions(0.)
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, place_pose)

    def test_hand_in_kitchen(self, kitchen_setup):
        """
        :type kitchen_setup: PR2
        :return:
        """

        base_pose = PoseStamped()
        base_pose.header.frame_id = u'map'
        base_pose.pose.position = Point(0.743, 0.586, 0.000)
        base_pose.pose.orientation.w = 1
        kitchen_setup.move_pr2_base(base_pose)

        # grasp
        p = PoseStamped()
        p.header.frame_id = kitchen_setup.l_tip
        p.pose.position.x = 0.09
        p.pose.orientation.w = 1
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, p)

        # p = PoseStamped()
        # p.header.frame_id = kitchen_setup.l_tip
        # p.pose.position.x = -0.1
        # p.pose.orientation.w = 1
        # kitchen_setup.avoid_all_collisions(0.05)
        # kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, p)
        # cpi = kitchen_setup.get_cpi(0.05)
        # cpi.

        # post grasp
        pregrasp_pose = PoseStamped()
        pregrasp_pose.header.frame_id = u'base_footprint'
        pregrasp_pose.pose.position.x = 0.611175722907
        pregrasp_pose.pose.position.y = -0.0244662287535
        pregrasp_pose.pose.position.z = 1.10803325995
        pregrasp_pose.pose.orientation.x = -0.0128682380997
        pregrasp_pose.pose.orientation.y = -0.710292569338
        pregrasp_pose.pose.orientation.z = 0.0148339707762
        pregrasp_pose.pose.orientation.w = -0.703632573456
        kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.set_and_check_cart_goal(kitchen_setup.default_root, kitchen_setup.l_tip, pregrasp_pose)
        # kitchen_setup.check_cpi_geq([u'edekabowl'], )

    def test_set_kitchen_joint_state(self):
        pass

    #
    # def test_place_spoon1(self):
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'map'
    #     base_pose.pose.position = Point(-1.010, -0.152, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, -0.707, 0.707)
    #     self.move_base(base_pose)
    #
    #     goal_js = {
    #         'r_elbow_flex_joint': -1.43322543123,
    #         'r_forearm_roll_joint': pi / 2,
    #         'r_gripper_joint': 2.22044604925e-16,
    #         'r_shoulder_lift_joint': 0,
    #         'r_shoulder_pan_joint': -1.39478600655,
    #         'r_upper_arm_roll_joint': -pi / 2,
    #         'r_wrist_flex_joint': -0.1001,
    #         'r_wrist_roll_joint': 0,
    #
    #         'l_elbow_flex_joint': -1.53432386765,
    #         'l_forearm_roll_joint': -0.335634766956,
    #         'l_gripper_joint': 2.22044604925e-16,
    #         'l_shoulder_lift_joint': 0.199493756207,
    #         'l_shoulder_pan_joint': 0.854317292495,
    #         'l_upper_arm_roll_joint': 1.90837777308,
    #         'l_wrist_flex_joint': -0.623267982468,
    #         'l_wrist_roll_joint': -0.910310693429,
    #
    #         'torso_lift_joint': 0.206303584043,
    #     }
    #     self.set_and_check_js_goal(goal_js)
    #     self.add_kitchen()
    #     self.giskard.avoid_collision(0.05, body_b='kitchen')
    #     p = PoseStamped()
    #     p.header.frame_id = 'base_footprint'
    #     p.pose.position = Point(0.69, -0.374, 0.82)
    #     p.pose.orientation = Quaternion(-0.010, 0.719, 0.006, 0.695)
    #     self.set_and_check_cart_goal(self.default_root, self.r_tip, p)

    #
    # def test_pick_up_spoon(self):
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'map'
    #     base_pose.pose.position = Point(0.365, 0.368, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, -0.007, 1.000)
    #     self.move_base(base_pose)
    #
    #     self.giskard.allow_all_collisions()
    #     self.set_and_check_js_goal(gaya_pose)
    #
    #     self.add_kitchen()
    #     kitchen_js = {'sink_area_left_upper_drawer_main_joint': 0.45}
    #     self.giskard.set_object_joint_state('kitchen', kitchen_js)
    #     rospy.sleep(.5)
    #
    #     # put gripper above drawer
    #     pick_spoon_pose = PoseStamped()
    #     pick_spoon_pose.header.frame_id = 'base_footprint'
    #     pick_spoon_pose.pose.position = Point(0.567, 0.498, 0.89)
    #     pick_spoon_pose.pose.orientation = Quaternion(0.018, 0.702, 0.004, 0.712)
    #     self.set_and_check_cart_goal(self.default_root, self.l_tip, pick_spoon_pose)
    #
    #     #put gripper in drawer
    #     self.giskard.set_collision_entries(self.get_allow_l_gripper('kitchen'))
    #     p = PoseStamped()
    #     p.header.frame_id = self.l_tip
    #     p.pose.position.x = 0.1
    #     p.pose.orientation.w = 1
    #     self.set_and_check_cart_goal(self.default_root, self.l_tip, p)
    #
    #     #attach spoon
    #     r = self.giskard.attach_box('pocky', [0.02, 0.02, 0.1], self.l_tip, [0, 0, 0])
    #     self.assertEqual(r.error_codes, UpdateWorldResponse.SUCCESS)
    #
    #     # allow grippe and spoon
    #     ces = self.get_allow_l_gripper('kitchen')
    #     ce = CollisionEntry()
    #     ce.type = CollisionEntry.ALLOW_COLLISION
    #     ce.robot_link = 'pocky'
    #     ce.body_b = 'kitchen'
    #     ces.append(ce)
    #     self.giskard.set_collision_entries(ces)
    #
    #     # pick up
    #     p = PoseStamped()
    #     p.header.frame_id = self.l_tip
    #     p.pose.position.x = -0.1
    #     p.pose.orientation.w = 1
    #     self.set_and_check_cart_goal(self.default_root, self.l_tip, p)
    #
