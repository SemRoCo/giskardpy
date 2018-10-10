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

from giskardpy.python_interface import GiskardWrapper
from giskardpy.test_utils import GiskardTestWrapper
from giskardpy.tfwrapper import transform_pose, lookup_transform, init as tf_init
from giskardpy.utils import msg_to_list

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
    c = GiskardTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: GiskardTestWrapper
    """
    print(u'resetting giskard')
    giskard.clear_world()
    giskard.reset_pr2_base()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type giskard: GiskardTestWrapper
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
    pocky_pose_setup.add_box()
    return pocky_pose_setup


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
        :type zero_pose: GiskardTestWrapper
        """
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        js = dict(pocky_pose.items()[:3])
        zero_pose.send_and_check_joint_goal(js)

    def test_continuous_joint1(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        js = {u'r_wrist_roll_joint': -pi,
              u'l_wrist_roll_joint': 3.5 * pi, }
        zero_pose.send_and_check_joint_goal(js)

    def test_undefined_type(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.UNDEFINED
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE

    def test_empty_goal(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        goal = MoveActionGoal()
        goal.goal.type = MoveGoal.PLAN_AND_EXECUTE
        result = zero_pose.send_goal(goal)
        assert result.error_code == MoveResult.INSOLVABLE

class TestCartGoals(object):
    def test_cart_goal_1eef(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.r_tip
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)

    def test_cart_goal_2eef(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        # FIXME? eef don't move at the same time
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
        zero_pose.send_and_check_goal()
        zero_pose.check_cart_goal(zero_pose.r_tip, r_goal)
        zero_pose.check_cart_goal(zero_pose.l_tip, l_goal)

    def test_weird_wiggling(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        # FIXME get rid of wiggling
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
        zero_pose.allow_all_collisions()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)

        p = PoseStamped()
        p.header.frame_id = zero_pose.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)

    def test_hot_init_failed(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
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

    def test_endless_wiggling(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
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

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.5
        p.pose.orientation.w = 1
        # self.giskard.allow_all_collisions()
        zero_pose.set_cart_goal(zero_pose.default_root, zero_pose.r_tip, p)
        zero_pose.send_and_check_goal(expected_error_code=MoveResult.INSOLVABLE)

    def test_root_link_not_equal_chain_root(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = u'base_footprint'
        p.pose.position.x = 0.8
        p.pose.position.y = -0.5
        p.pose.position.z = 1
        p.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(u'torso_lift_link', zero_pose.r_tip, p)
        zero_pose.send_and_check_goal()

    # def test_waypoints(self, zero_pose):
    #     """
    #     :type zero_pose: GiskardTestWrapper
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
        :type zero_pose: GiskardTestWrapper
        """
        object_name = u'muh'
        zero_pose.add_box(object_name, position=[1.2, 0, 1.6])

    def test_add_remove_sphere(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        object_name = u'muh'
        zero_pose.add_sphere(object_name, position=[1.2, 0, 1.6])
        zero_pose.remove_object(object_name)

    def test_add_remove_cylinder(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        object_name = u'muh'
        zero_pose.add_cylinder(object_name, position=[1.2, 0, 1.6])
        zero_pose.remove_object(object_name)

    def test_add_urdf_body(self, kitchen_setup):
        """
        :type kitchen_setup: GiskardTestWrapper
        """
        kitchen_setup.remove_object(u'kitchen')

    def test_attach_box(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        pocky = u'http://muh#pocky'
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0])

    def test_add_remove_object(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        object_name = u'muh'
        zero_pose.add_box(object_name, position=[1.2, 0, 1.6])
        zero_pose.remove_object(object_name)

    def test_invalid_update_world(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        req = UpdateWorldRequest(42, WorldBody(), True, PoseStamped())
        assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.INVALID_OPERATION

    def test_missing_body_error(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        zero_pose.remove_object(u'muh', expected_response=UpdateWorldResponse.MISSING_BODY_ERROR)

    def test_attach_existing_object(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        # TODO if we don't allow there to be attached objects and free objects of the same name,
        # we also have to check robot links?
        pocky = u'http://muh#pocky'
        zero_pose.add_box(pocky, position=[1.2, 0, 1.6])
        zero_pose.attach_box(pocky, [0.1, 0.02, 0.02], zero_pose.r_tip, [0.05, 0, 0],
                             expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)

    def test_corrupt_shape_error(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, WorldBody(type=WorldBody.PRIMITIVE_BODY,
                                                                   shape=SolidPrimitive(type=42)), True, PoseStamped())
        assert zero_pose.wrapper.update_world.call(req).error_codes == UpdateWorldResponse.CORRUPT_SHAPE_ERROR

    def test_unsupported_options(self, kitchen_setup):
        """
        :type kitchen_setup: GiskardTestWrapper
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
        :type box_setup: GiskardTestWrapper
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.link_b = u'asdf'
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.INSOLVABLE)

    def test_unknown_robot_link(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_link = u'asdf'
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_unknown_body_b(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'asdf'
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_unknown_link_b(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.body_b = u'box'
        ce.link_b = u'asdf'
        box_setup.add_collision_entries([ce])
        box_setup.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_base_link_in_collision(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
        """
        # FIXME fails with self collision avoidance
        zero_pose.add_box(position=[0, 0, -0.2])
        zero_pose.send_and_check_joint_goal(pocky_pose)

    def test_unknown_object1(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
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
        :type zero_pose: GiskardTestWrapper
        """
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.1)
        zero_pose.check_cpi_geq(zero_pose.get_l_gripper_links(), 0.1)

    def test_allow_self_collision2(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
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
        zero_pose.disable_self_collision()
        zero_pose.set_and_check_cart_goal(zero_pose.default_root, zero_pose.l_tip, p)
        zero_pose.check_cpi_leq(zero_pose.get_l_gripper_links(), 0.01)
        zero_pose.check_cpi_leq([u'r_forearm_link'], 0.01)
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.05)

    def test_allow_self_collision3(self, zero_pose):
        """
        :type zero_pose: GiskardTestWrapper
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
        :type zero_pose: GiskardTestWrapper
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
        zero_pose.check_cpi_geq(zero_pose.get_r_gripper_links(), 0.048)

    def test_avoid_collision(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
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
        :type box_setup: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = u'box'
        collision_entry.link_b = u'base'
        box_setup.wrapper.set_collision_entries([collision_entry])

        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)

        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_leq(box_setup.get_r_gripper_links(), 0.0)

    def test_avoid_collision2(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
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
        :type pocky_pose_setup: GiskardTestWrapper
        """
        pocky_pose_setup.add_box(position=[25, 25, 25])
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
        :type box_setup: GiskardTestWrapper
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
        :type box_setup: GiskardTestWrapper
        """
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

    def test_allow_collision_gripper(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        ces = box_setup.get_allow_l_gripper(u'box')
        box_setup.add_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = box_setup.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.11
        p.pose.orientation.w = 1
        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.l_tip, p)
        # box_setup.check_cpi_leq(box_setup.get_l_gripper_links(), 0.0)
        box_setup.check_cpi_geq(box_setup.get_r_gripper_links(), 0.048)

    def test_attached_collision1(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
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
        box_setup.remove_object(attached_link_name)

    def test_attached_collision_avoidance(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        pocky = 'http://muh#pocky'
        box_setup.attach_box(pocky, [0.1, 0.02, 0.02], box_setup.r_tip, [0.05, 0, 0])

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = pocky
        ce.body_b = 'box'
        ces.append(ce)
        box_setup.add_collision_entries(ces)

        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.y = -0.11
        p.pose.orientation.w = 1
        box_setup.set_and_check_cart_goal(box_setup.default_root, box_setup.r_tip, p)
        box_setup.check_cpi_geq(box_setup.get_l_gripper_links(), 0.048)

    def test_collision_during_planning1(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        # FIXME sometimes says endless wiggle detected
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 1
        box_setup.add_collision_entries([collision_entry])
        box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
        box_setup.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)

    def test_collision_during_planning2(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
        """
        # FIXME sometimes says endless wiggle detected
        p = PoseStamped()
        p.header.frame_id = box_setup.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 1
        box_setup.add_collision_entries([collision_entry])
        box_setup.set_cart_goal(box_setup.default_root, box_setup.r_tip, p)
        box_setup.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)

        box_setup.set_joint_goal(pocky_pose)
        box_setup.allow_all_collisions()
        box_setup.send_and_check_goal()

    def test_avoid_collision_gripper(self, box_setup):
        """
        :type box_setup: GiskardTestWrapper
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


    #
    # def test_end_state_collision(self, box_setup):
    #     """
    #     :type box_setup: GiskardTestWrapper
    #     """
    #     # TODO endstate impossible as long as we check for path collision?
    #     pass

    # def test_filled_vel_values(self, box_setup):
    #     """
    #     :type box_setup: GiskardTestWrapper
    #     """
    #     pass
    #
    # def test_undefined_goal(self, box_setup):
    #     """
    #     :type box_setup: GiskardTestWrapper
    #     """
    #     pass

    # TODO test plan only

    # TODO test translation and orientation goal in different frame

    #
    # def test_place_spoon1(self):
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'map'
    #     base_pose.pose.position = Point(-1.010, -0.152, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, -0.707, 0.707)
    #     self.move_pr2_base(base_pose)
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
    #
    # def test_pick_up_spoon(self):
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = 'map'
    #     base_pose.pose.position = Point(0.365, 0.368, 0.000)
    #     base_pose.pose.orientation = Quaternion(0.000, 0.000, -0.007, 1.000)
    #     self.move_pr2_base(base_pose)
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
