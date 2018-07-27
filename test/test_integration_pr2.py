from multiprocessing import Queue
from threading import Thread
import numpy as np
import pytest
import rospy
from numpy import pi
from angles import normalize_angle, normalize_angle_positive, shortest_angular_distance
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveActionResult, CollisionEntry, MoveActionGoal, MoveResult
from giskard_msgs.srv import UpdateWorldResponse
from sensor_msgs.msg import JointState

from giskardpy.python_interface import GiskardWrapper
from giskardpy.test_utils import GiskardTestWrapper
from giskardpy.tfwrapper import transform_pose, lookup_transform, init as tf_init
from giskardpy.utils import to_list
from ros_trajectory_controller_main import giskard_pm

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



@pytest.fixture(scope='module')
def ros(request):
    print('init ros')
    rospy.init_node('tests')
    tf_init(60)

    def kill_ros():
        print('shutdown ros')
        rospy.signal_shutdown('die')

    request.addfinalizer(kill_ros)


@pytest.fixture(scope='module')
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
    giskard.set_joint_goal(default_pose)

    collision_entry = CollisionEntry()
    collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
    giskard.wrapper.set_collision_entries([collision_entry])
    giskard.send_and_check_goal()
    giskard.check_joint_state(default_pose)
    giskard.reset_pr2_base()
    return giskard


@pytest.fixture()
def pocky(resetted_giskard):
    resetted_giskard.send_and_check_joint_goal(pocky_pose)
    resetted_giskard.add_box()


class TestJointGoals(object):
    def test_joint_movement1(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        resetted_giskard.send_and_check_joint_goal(pocky_pose)

    def test_partial_joint_state_goal1(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        js = dict(pocky_pose.items()[:3])
        resetted_giskard.send_and_check_joint_goal(js)

    def test_continuous_joint1(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        js = {u'r_wrist_roll_joint': -pi,
              u'l_wrist_roll_joint': 3.5 * pi, }
        resetted_giskard.send_and_check_joint_goal(js)


class TestCartGoals(object):
    def test_cart_goal_1eef(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

    def test_cart_goal_2eef(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        # FIXME? eef don't move at the same time
        r_goal = PoseStamped()
        r_goal.header.frame_id = resetted_giskard.r_tip
        r_goal.header.stamp = rospy.get_rostime()
        r_goal.pose.position = Point(-0.1, 0, 0)
        r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, r_goal)
        l_goal = PoseStamped()
        l_goal.header.frame_id = resetted_giskard.l_tip
        l_goal.header.stamp = rospy.get_rostime()
        l_goal.pose.position = Point(-0.05, 0, 0)
        l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.l_tip, l_goal)
        resetted_giskard.send_and_check_goal()
        resetted_giskard.check_cart_goal(resetted_giskard.r_tip, r_goal)
        resetted_giskard.check_cart_goal(resetted_giskard.l_tip, l_goal)

    def test_weird_wiggling(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        # FIXME get rid of wiggling
        goal_js = {
            u'torso_lift_joint': 0.137863459754,

            u'r_upper_arm_roll_joint': -3.24008158875,
            u'r_shoulder_pan_joint': 0.206949462075,
            u'r_shoulder_lift_joint': -0.249453873652,
            u'r_forearm_roll_joint': 1.83979114862,
            u'r_elbow_flex_joint': -1.36820120012,
            u'r_wrist_flex_joint': -1.52492789587,
            u'r_wrist_roll_joint': -13.6248743778,

            u'l_upper_arm_roll_joint': 1.63487737202,
            u'l_shoulder_pan_joint': 1.36222920328,
            u'l_shoulder_lift_joint': 0.229120778526,
            u'l_forearm_roll_joint': 13.7578920265,
            u'l_elbow_flex_joint': -1.48141189643,
            u'l_wrist_flex_joint': -1.22662876066,
            u'l_wrist_roll_joint': -53.6150824007,
        }
        resetted_giskard.allow_all_collisions()
        resetted_giskard.send_and_check_joint_goal(goal_js)

        p = PoseStamped()
        p.header.frame_id = resetted_giskard.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.1
        p.pose.orientation.w = 1
        resetted_giskard.allow_all_collisions()
        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.l_tip, p)

        p = PoseStamped()
        p.header.frame_id = resetted_giskard.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.2
        p.pose.orientation.w = 1
        # self.giskard.allow_all_collisions()
        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.l_tip, p)

    def test_root_link_not_equal_chain_root(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = u'base_footprint'
        p.pose.position.x = 0.8
        p.pose.position.y = -0.5
        p.pose.position.z = 1
        p.pose.orientation.w = 1
        resetted_giskard.allow_all_collisions()
        resetted_giskard.set_cart_goal(u'torso_lift_link', resetted_giskard.r_tip, p)
        resetted_giskard.send_and_check_goal()

    def test_waypoints(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        resetted_giskard.add_waypoint()
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(-0.1, 0, -0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        resetted_giskard.add_waypoint()
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.2, 0, 0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        resetted_giskard.send_and_check_goal()

    # def test_interrupt1(self, resetted_giskard):
    #     """
    #     :type resetted_giskard: Context
    #     """
    #     # FIXME
    #     resetted_giskard.add_box()
    #     p = PoseStamped()
    #     p.header.frame_id = resetted_giskard.r_tip
    #     p.pose.position = Point(0.1, 0, 0)
    #     p.pose.orientation = Quaternion(0, 0, 0, 1)
    #     resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)
    #
    #     collision_entry = CollisionEntry()
    #     collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
    #     collision_entry.min_dist = 0.5
    #     resetted_giskard.set_collision_entries([collision_entry])
    #
    #     resetted_giskard.send_goal(wait=False)
    #     rospy.sleep(.5)
    #     resetted_giskard.interrupt()
    #     result = self.giskard.get_result()
    #     self.assertEqual(result.error_code, MoveResult.INTERRUPTED)


class TestCollisionAvoidanceGoals(object):
    def test_base_link_in_collision(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        # FIXME fails with self collision avoidance
        resetted_giskard.add_box(position=[0, 0, -0.2])
        resetted_giskard.send_and_check_joint_goal(pocky_pose)

    def test_unknown_object1(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        # TODO should we throw unknown object?
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'muh'
        resetted_giskard.add_collision_entries([collision_entry])

        resetted_giskard.send_and_check_goal(MoveResult.UNKNOWN_OBJECT)

    def test_allow_collision(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.body_b = u'box'
        resetted_giskard.wrapper.set_collision_entries([collision_entry])

        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

    def test_avoid_collision1(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        # resetted_giskard.send_and_check_joint_goal(pocky_pose)
        # resetted_giskard.add_box()
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = 0.05
        collision_entry.body_b = u'box'
        resetted_giskard.add_collision_entries([collision_entry])

        resetted_giskard.send_and_check_goal()

    def test_avoid_all_collision1(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        resetted_giskard.add_collision_entries([collision_entry])

        resetted_giskard.send_and_check_goal()

    def test_get_out_of_collision(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.pose.position = Point(0.15, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        resetted_giskard.add_collision_entries([collision_entry])

        resetted_giskard.send_and_check_goal()

        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 0.05
        resetted_giskard.add_collision_entries([collision_entry])

        resetted_giskard.send_and_check_goal()

    def test_allow_collision_gripper(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        ces = resetted_giskard.get_allow_l_gripper(u'box')
        resetted_giskard.add_collision_entries(ces)
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.l_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = 0.11
        p.pose.orientation.w = 1
        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.l_tip, p)

    def test_attached_collision1(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        resetted_giskard.attach_box('pocky', [0.1, 0.02, 0.02], resetted_giskard.r_tip, [0.05, 0, 0])
        resetted_giskard.attach_box('pocky', [0.1, 0.02, 0.02], resetted_giskard.r_tip, [0.05, 0, 0],
                                    expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.x = -0.11
        p.pose.orientation.w = 1
        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)
        resetted_giskard.remove_object('pocky')

    def test_attach_existing_object(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        # TODO if we don't allow there to be attached objects and free objects of the same name, we also have to check robot links?
        pocky = 'http://muh#pocky'
        resetted_giskard.add_box(pocky, position=[1.2, 0, 1.6])
        resetted_giskard.attach_box(pocky, [0.1, 0.02, 0.02], resetted_giskard.r_tip, [0.05, 0, 0],
                                    expected_response=UpdateWorldResponse.DUPLICATE_BODY_ERROR)

    def test_add_remove_object(self, resetted_giskard):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        object_name = 'muh'
        resetted_giskard.add_box(object_name, position=[1.2, 0, 1.6])
        resetted_giskard.remove_object(object_name)

    def test_attached_collision_avoidance(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        pocky = 'http://muh#pocky'
        resetted_giskard.attach_box(pocky, [0.1, 0.02, 0.02], resetted_giskard.r_tip, [0.05, 0, 0])

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_link = pocky
        ce.body_b = 'box'
        ces.append(ce)
        resetted_giskard.add_collision_entries(ces)

        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position.y = -0.11
        p.pose.orientation.w = 1
        resetted_giskard.set_and_check_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)

    def test_collision_during_planning1(self, resetted_giskard, pocky):
        """
        :type resetted_giskard: GiskardTestWrapper
        """
        p = PoseStamped()
        p.header.frame_id = resetted_giskard.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_ALL_COLLISIONS
        collision_entry.min_dist = 1
        resetted_giskard.add_collision_entries([collision_entry])
        resetted_giskard.set_cart_goal(resetted_giskard.default_root, resetted_giskard.r_tip, p)
        resetted_giskard.send_and_check_goal(expected_error_code=MoveResult.PATH_COLLISION)

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
