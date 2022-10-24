from copy import copy

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped, PointStamped
from tf.transformations import quaternion_from_matrix

from giskardpy.utils.tfwrapper import lookup_point, transform_point, \
    transform_pose
from utils_for_tests import BoxyTestWrapper


# TODO roslaunch iai_boxy_sim ros_control_sim.launch


class BoxyTestWrapper(GiskardTestWrapper):
    default_pose = {
        'neck_shoulder_pan_joint': 0.0,
        'neck_shoulder_lift_joint': 0.0,
        'neck_elbow_joint': 0.0,
        'neck_wrist_1_joint': 0.0,
        'neck_wrist_2_joint': 0.0,
        'neck_wrist_3_joint': 0.0,
        'triangle_base_joint': 0.0,
        'left_arm_0_joint': 0.0,
        'left_arm_1_joint': 0.0,
        'left_arm_2_joint': 0.0,
        'left_arm_3_joint': 0.0,
        'left_arm_4_joint': 0.0,
        'left_arm_5_joint': 0.0,
        'left_arm_6_joint': 0.0,
        'right_arm_0_joint': 0.0,
        'right_arm_1_joint': 0.0,
        'right_arm_2_joint': 0.0,
        'right_arm_3_joint': 0.0,
        'right_arm_4_joint': 0.0,
        'right_arm_5_joint': 0.0,
        'right_arm_6_joint': 0.0,
    }

    better_pose = {
        'neck_shoulder_pan_joint': -1.57,
        'neck_shoulder_lift_joint': -1.88,
        'neck_elbow_joint': -2.0,
        'neck_wrist_1_joint': 0.139999387693,
        'neck_wrist_2_joint': 1.56999999998,
        'neck_wrist_3_joint': 0,
        'triangle_base_joint': -0.24,
        'left_arm_0_joint': -0.68,
        'left_arm_1_joint': 1.08,
        'left_arm_2_joint': -0.13,
        'left_arm_3_joint': -1.35,
        'left_arm_4_joint': 0.3,
        'left_arm_5_joint': 0.7,
        'left_arm_6_joint': -0.01,
        'right_arm_0_joint': 0.68,
        'right_arm_1_joint': -1.08,
        'right_arm_2_joint': 0.13,
        'right_arm_3_joint': 1.35,
        'right_arm_4_joint': -0.3,
        'right_arm_5_joint': -0.7,
        'right_arm_6_joint': 0.01,
    }

    def __init__(self, config=None):
        self.camera_tip = 'camera_link'
        self.r_tip = 'right_gripper_tool_frame'
        self.l_tip = 'left_gripper_tool_frame'
        self.robot_name = 'boxy'
        self.r_gripper_group = 'r_gripper'
        super().__init__(Boxy)

    def move_base(self, goal_pose):
        goal_pose = tf.transform_pose(self.default_root, goal_pose)
        js = {'odom_x_joint': goal_pose.pose.position.x,
              'odom_y_joint': goal_pose.pose.position.y,
              'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                      goal_pose.pose.orientation.y,
                                                                      goal_pose.pose.orientation.z,
                                                                      goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset(self):
        self.clear_world()
        self.reset_base()
        self.register_group(self.r_gripper_group, self.get_robot_name(), 'right_arm_7_link')


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = BoxyTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        js = copy(zero_pose.better_pose)
        js['triangle_base_joint'] = zero_pose.default_pose['triangle_base_joint']
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()


class TestConstraints(object):
    def test_pointing(self, better_pose):
        """
        :type better_pose: BoxyTestWrapper
        """
        tip = 'head_mount_kinect2_rgb_optical_frame'
        goal_point = lookup_point('map', better_pose.r_tip)
        better_pose.set_pointing_goal(tip, goal_point)
        better_pose.plan_and_execute()

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.z = 1

        expected_x = transform_point(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.x, 0, 2)

        goal_point = lookup_point('map', better_pose.r_tip)
        better_pose.set_pointing_goal(tip, goal_point, root_link=better_pose.r_tip)

        r_goal = PoseStamped()
        r_goal.header.frame_id = better_pose.r_tip
        r_goal.pose.position.x -= 0.2
        r_goal.pose.position.z -= 0.5
        r_goal.pose.orientation.w = 1
        r_goal = transform_pose(better_pose.default_root, r_goal)
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                      [0, -1, 0, 0],
                                                                      [1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))

        better_pose.set_cart_goal(r_goal, better_pose.r_tip, 'base_footprint')
        better_pose.plan_and_execute()

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.z = 1

        expected_x = lookup_point(tip, better_pose.r_tip)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.x, 0, 2)

    def test_open_drawer(self, kitchen_setup):  # where is the kitchen_setup actually loaded
        """"
        :type kitchen_setup: BoxyTestWrapper
        """
        handle_frame_id = 'iai_kitchen/sink_area_left_middle_drawer_handle'
        handle_name = 'sink_area_left_middle_drawer_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.y = 1

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
                                    tip_link=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.4)

        # Create gripper from kitchen object
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.z = 1

        # Get goal for grasping the handle
        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        # Align planes for gripper to be horizontal/vertical
        kitchen_setup.set_align_planes_goal(kitchen_setup.l_tip,
                                            x_gripper,
                                            root_normal=x_goal)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard

        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.48})

        # Close drawer partially
        kitchen_setup.set_json_goal('Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0.2)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.2})

        kitchen_setup.set_json_goal('Close',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({'sink_area_left_middle_drawer_main_joint': 0.0})
