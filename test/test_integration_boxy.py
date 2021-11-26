import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped, PointStamped
from tf.transformations import quaternion_from_matrix

from giskardpy.utils.tfwrapper import lookup_point, transform_point, \
    transform_pose
from utils_for_tests import Donbot, Boxy


# TODO roslaunch iai_boxy_sim ros_control_sim.launch


@pytest.fixture(scope=u'module')
def giskard(request, ros):
    c = Boxy()
    request.addfinalizer(c.tear_down)
    return c


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.plan_and_execute()


class TestConstraints(object):
    def test_pointing(self, better_pose):
        """
        :type better_pose: Boxy
        """
        tip = u'head_mount_kinect2_rgb_optical_frame'
        goal_point = lookup_point(u'map', better_pose.r_tip)
        better_pose.set_pointing_goal(tip, goal_point)
        better_pose.plan_and_execute()

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.z = 1

        expected_x = transform_point(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.x, 0, 2)

        goal_point = lookup_point(u'map', better_pose.r_tip)
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

        better_pose.set_cart_goal(r_goal, better_pose.r_tip, u'base_footprint')
        better_pose.plan_and_execute()

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.z = 1

        expected_x = lookup_point(tip, better_pose.r_tip)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.x, 0, 2)

    def test_open_drawer(self, kitchen_setup):  # where is the kitchen_setup actually loaded
        """"
        :type kitchen_setup: Boxy
        """
        handle_frame_id = u'iai_kitchen/sink_area_left_middle_drawer_handle'
        handle_name = u'sink_area_left_middle_drawer_handle'
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.y = 1

        kitchen_setup.set_json_goal(u'GraspBar',
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

        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.48})

        # Close drawer partially
        kitchen_setup.set_json_goal(u'Open',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name,
                                    goal_joint_state=0.2)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.2})

        kitchen_setup.set_json_goal(u'Close',
                                    tip_link=kitchen_setup.l_tip,
                                    environment_link=handle_name)
        kitchen_setup.allow_all_collisions()  # makes execution faster
        kitchen_setup.plan_and_execute()  # send goal to Giskard
        # Update kitchen object
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.0})
