import numpy as np
import pytest
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped, PointStamped
from tf.transformations import quaternion_from_matrix
import giskardpy.tfwrapper as tf
from giskardpy import logging
from giskardpy.tfwrapper import lookup_transform, init as tf_init, lookup_point, transform_point, \
    transform_pose
from utils_for_tests import Donbot, Boxy

# TODO roslaunch iai_donbot_sim ros_control_sim.launch


default_js = {
    u'neck_shoulder_pan_joint': 0.0,
    u'neck_shoulder_lift_joint': 0.0,
    u'neck_elbow_joint': 0.0,
    u'neck_wrist_1_joint': 0.0,
    u'neck_wrist_2_joint': 0.0,
    u'neck_wrist_3_joint': 0.0,
    u'triangle_base_joint': 0.0,
    u'left_arm_0_joint': 0.0,
    u'left_arm_1_joint': 0.0,
    u'left_arm_2_joint': 0.0,
    u'left_arm_3_joint': 0.0,
    u'left_arm_4_joint': 0.0,
    u'left_arm_5_joint': 0.0,
    u'left_arm_6_joint': 0.0,
    u'right_arm_0_joint': 0.0,
    u'right_arm_1_joint': 0.0,
    u'right_arm_2_joint': 0.0,
    u'right_arm_3_joint': 0.0,
    u'right_arm_4_joint': 0.0,
    u'right_arm_5_joint': 0.0,
    u'right_arm_6_joint': 0.0,
}
better_js = {
    u'neck_shoulder_pan_joint': -1.57,
    u'neck_shoulder_lift_joint': -1.88,
    u'neck_elbow_joint': -2.0,
    u'neck_wrist_1_joint': 0.139999387693,
    u'neck_wrist_2_joint': 1.56999999998,
    u'neck_wrist_3_joint': 0,
    u'triangle_base_joint': -0.24,
    u'left_arm_0_joint': -0.68,
    u'left_arm_1_joint': 1.08,
    u'left_arm_2_joint': -0.13,
    u'left_arm_3_joint': -1.35,
    u'left_arm_4_joint': 0.3,
    u'left_arm_5_joint': 0.7,
    u'left_arm_6_joint': -0.01,
    u'right_arm_0_joint': 0.68,
    u'right_arm_1_joint': -1.08,
    u'right_arm_2_joint': 0.13,
    u'right_arm_3_joint': 1.35,
    u'right_arm_4_joint': -0.3,
    u'right_arm_5_joint': -0.7,
    u'right_arm_6_joint': 0.01,
}

folder_name = u'tmp_data/'


@pytest.fixture(scope=u'module')
def ros(request):
    try:
        logging.loginfo(u'deleting tmp test folder')
        # shutil.rmtree(folder_name)
    except Exception:
        pass

        logging.loginfo(u'init ros')
    rospy.init_node(u'tests')
    tf_init(60)
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    rospy.set_param('/joint_trajectory_splitter/state_topics',
                    [
                        '/whole_body_controller/base/state',
                        '/whole_body_controller/torso/state',
                        '/whole_body_controller/neck/state',
                        '/whole_body_controller/left_arm/state',
                        '/whole_body_controller/right_arm/state',
                    ])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    [
                        '/whole_body_controller/base/follow_joint_trajectory',
                        '/whole_body_controller/torso/follow_joint_trajectory',
                        '/whole_body_controller/neck/follow_joint_trajectory',
                        '/whole_body_controller/left_arm/follow_joint_trajectory',
                        '/whole_body_controller/right_arm/follow_joint_trajectory',
                    ])
    node = roslaunch.core.Node('giskardpy', 'joint_trajectory_splitter.py', name='joint_trajectory_splitter')
    joint_trajectory_splitter = launch.launch(node)

    def kill_ros():
        joint_trajectory_splitter.stop()
        rospy.delete_param('/joint_trajectory_splitter/state_topics')
        rospy.delete_param('/joint_trajectory_splitter/client_topics')
        logging.loginfo(u'shutdown ros')
        rospy.signal_shutdown(u'die')
        try:
            logging.loginfo(u'deleting tmp test folder')
            # shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_ros)


@pytest.fixture(scope=u'module')
def giskard(request, ros):
    c = Boxy()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: Donbot
    """
    logging.loginfo(u'resetting giskard')
    giskard.clear_world()
    giskard.reset_base()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type giskard: Donbot
    """
    resetted_giskard.set_joint_goal(default_js)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_goal()
    return resetted_giskard


@pytest.fixture()
def better_pose(resetted_giskard):
    """
    :type pocky_pose_setup: Donbot
    :rtype: Donbot
    """
    resetted_giskard.set_joint_goal(better_js)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_goal()
    return resetted_giskard


@pytest.fixture()
def fake_table_setup(zero_pose):
    """
    :type zero_pose: Donbot
    :rtype: Donbot
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
def kitchen_setup(better_pose):
    # better_pose.allow_all_collisions()
    # better_pose.send_and_check_joint_goal(gaya_pose)
    object_name = u'kitchen'
    better_pose.add_urdf(object_name, rospy.get_param(u'kitchen_description'),
                              tf.lookup_pose(u'map', u'iai_kitchen/world'), u'/kitchen/joint_states')
    js = {k: 0.0 for k in better_pose.get_world().get_object(object_name).get_movable_joints()}
    better_pose.set_kitchen_js(js)
    return better_pose



class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(better_js)


class TestConstraints(object):
    def test_pointing(self, better_pose):
        tip = u'head_mount_kinect2_rgb_optical_frame'
        goal_point = lookup_point(u'map', better_pose.r_tip)
        better_pose.wrapper.pointing(tip, goal_point)
        better_pose.send_and_check_goal()

        current_x = Vector3Stamped()
        current_x.header.frame_id = tip
        current_x.vector.z = 1

        expected_x = transform_point(tip, goal_point)
        np.testing.assert_almost_equal(expected_x.point.y, 0, 2)
        np.testing.assert_almost_equal(expected_x.point.x, 0, 2)

        goal_point = lookup_point(u'map', better_pose.r_tip)
        better_pose.wrapper.pointing(tip, goal_point, root=better_pose.r_tip)

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

        better_pose.set_and_check_cart_goal(r_goal, better_pose.r_tip, u'base_footprint')

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
        # TODO: Verify that tip on boxy is actually called like that
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.y = 1

        kitchen_setup.add_json_goal(u'GraspBar',
                                    root=kitchen_setup.default_root,
                                    tip=kitchen_setup.l_tip,
                                    tip_grasp_axis=tip_grasp_axis,
                                    bar_center=bar_center,
                                    bar_axis=bar_axis,
                                    bar_length=0.4)  # TODO: check for real length
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1

        kitchen_setup.align_planes(kitchen_setup.l_tip,
                                   x_gripper,
                                   root_normal=x_goal)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()

        # From here on new (= OpenDrawer)
        kitchen_setup.add_json_goal(u'Open',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.48})
        kitchen_setup.send_and_check_goal()

        kitchen_setup.add_json_goal(u'OpenDrawer',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name,
                                    distance_goal=0.3)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.48})
        kitchen_setup.send_and_check_goal()

        kitchen_setup.add_json_goal(u'Close',
                                    tip=kitchen_setup.l_tip,
                                    object_name=u'kitchen',
                                    handle_link=handle_name)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.send_and_check_goal()
        kitchen_setup.set_kitchen_js({u'sink_area_left_middle_drawer_main_joint': 0.0})
        kitchen_setup.send_and_check_goal()

        pass
