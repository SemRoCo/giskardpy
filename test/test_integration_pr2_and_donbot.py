import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from giskardpy.utils import logging
import giskardpy.utils.tfwrapper as tf
from utils_for_tests import PR2AndDonbot

folder_name = 'tmp_data/'

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


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR2AndDonbot()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: PR2AndDonbot
    """
    logging.loginfo(u'resetting giskard')
    giskard.open_l_gripper(giskard.pr2)
    giskard.open_r_gripper(giskard.pr2)
    giskard.open_gripper(giskard.donbot)
    giskard.clear_world()
    for robot_name in giskard.collision_scene.robot_names:
        giskard.reset_base(robot_name)
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position = Point(0, 1, 0)
    p.pose.orientation = Quaternion(0, 0, 0, 1)
    giskard.move_base(p, giskard.pr2)
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type resetted_giskard: PR2AndDonbot
    """
    resetted_giskard.allow_all_collisions()
    for robot_name in resetted_giskard.collision_scene.robot_names:
        resetted_giskard.set_joint_goal(resetted_giskard.default_poses[robot_name], group_name=robot_name)
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def kitchen_setup(zero_pose):
    """
    :type better_pose: GiskardTestWrapper
    :return: GiskardTestWrapper
    """
    object_name = 'kitchen'
    zero_pose.add_urdf(name=object_name,
                       urdf=rospy.get_param('kitchen_description'),
                       pose=tf.lookup_pose('map', 'iai_kitchen/world'),
                       js_topic='/kitchen/joint_states',
                       set_js_topic='/kitchen/cram_joint_states')
    js = {str(k): 0.0 for k in zero_pose.world.groups[object_name].movable_joint_names}
    zero_pose.set_kitchen_js(js)
    return zero_pose


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: PR2AndDonbot
        """
        zero_pose.allow_self_collision(zero_pose.pr2)
        zero_pose.allow_self_collision(zero_pose.donbot)
        zero_pose.set_joint_goal(floor_detection_js, group_name=zero_pose.donbot)
        zero_pose.set_joint_goal(pocky_pose, group_name=zero_pose.pr2)
        zero_pose.plan_and_execute()


class TestCollisionAvoidance(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: PR2AndDonbot
        """
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(0, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 1, 0)
        zero_pose.move_base(p, zero_pose.donbot)

        zero_pose.avoid_all_collisions()
        zero_pose.set_joint_goal(floor_detection_js, group_name=zero_pose.donbot)
        zero_pose.set_joint_goal(pocky_pose, group_name=zero_pose.pr2)
        zero_pose.plan_and_execute()

    def test_joint_movement2(self, kitchen_setup):
        """
        :type zero_pose: PR2AndDonbot
        """
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position = Point(0, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 1, 0)
        kitchen_setup.move_base(p, kitchen_setup.donbot)

        kitchen_setup.avoid_all_collisions()
        kitchen_setup.set_joint_goal(floor_detection_js, group_name=kitchen_setup.donbot)
        kitchen_setup.set_joint_goal(pocky_pose, group_name=kitchen_setup.pr2)
        kitchen_setup.plan_and_execute()
