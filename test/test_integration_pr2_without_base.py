import shutil

import pytest
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped

from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import init as tf_init, lookup_pose
from utils_for_tests import TestPR2

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
                'head_tilt_joint': 0,
                }

gaya_pose = {'r_shoulder_pan_joint': -1.7125,
             'r_shoulder_lift_joint': -0.25672,
             'r_upper_arm_roll_joint': -1.46335,
             'r_elbow_flex_joint': -2.12216,
             'r_forearm_roll_joint': 1.76632,
             'r_wrist_flex_joint': -0.10001,
             'r_wrist_roll_joint': 0.05106,
             'l_shoulder_pan_joint': 1.9652,
             'l_shoulder_lift_joint': - 0.26499,
             'l_upper_arm_roll_joint': 1.3837,
             'l_elbow_flex_joint': - 2.1224,
             'l_forearm_roll_joint': 16.99,
             'l_wrist_flex_joint': - 0.10001,
             'l_wrist_roll_joint': 0,
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


@pytest.fixture(scope='module')
def ros(request):
    try:
        logging.loginfo('deleting tmp test folder')
        shutil.rmtree(folder_name)
    except Exception:
        pass

    logging.loginfo('init ros')
    rospy.init_node('tests')
    tf_init(60)

    def kill_ros():
        logging.loginfo('shutdown ros')
        rospy.signal_shutdown('die')
        try:
            logging.loginfo('deleting tmp test folder')
            shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_ros)


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = TestPR2()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def resetted_giskard(giskard):
    """
    :type giskard: TestPR2
    """
    logging.loginfo('resetting giskard')
    giskard.clear_world()
    return giskard


@pytest.fixture()
def zero_pose(resetted_giskard):
    """
    :type giskard: TestPR2
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
    :type pocky_pose_setup: TestPR2
    :rtype: TestPR2
    """
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.5
    p.pose.orientation.w = 1
    pocky_pose_setup.add_box(size=[1, 1, 1], pose=p)
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(zero_pose):
    """
    :type zero_pose: TestPR2
    :rtype: TestPR2
    """
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 0.9
    p.pose.position.y = 0
    p.pose.position.z = 0.2
    p.pose.orientation.w = 1
    zero_pose.add_box(pose=p)
    return zero_pose


@pytest.fixture()
def kitchen_setup(resetted_giskard):
    """
    :type resetted_giskard: GiskardTestWrapper
    :return:
    """
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_joint_goal(gaya_pose)
    object_name = 'kitchen'
    resetted_giskard.add_urdf(object_name, rospy.get_param('kitchen_description'),
                              lookup_pose('map', 'iai_kitchen/world'), '/kitchen/joint_states')
    js = {k: 0.0 for k in resetted_giskard.get_world().get_object(object_name).get_movable_joints()}
    resetted_giskard.set_kitchen_js(js)
    return resetted_giskard


class TestJointGoals(object):

    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: TestPR2
        """
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(pocky_pose)

    # TODO without base movement there is probably a bug, when an avoid collision entry is added with base_link as link_a