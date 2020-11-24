import numpy as np
import pytest
import roslaunch
import rospy
from betterpybullet import Quaternion
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskardpy import logging
from giskardpy.tfwrapper import lookup_transform, init as tf_init
from utils_for_tests import Donbot, KMR_IIWA

# TODO roslaunch iai_donbot_sim ros_control_sim.launch


default_js = {
    u'iiwa_joint_1': 0.0,
    u'iiwa_joint_2': 0.0,
    u'iiwa_joint_3': 0.0,
    u'iiwa_joint_4': 0.0,
    u'iiwa_joint_5': 0.0,
    u'iiwa_joint_6': 0.0,
    u'iiwa_joint_7': 0.0,
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
                    ['/whole_body_controller/base/state',
                     '/whole_body_controller/body/state',
                     '/refills_finger/state'])
    rospy.set_param('/joint_trajectory_splitter/client_topics',
                    ['/whole_body_controller/base/follow_joint_trajectory',
                     '/whole_body_controller/body/follow_joint_trajectory',
                     '/whole_body_controller/refills_finger/follow_joint_trajectory'])
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
    c = KMR_IIWA()
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
    resetted_giskard.allow_all_collisions()
    resetted_giskard.send_and_check_joint_goal(default_js)
    return resetted_giskard


#
# @pytest.fixture()
# def better_pose(resetted_giskard):
#     """
#     :type pocky_pose_setup: Donbot
#     :rtype: Donbot
#     """
#     resetted_giskard.set_joint_goal(better_js)
#     resetted_giskard.allow_all_collisions()
#     resetted_giskard.send_and_check_goal()
#     return resetted_giskard
#
#
# @pytest.fixture()
# def self_collision_pose(resetted_giskard):
#     """
#     :type pocky_pose_setup: Donbot
#     :rtype: Donbot
#     """
#     resetted_giskard.set_joint_goal(self_collision_js)
#     resetted_giskard.allow_all_collisions()
#     resetted_giskard.send_and_check_goal()
#     return resetted_giskard


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
def shelf_setup(better_pose):
    """
    :type better_pose: Donbot
    :rtype: Donbot
    """
    layer1 = u'layer1'
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 0
    p.pose.position.y = -1.25
    p.pose.position.z = 1
    p.pose.orientation.w = 1
    better_pose.add_box(layer1, size=[1, 0.5, 0.02], pose=p)

    layer2 = u'layer2'
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 0
    p.pose.position.y = -1.25
    p.pose.position.z = 1.3
    p.pose.orientation.w = 1
    better_pose.add_box(layer2, size=[1, 0.5, 0.02], pose=p)

    back = u'back'
    p = PoseStamped()
    p.header.frame_id = u'map'
    p.pose.position.x = 0
    p.pose.position.y = -1.5
    p.pose.position.z = 1
    p.pose.orientation.w = 1
    better_pose.add_box(back, size=[1, 0.05, 2], pose=p)
    return better_pose


@pytest.fixture()
def kitchen_setup(zero_pose):
    object_name = u'kitchen'
    zero_pose.add_urdf(object_name, rospy.get_param(u'kitchen_description'), u'/kitchen/joint_states',
                       lookup_transform(u'map', u'iai_kitchen/world'))
    return zero_pose


class TestCartGoals(object):
    def test_cart_goal_1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {
            u'iiwa_joint_1': 0.0,
            u'iiwa_joint_2': -1.66,
            u'iiwa_joint_3': 0.0,
            u'iiwa_joint_4': 0.55,
            u'iiwa_joint_5': 0.0,
            u'iiwa_joint_6': -1.39,
            u'iiwa_joint_7': 0.0,
        }
        zero_pose.send_and_check_joint_goal(js)
        for x in np.arange(0.1, 0.301, 0.05):
            for y in np.arange(-0.15, 0.151, 0.05):
                p = PoseStamped()
                p.header.stamp = rospy.get_rostime()
                p.header.frame_id = 'base_footprint'
                p.pose.position = Point(x, y, 0.75)
                p.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
                zero_pose.set_and_check_cart_goal(p, zero_pose.gripper_tip, 'base_footprint')

    def test_cart_goal_2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {'iiwa_joint_1': 0.0,
              'iiwa_joint_2': -0.54,
              'iiwa_joint_3': 4.35716397646e-18,
              'iiwa_joint_4': -0.46,
              'iiwa_joint_5': -1.57,
              'iiwa_joint_6': -1.7,
              'iiwa_joint_7': 0.9,
              }
        zero_pose.send_and_check_joint_goal(js)
        for z in np.arange(1., 0.4, -0.1):
            p = PoseStamped()
            p.header.stamp = rospy.get_rostime()
            p.header.frame_id = 'map'
            p.pose.position = Point(-1, 0, z)
            p.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                     [0, 0, -1, 0],
                                                                     [0, -1, 0, 0],
                                                                     [0, 0, 0, 1]]))
            zero_pose.set_and_check_cart_goal(p, zero_pose.gripper_tip, 'odom')

    def test_wiggle1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {
            u'iiwa_joint_1': 0.0,
            u'iiwa_joint_2': 0.514796535012,
            u'iiwa_joint_3': 0.0,
            u'iiwa_joint_4': -1.01832869706,
            u'iiwa_joint_5': 0.0,
            u'iiwa_joint_6': 1.246874304,
            u'iiwa_joint_7': 0.0,
        }
        zero_pose.send_and_check_joint_goal(js)
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.gripper_tip
        p.pose.position = Point(0, 0, 0)
        p.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 1, 0]))
        zero_pose.set_and_check_cart_goal(p, zero_pose.gripper_tip, 'odom')

    def test_place_object(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        object_height = 0.145
        camera_link = u'camera_link'
        refills_finger = u'refills_finger'

        # pre_grasp_left = {
        #     'iiwa_joint_1': -0.05,
        #     'iiwa_joint_2': -0.851187914658,
        #     'iiwa_joint_3': 0.0399109025867,
        #     'iiwa_joint_4': -2.02477636846,
        #     'iiwa_joint_5': -1.74610553316,
        #     'iiwa_joint_6': -1.49375643375,
        #     'iiwa_joint_7': -0.395375673321,
        # }
        pre_grasp_left = {
            'iiwa_joint_1': 0.8,
            'iiwa_joint_2': -1.26293805785,
            'iiwa_joint_3': -0.434284463224,
            'iiwa_joint_4': 1.74242120759,
            'iiwa_joint_5': 2.00150718678,
            'iiwa_joint_6': -1.67087526211,
            'iiwa_joint_7': -0.484213467998,
        }

        zero_pose.send_and_check_joint_goal(pre_grasp_left)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = camera_link
        grasp_pose.pose.position.z = +0.2
        grasp_pose.pose.orientation.w = 1
        zero_pose.set_and_check_cart_goal(grasp_pose,
                                          tip_link=camera_link,
                                          root_link=u'base_link')

        box_pose = PoseStamped()
        box_pose.header.frame_id = refills_finger
        box_pose.pose.orientation.w = 1
        zero_pose.attach_box(name='box',
                             size=[0.05, object_height, 0.115],
                             frame_id=refills_finger,
                             pose=box_pose)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = camera_link
        grasp_pose.pose.position.z = -0.2
        grasp_pose.pose.orientation.w = 1
        zero_pose.set_and_check_cart_goal(grasp_pose,
                                          tip_link=camera_link,
                                          root_link=u'base_link')


        tip_normal = Vector3Stamped()
        tip_normal.header.frame_id = refills_finger
        tip_normal.vector.y = 1
        root_normal = Vector3Stamped()
        root_normal.header.frame_id = u'map'
        root_normal.vector.z = 1
        zero_pose.align_planes(tip_link=refills_finger,
                               tip_normal=tip_normal,
                               root_link=u'base_link',
                               root_normal=root_normal)
        pre_place = {
            u'iiwa_joint_1': 0,
            u'iiwa_joint_2': -1.28,
            u'iiwa_joint_3': 0,
            u'iiwa_joint_4': 1.29,
            u'iiwa_joint_5': 0,
            u'iiwa_joint_6': -0.74,
            u'iiwa_joint_7': -1.57,
        }
        zero_pose.send_and_check_joint_goal(pre_place)




        box_goal = PoseStamped()
        box_goal.header.frame_id = u'angle_adapter_base'
        box_goal.pose.position.z = object_height / 2 + 0.01
        box_goal.pose.position.y = 0.35
        box_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                        [0, 0, -1, 0],
                                                                        [0, 1, 0, 0],
                                                                        [0, 0, 0, 1]]))

        zero_pose.set_and_check_cart_goal(goal_pose=box_goal,
                                          tip_link=u'refills_finger',
                                          root_link=u'base_link')
