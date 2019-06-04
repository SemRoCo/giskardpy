import numpy as np
from copy import deepcopy

import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped
from giskard_msgs.msg import MoveActionGoal, MoveResult, MoveGoal, CollisionEntry
from tf.transformations import quaternion_from_matrix

from giskardpy import logging
from giskardpy.tfwrapper import lookup_transform, init as tf_init
from utils_for_tests import Donbot

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

better_js = {
    u'ur5_shoulder_pan_joint': -np.pi / 2,
    u'ur5_shoulder_lift_joint': -2.44177755311,
    u'ur5_elbow_joint': 2.15026930371,
    u'ur5_wrist_1_joint': 0.291547812391,
    u'ur5_wrist_2_joint': np.pi / 2,
    u'ur5_wrist_3_joint': np.pi / 2
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

    def kill_ros():
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
    c = Donbot()
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
    resetted_giskard.set_joint_goal(default_pose)
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


class TestJointGoals(object):
    def test_joint_movement1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(floor_detection_pose)

    def test_joint_movement2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        js = {
            u'ur5_shoulder_pan_joint': -1.5438225905,
            u'ur5_shoulder_lift_joint': -1.20804578463,
            u'ur5_elbow_joint': -2.21223670641,
            u'ur5_wrist_1_joint': -1.5827181975,
            u'ur5_wrist_2_joint': -4.71748859087,
            u'ur5_wrist_3_joint': -1.57543737093,
        }
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(js)

        js2 = {
            u'ur5_shoulder_pan_joint': -np.pi / 2,
            u'ur5_shoulder_lift_joint': -np.pi / 2,
            u'ur5_elbow_joint': -2.3,
            u'ur5_wrist_1_joint': -np.pi / 2,
            u'ur5_wrist_2_joint': 0,
            u'ur5_wrist_3_joint': -np.pi / 2,
        }
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(js2)

    def test_partial_joint_state_goal1(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        zero_pose.allow_self_collision()
        js = dict(floor_detection_pose.items()[:3])
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
        zero_pose.set_and_check_cart_goal(p, zero_pose.gripper_tip, zero_pose.default_root)

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
        zero_pose.set_and_check_cart_goal(goal_pose, zero_pose.camera_tip, zero_pose.default_root)

    def test_endless_wiggling2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = u'base_link'
        goal_pose.pose.position.x = 0.212
        goal_pose.pose.position.y = -0.314
        goal_pose.pose.position.z = 0.873
        goal_pose.pose.orientation.x = 0.004
        goal_pose.pose.orientation.y = 0.02
        goal_pose.pose.orientation.z = 0.435
        goal_pose.pose.orientation.w = .9

        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(goal_pose, zero_pose.gripper_tip, zero_pose.default_root)


class TestCollisionAvoidanceGoals(object):
    def test_attach_existing_box_non_fixed(self, better_pose):
        """
        :type zero_pose: Donbot
        """
        pocky = u'box'
        # hack_link_name = u'hack_link'
        # box_object = URDFObject.from_world_body(make_world_body_box(box_name, 0.05, 0.03, 0.2))
        # link_object = URDFObject.from_world_body(make_world_body_box(hack_link_name, 0.01, 0.01, 0.01))

        p = PoseStamped()
        p.header.frame_id = u'refills_finger'
        p.pose.position.y = -0.075
        p.pose.orientation = Quaternion(0, 0, 0, 1)

        better_pose.add_box(pocky, [0.05, 0.2, 0.03], p)
        better_pose.attach_existing(pocky, frame_id=u'refills_finger')

        tip_normal = Vector3Stamped()
        tip_normal.header.frame_id = pocky
        tip_normal.vector.y = 1

        root_normal = Vector3Stamped()
        root_normal.header.frame_id = u'base_footprint'
        root_normal.vector.z = 1
        better_pose.align_planes(pocky, tip_normal, u'base_footprint', root_normal)
        # better_pose.add_json_goal(u'GravityJoint', joint_name=u'refills_finger', object_name=pocky)

        pocky_goal = PoseStamped()
        pocky_goal.header.frame_id = pocky
        pocky_goal.pose.position.y = -.5
        pocky_goal.pose.position.x = .3
        pocky_goal.pose.position.z = -.2
        pocky_goal.pose.orientation.w = 1
        better_pose.allow_self_collision()
        better_pose.set_translation_goal(pocky_goal, pocky, u'base_footprint')
        better_pose.send_and_check_goal()

    def test_pick_low_place_high(self, better_pose):
        pass

    def test_place_in_shelf(self, shelf_setup):
        """
        :type shelf_setup: Donbot
        """
        box = u'box'
        box_pose = PoseStamped()
        box_pose.header.frame_id = u'map'
        box_pose.pose.orientation.w = 1
        box_pose.pose.position.y = -0.5
        box_pose.pose.position.z = .1
        shelf_setup.add_box(box, [0.05, 0.05, 0.2], box_pose)

        grasp_pose = deepcopy(box_pose)
        grasp_pose.pose.position.z += 0.05
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                          [0, 1, 0, 0],
                                                                          [0, 0, -1, 0],
                                                                          [0, 0, 0, 1]]))
        # shelf_setup.allow_self_collision()
        shelf_setup.allow_collision([CollisionEntry.ALL], box, [CollisionEntry.ALL])
        shelf_setup.set_and_check_cart_goal(grasp_pose, shelf_setup.gripper_tip)

        box_goal = PoseStamped()
        box_goal.header.frame_id = u'map'
        box_goal.pose.position.z = 1.12
        box_goal.pose.position.y = -1
        grasp_pose.pose.orientation.w = 1
        shelf_setup.set_and_check_cart_goal(box_goal, box)
        shelf_setup.attach_existing(box, u'refills_finger')

        # p = PoseStamped()
        # p.header.frame_id = u'refills_finger'
        # p.pose.position.y = -0.075
        # p.pose.orientation = Quaternion(0, 0, 0, 1)
        #
        # better_pose.add_box(box, [0.05, 0.2, 0.03], p)
        # better_pose.attach_existing(box, frame_id=u'refills_finger')
        #
        # tip_normal = Vector3Stamped()
        # tip_normal.header.frame_id = box
        # tip_normal.vector.y = 1
        #
        # root_normal = Vector3Stamped()
        # root_normal.header.frame_id = u'base_footprint'
        # root_normal.vector.z = 1
        # better_pose.align_planes(box, tip_normal, u'base_footprint', root_normal)
        # # better_pose.add_json_goal(u'GravityJoint', joint_name=u'refills_finger', object_name=pocky)
        #
        # pocky_goal = PoseStamped()
        # pocky_goal.header.frame_id = box
        # pocky_goal.pose.position.y = -.5
        # pocky_goal.pose.position.x = .3
        # pocky_goal.pose.position.z = -.2
        # pocky_goal.pose.orientation.w = 1
        # better_pose.allow_self_collision()
        # better_pose.set_translation_goal(pocky_goal, box, u'base_footprint')
        # better_pose.send_and_check_goal()

    def test_allow_self_collision2(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        goal_js = {
            u'ur5_shoulder_lift_joint': .5,
        }
        zero_pose.allow_self_collision()
        zero_pose.send_and_check_joint_goal(goal_js)

        arm_goal = PoseStamped()
        arm_goal.header.frame_id = zero_pose.gripper_tip
        arm_goal.pose.position.y = -.1
        arm_goal.pose.orientation.w = 1
        zero_pose.allow_self_collision()
        zero_pose.set_and_check_cart_goal(arm_goal, zero_pose.gripper_tip, zero_pose.default_root)

    def test_avoid_self_collision(self, zero_pose):
        """
        :type zero_pose: Donbot
        """
        goal_js = {
            u'ur5_shoulder_lift_joint': .5,
        }
        zero_pose.wrapper.set_self_collision_distance(0.025)
        zero_pose.send_and_check_joint_goal(goal_js)

        arm_goal = PoseStamped()
        arm_goal.header.frame_id = zero_pose.gripper_tip
        arm_goal.pose.position.y = -.1
        arm_goal.pose.orientation.w = 1
        zero_pose.wrapper.set_self_collision_distance(0.025)
        zero_pose.set_and_check_cart_goal(arm_goal, zero_pose.gripper_tip, zero_pose.default_root)
