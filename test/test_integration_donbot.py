from typing import Optional

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from tf.transformations import quaternion_about_axis, quaternion_matrix, rotation_from_matrix

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import GiskardError
from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.iai_robots.donbot import WorldWithBoxyBaseConfig, DonbotCollisionAvoidanceConfig, DonbotStandaloneInterfaceConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.god_map import god_map
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import GiskardTestWrapper

# TODO roslaunch iai_donbot_sim ros_control_sim.launch


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


class DonbotTestWrapper(GiskardTestWrapper):
    default_pose = {
        'ur5_elbow_joint': 0.0,
        'ur5_shoulder_lift_joint': 0.0,
        'ur5_shoulder_pan_joint': 0.0,
        'ur5_wrist_1_joint': 0.0,
        'ur5_wrist_2_joint': 0.0,
        'ur5_wrist_3_joint': 0.0
    }

    better_pose = {
        'ur5_shoulder_pan_joint': -np.pi / 2,
        'ur5_shoulder_lift_joint': -2.44177755311,
        'ur5_elbow_joint': 2.15026930371,
        'ur5_wrist_1_joint': 0.291547812391,
        'ur5_wrist_2_joint': np.pi / 2,
        'ur5_wrist_3_joint': np.pi / 2
    }

    def __init__(self):
        # from iai_wsg_50_msgs.msg import PositionCmd
        self.camera_tip = 'camera_link'
        self.gripper_tip = 'gripper_tool_frame'
        # self.gripper_pub = rospy.Publisher('/wsg_50_driver/goal_position', PositionCmd, queue_size=10)
        # self.mujoco_reset = rospy.ServiceProxy('donbot/reset', Trigger)
        giskard = Giskard(world_config=WorldWithBoxyBaseConfig(),
                          collision_avoidance_config=DonbotCollisionAvoidanceConfig(),
                          robot_interface_config=DonbotStandaloneInterfaceConfig(),
                          behavior_tree_config=StandAloneBTConfig(),
                          qp_controller_config=QPControllerConfig())
        super().__init__(giskard)

    def open_gripper(self):
        self.set_gripper(0.109)

    def close_gripper(self):
        self.set_gripper(0)

    def set_gripper(self, width: float, gripper_joint: str = 'gripper_joint'):
        width = max(0.0065, min(0.109, width))
        js = {gripper_joint: width}
        self.set_joint_goal(js)
        self.allow_all_collisions()
        self.plan_and_execute()

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        goal_pose = tf.transform_pose(self.default_root, goal_pose)
        js = {'odom_x_joint': goal_pose.pose.position.x,
              'odom_y_joint': goal_pose.pose.position.y,
              'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
                                                                      goal_pose.pose.orientation.y,
                                                                      goal_pose.pose.orientation.z,
                                                                      goal_pose.pose.orientation.w]))[0]}
        self.allow_all_collisions()
        self.set_seed_configuration(js)
        self.plan_and_execute()

    def move_base(self, goal_pose):
        self.set_cart_goal(goal_pose, tip_link='base_footprint', root_link='odom')
        self.plan_and_execute()

    def set_localization(self, map_T_odom: PoseStamped):
        self.teleport_base(map_T_odom)

    def reset(self):
        self.open_gripper()
        self.reset_base()
        self.clear_world()
        # self.register_group('gripper',
        #                     root_link_group_name=self.robot_name,
        #                     root_link_name='l_wrist_roll_link')


@pytest.fixture(scope='module')
def giskard(request, ros) -> DonbotTestWrapper:
    launch_launchfile('package://iai_donbot_description/launch/upload.launch')
    c = DonbotTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def self_collision_pose(resetted_giskard: DonbotTestWrapper) -> DonbotTestWrapper:
    resetted_giskard.set_joint_goal(self_collision_js)
    resetted_giskard.allow_all_collisions()
    resetted_giskard.plan_and_execute()
    return resetted_giskard


@pytest.fixture()
def fake_table_setup(zero_pose: DonbotTestWrapper) -> DonbotTestWrapper:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 0.9
    p.pose.position.y = 0
    p.pose.position.z = 0.2
    p.pose.orientation.w = 1
    zero_pose.add_box_to_world(name='box', size=(1, 1, 1), pose=p)
    return zero_pose


class TestJointGoals:
    def test_joint_movement1(self, zero_pose: DonbotTestWrapper):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(floor_detection_js)
        zero_pose.plan_and_execute()

    def test_joint_movement_gaya(self, zero_pose: DonbotTestWrapper):
        js1 = {"ur5_shoulder_pan_joint": 1.475476861000061,
               "ur5_shoulder_lift_joint": -1.664506737385885,
               "ur5_elbow_joint": -2.0976365248309534,
               "ur5_wrist_1_joint": 0.6524184942245483,
               "ur5_wrist_2_joint": 1.7044463157653809,
               "ur5_wrist_3_joint": -1.5686963240252894}
        js2 = {
            "ur5_shoulder_pan_joint": 4.112661838531494,
            "ur5_shoulder_lift_joint": - 1.6648781935321253,
            "ur5_elbow_joint": - 1.4145501295672815,
            "ur5_wrist_1_joint": - 1.608563248311178,
            "ur5_wrist_2_joint": 1.5707963267948966,
            "ur5_wrist_3_joint": - 1.6503928343402308
        }
        zero_pose.set_joint_goal(js2)
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(js1)
        zero_pose.plan_and_execute()

    def test_empty_joint_goal(self, zero_pose: DonbotTestWrapper):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal({
            'ur5_shoulder_pan_joint': -0.15841275850404912,
            'ur5_shoulder_lift_joint': -2.2956998983966272,
            'ur5_elbow_joint': 2.240689277648926,
            'ur5_wrist_1_joint': -2.608211342488424,
            'ur5_wrist_2_joint': -2.7356796900378626,
            'ur5_wrist_3_joint': -2.5249870459186,
        })
        zero_pose.set_joint_goal({})
        zero_pose.plan_and_execute(expected_error_code=GiskardError.GOAL_INITIALIZATION_ERROR)

    def test_joint_movement2(self, zero_pose: DonbotTestWrapper):
        js = {
            'ur5_shoulder_pan_joint': -1.5438225905,
            'ur5_shoulder_lift_joint': -1.20804578463,
            'ur5_elbow_joint': -2.21223670641,
            'ur5_wrist_1_joint': -1.5827181975,
            'ur5_wrist_2_joint': -4.71748859087,
            'ur5_wrist_3_joint': -1.57543737093,
        }
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        js2 = {
            'ur5_shoulder_pan_joint': -np.pi / 2,
            'ur5_shoulder_lift_joint': -np.pi / 2,
            'ur5_elbow_joint': -2.3,
            'ur5_wrist_1_joint': -np.pi / 2,
            'ur5_wrist_2_joint': 0,
            'ur5_wrist_3_joint': -np.pi / 2,
        }
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(js2)
        zero_pose.plan_and_execute()

    def test_joint_movement3(self, zero_pose: DonbotTestWrapper):
        js = {
            'odom_x_joint': 1,
            'odom_y_joint': 1,
            'odom_z_joint': 1,
            'ur5_shoulder_pan_joint': -1.5438225905,
            'ur5_shoulder_lift_joint': -1.20804578463,
            'ur5_elbow_joint': -2.21223670641,
            'ur5_wrist_1_joint': -1.5827181975,
            'ur5_wrist_2_joint': -4.71748859087,
            'ur5_wrist_3_joint': -1.57543737093,
        }
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

    def test_partial_joint_state_goal1(self, zero_pose: DonbotTestWrapper):
        zero_pose.allow_self_collision()
        js = dict(list(floor_detection_js.items())[:3])
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()


class TestConstraints:
    def test_pointing(self, better_pose: DonbotTestWrapper):
        tip = 'rs_camera_link'
        goal_point = god_map.world.compute_fk_point('map', 'base_footprint')
        z = Vector3Stamped()
        z.header.frame_id = 'rs_camera_link'
        z.vector.z = 1
        better_pose.set_pointing_goal(goal_point=goal_point, tip_link=tip, pointing_axis=z,
                                      root_link=better_pose.default_root)
        better_pose.plan_and_execute()

        goal_point = god_map.world.compute_fk_point('map', tip)
        better_pose.set_pointing_goal(goal_point=goal_point, tip_link=tip, pointing_axis=z,
                                      root_link=tip)
        better_pose.plan_and_execute()

    def test_open_fridge(self, kitchen_setup: DonbotTestWrapper):
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position = Point(0.3, -0.5, 0)
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 2, [0, 0, 1]))
        kitchen_setup.teleport_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.gripper_tip
        tip_grasp_axis.vector.y = -1

        kitchen_setup.set_grasp_bar_goal(root_link=kitchen_setup.default_root,
                                         tip_link=kitchen_setup.gripper_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.gripper_tip
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.gripper_tip,
                                            tip_normal=x_gripper,
                                            goal_normal=x_goal,
                                            root_link=kitchen_setup.default_root)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.gripper_tip,
                                              environment_link=handle_name,
                                              goal_joint_state=1.5)
        kitchen_setup.set_avoid_joint_limits_goal(percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_env_state({'iai_fridge_door_joint': 1.5})

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.gripper_tip,
                                              environment_link=handle_name,
                                              goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.set_avoid_joint_limits_goal(percentage=40)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_env_state({'iai_fridge_door_joint': 0})

        # kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()


class TestCartGoals:
    def test_cart_goal_1eef(self, zero_pose: DonbotTestWrapper):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = zero_pose.gripper_tip
        p.pose.position = Point(0, -0.1, 0)
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [1, 0, 0]))
        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(p, zero_pose.gripper_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_cart_goal2(self, zero_pose: DonbotTestWrapper):
        js = {
            'ur5_shoulder_pan_joint': 3.141554832458496,
            'ur5_shoulder_lift_joint': -1.3695076147662562,
            'ur5_elbow_joint': 0.5105495452880859,
            'ur5_wrist_1_joint': -0.7200177351581019,
            'ur5_wrist_2_joint': -0.22007495561708623,
            'ur5_wrist_3_joint': 0,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()
        p = PoseStamped()
        p.header.frame_id = 'camera_link'
        p.pose.position = Point(0, 1, 0)
        p.pose.orientation.w = 1
        # zero_pose.allow_self_collision()
        zero_pose.set_straight_cart_goal(goal_pose=p,
                                         tip_link=zero_pose.camera_tip,
                                         root_link='ur5_shoulder_link')
        # zero_pose.set_translation_goal(p, zero_pose.camera_tip, 'ur5_shoulder_link', weight=WEIGHT_BELOW_CA)
        # zero_pose.set_rotation_goal(p, zero_pose.camera_tip, 'ur5_shoulder_link', weight=WEIGHT_ABOVE_CA)
        zero_pose.plan_and_execute()

    def test_endless_wiggling1(self, zero_pose: DonbotTestWrapper):
        start_pose = {
            'ur5_elbow_joint': 2.14547738764,
            'ur5_shoulder_lift_joint': -1.177280122,
            'ur5_shoulder_pan_joint': -1.8550731481,
            'ur5_wrist_1_joint': -3.70994178242,
            'ur5_wrist_2_joint': -1.30010203311,
            'ur5_wrist_3_joint': 1.45079807832,
        }

        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(start_pose)
        zero_pose.plan_and_execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'base_link'
        goal_pose.pose.position.x = -0.512
        goal_pose.pose.position.y = -1.036126
        goal_pose.pose.position.z = 0.605
        goal_pose.pose.orientation.x = -0.007
        goal_pose.pose.orientation.y = -0.684
        goal_pose.pose.orientation.z = 0.729
        goal_pose.pose.orientation.w = 0

        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(goal_pose, zero_pose.camera_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    def test_endless_wiggling2(self, zero_pose: DonbotTestWrapper):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'base_link'
        goal_pose.pose.position.x = 0.212
        goal_pose.pose.position.y = -0.314
        goal_pose.pose.position.z = 0.873
        goal_pose.pose.orientation.x = 0.004
        goal_pose.pose.orientation.y = 0.02
        goal_pose.pose.orientation.z = 0.435
        goal_pose.pose.orientation.w = .9

        zero_pose.allow_self_collision()
        zero_pose.set_cart_goal(goal_pose, zero_pose.gripper_tip, zero_pose.default_root)
        zero_pose.plan_and_execute()

    # def test_elbow_singularity(self, better_pose: DonbotTestWrapper):
    #     #FIXME fix singularities
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = better_pose.gripper_tip
    #     hand_goal.pose.position.z = 1
    #     hand_goal.pose.orientation.w = 1
    #     better_pose.set_cart_goal(goal_pose=hand_goal,
    #                               tip_link=better_pose.gripper_tip,
    #                               root_link='base_footprint',
    #                               add_monitor=False)
    #     better_pose.plan_and_execute()
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = better_pose.gripper_tip
    #     hand_goal.pose.position.z = -0.2
    #     hand_goal.pose.orientation.w = 1
    #     better_pose.set_cart_goal(goal_pose=hand_goal,
    #                               tip_link=better_pose.gripper_tip,
    #                               root_link='base_footprint')
    #     better_pose.plan_and_execute()
    #
    # def test_elbow_singularity2(self, zero_pose: DonbotTestWrapper):
    #     # FIXME fix singularities
    #     tip = 'ur5_wrist_1_link'
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = tip
    #     hand_goal.pose.position.x = 0.5
    #     hand_goal.pose.orientation.w = 1
    #     zero_pose.set_cart_goal(goal_pose=hand_goal,
    #                             tip_link=tip,
    #                             root_link='base_footprint')
    #     zero_pose.allow_all_collisions()
    #     zero_pose.plan_and_execute()
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = tip
    #     hand_goal.pose.position.x = -0.6
    #     hand_goal.pose.orientation.w = 1
    #     zero_pose.set_cart_goal(goal_pose=hand_goal,
    #                             tip_link=tip,
    #                             root_link='base_footprint',
    #                             weight=WEIGHT_BELOW_CA / 2,
    #                             add_monitor=False)
    #     zero_pose.allow_all_collisions()
    #     zero_pose.plan_and_execute()

    def test_base_driving(self, zero_pose: DonbotTestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        zero_pose.teleport_base(p)
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position.y = 1
        p.pose.orientation.w = 1
        # zero_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        zero_pose.set_cart_goal(goal_pose=p, tip_link='base_footprint', root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    # def test_shoulder_singularity(self, better_pose: DonbotTestWrapper):
    #     hand_goal = PoseStamped()
    #     hand_goal.header.frame_id = 'ur5_base_link'
    #     hand_goal.pose.position.x = 0.05
    #     hand_goal.pose.position.y = 0.2
    #     hand_goal.pose.position.z = 0.4
    #     hand_goal.pose.orientation = Quaternion(*quaternion_from_matrix(
    #         [
    #             [0, -1, 0, 0],
    #             [-1, 0, 0, 0],
    #             [0, 0, -1, 0],
    #             [0, 0, 0, 1],
    #         ]
    #     ))
    #     better_pose.allow_all_collisions()
    #     better_pose.set_cart_goal(hand_goal, 'ur5_wrist_2_link', 'base_footprint', weight=WEIGHT_BELOW_CA)
    #     better_pose.plan_and_execute()
    #     hand_goal.pose.position.y = -0.05
    #     better_pose.allow_all_collisions()
    #     # better_pose.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
    #     better_pose.set_cart_goal(hand_goal, 'ur5_wrist_2_link', 'base_footprint', weight=WEIGHT_BELOW_CA)
    #     better_pose.plan_and_execute()
