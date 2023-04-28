from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PointStamped, Vector3Stamped, QuaternionStamped
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis, quaternion_from_matrix

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult
from giskardpy import identifier
from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.configs.tiago import TiagoMujoco, Tiago_Standalone
from giskardpy.goals.goal import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.model.joints import OneDofJoint
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.utils.utils import publish_pose, launch_launchfile
from utils_for_tests import GiskardTestWrapper, RotationGoalChecker, TranslationGoalChecker


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://iai_tiago_description/launch/upload.launch')
    c = TiagoTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TiagoTestWrapper(GiskardTestWrapper):
    default_pose = {
        'torso_lift_joint': 0,
        'head_1_joint': 0.0,
        'head_2_joint': 0.0,
        'arm_left_1_joint': 0.0,
        'arm_left_2_joint': 0.0,
        'arm_left_3_joint': 0.0,
        'arm_left_4_joint': 0.0,
        'arm_left_5_joint': 0.0,
        'arm_left_6_joint': 0.0,
        'arm_left_7_joint': 0.0,
        'arm_right_1_joint': 0.0,
        'arm_right_2_joint': 0.0,
        'arm_right_3_joint': 0.0,
        'arm_right_4_joint': 0.0,
        'arm_right_5_joint': 0.0,
        'arm_right_6_joint': 0.0,
        'arm_right_7_joint': 0.0,
        'gripper_right_left_finger_joint': 0.001,
        'gripper_right_right_finger_joint': 0.001,
        'gripper_left_left_finger_joint': 0.001,
        'gripper_left_right_finger_joint': 0.001,
    }

    better_pose = {
        'arm_left_1_joint': - 1.0,
        'arm_left_2_joint': 0.0,
        'arm_left_3_joint': 1.5,
        'arm_left_4_joint': 2.2,
        'arm_left_5_joint': - 1.5,
        'arm_left_6_joint': 0.5,
        'arm_left_7_joint': 0.0,
        'arm_right_1_joint': - 1.0,
        'arm_right_2_joint': 0.0,
        'arm_right_3_joint': 1.5,
        'arm_right_4_joint': 2.2,
        'arm_right_5_joint': - 1.5,
        'arm_right_6_joint': 0.5,
        'arm_right_7_joint': 0.0,
        'torso_lift_joint': 0.35,
        'gripper_right_left_finger_joint': 0.045,
        'gripper_right_right_finger_joint': 0.045,
        'gripper_left_left_finger_joint': 0.045,
        'gripper_left_right_finger_joint': 0.045,
    }

    better_pose2 = {
        'arm_left_1_joint': 0.27,
        'arm_left_2_joint': - 1.07,
        'arm_left_3_joint': 1.5,
        'arm_left_4_joint': 2.0,
        'arm_left_5_joint': - 2.0,
        'arm_left_6_joint': 1.2,
        'arm_left_7_joint': 0.5,
        'arm_right_1_joint': 0.27,
        'arm_right_2_joint': - 1.07,
        'arm_right_3_joint': 1.5,
        'arm_right_4_joint': 2.0,
        'arm_right_5_joint': - 2.0,
        'arm_right_6_joint': 1.2,
        'arm_right_7_joint': 0.5,
        'gripper_right_left_finger_joint': 0.001,
        'gripper_right_right_finger_joint': 0.001,
        'gripper_left_left_finger_joint': 0.001,
        'gripper_left_right_finger_joint': 0.001,
    }

    def __init__(self, config=None):
        if config is None:
            config = Tiago_Standalone
        super().__init__(config)

    def move_base(self, goal_pose: PoseStamped, check: bool = True):
        tip_link = PrefixName('base_footprint', self.robot_name)
        root_link = self.default_root
        self.set_json_goal(constraint_type='DiffDriveBaseGoal',
                           tip_link=tip_link,
                           root_link=root_link,
                           goal_pose=goal_pose)
        # self.allow_all_collisions()

        if check:
            goal_position = PointStamped()
            goal_position.header = goal_pose.header
            goal_position.point = goal_pose.pose.position
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group='',
                                                                       tip_link='base_footprint', tip_group=self.robot_name)
            self.add_goal_check(TranslationGoalChecker(self, full_tip_link, full_root_link, goal_position))

            goal_orientation = QuaternionStamped()
            goal_orientation.header = goal_pose.header
            goal_orientation.quaternion = goal_pose.pose.orientation
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group='',
                                                                       tip_link='base_footprint', tip_group=self.robot_name)
            self.add_goal_check(RotationGoalChecker(self, full_tip_link, full_root_link, goal_orientation))
        self.plan_and_execute()

    def open_right_gripper(self, goal: float = 0.45):
        js = {
            'gripper_right_left_finger_joint': goal,
            'gripper_right_right_finger_joint': goal,
            'gripper_left_left_finger_joint': goal,
            'gripper_left_right_finger_joint': goal,
        }
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset(self):
        self.clear_world()
        self.reset_base()

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        if self.is_standalone():
            self.teleport_base(p)
        else:
            self.move_base(p)

    def set_localization(self, map_T_odom: PoseStamped):
        map_T_odom.pose.position.z = 0
        self.teleport_base(map_T_odom)

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.set_seed_odometry(base_pose=goal_pose, group_name=group_name)
        self.allow_all_collisions()
        self.plan_and_execute()


class TestCartGoals:

    def test_drive_topright_bottom_left(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(0.489, -0.598, 0.000)
        goal.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(-0.026, 0.569, 0.000)
        goal.pose.orientation = Quaternion(0, 0, 0.916530200374776, 0.3999654882623912)
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive_forward_forward(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1
        goal.pose.position.y = 1
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 2
        goal.pose.position.y = 0
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()

    def test_drive_rotate(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive_backward_backward(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = -1
        goal.pose.position.y = -1
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = -2
        goal.pose.position.y = 0
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()

    def test_drive_left(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 0.01
        goal.pose.position.y = 0.5
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 8, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive_left_right(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 0.01
        goal.pose.position.y = 0.5
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = -0.01
        goal.pose.position.y = -0.5
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi * 0.2, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(goal)

    def test_drive7(self, apartment_setup: TiagoTestWrapper):
        apartment_setup.set_joint_goal(apartment_setup.better_pose2)
        apartment_setup.plan_and_execute()

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.1
        goal.pose.position.y = 2
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi, [0, 0, 1]))
        apartment_setup.set_joint_goal(apartment_setup.better_pose2)
        apartment_setup.move_base(goal)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.1
        goal.pose.position.y = 3
        goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi, [0, 0, 1]))
        # apartment_setup.set_joint_goal(apartment_setup.better_pose2)
        apartment_setup.move_base(goal)

    def test_drive3(self, apartment_setup: TiagoTestWrapper):
        countertop_frame = 'iai_apartment/island_countertop'

        start_base_pose = PoseStamped()
        start_base_pose.header.frame_id = 'map'
        start_base_pose.pose.position = Point(1.295, 2.294, 0)
        start_base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.990, -0.139)
        # apartment_setup.allow_all_collisions()
        apartment_setup.move_base(start_base_pose)

        countertip_P_goal = PointStamped()
        countertip_P_goal.header.frame_id = countertop_frame
        countertip_P_goal.point.x = 1.3
        countertip_P_goal.point.y = -0.3
        map_P_goal = tf.msg_to_homogeneous_matrix(apartment_setup.transform_msg('map', countertip_P_goal))
        map_P_goal[-2] = 0

        # map_P_base_footprint = tf.msg_to_homogeneous_matrix(tf.lookup_point('map', 'base_footprint'))
        # # map_P_goal = np.array([1.3, -0.3, 0, 1])
        # x = map_P_goal - map_P_base_footprint
        # x = x[:3]
        # x /= np.linalg.norm(x)
        # z = np.array([0,0,1])
        # y = np.cross(z, x)
        # y /= np.linalg.norm(y)
        # map_R_goal = np.vstack([x,y,z]).T
        # map_R_goal = np.vstack([np.vstack([map_R_goal.T, [0,0,0]]).T, [0,0,0,1]])
        #
        # base_pose = tf.lookup_pose('map', 'base_footprint')
        # base_pose.pose.orientation = Quaternion(*quaternion_from_matrix(map_R_goal))
        # # base_pose = tf.transform_pose(apartment_setup.default_root, base_pose)
        # # base_pose.pose.position.z = 0
        # apartment_setup.allow_all_collisions()
        # apartment_setup.move_base(base_pose)

        base_pose = PoseStamped()
        base_pose.header.frame_id = countertop_frame
        base_pose.pose.position.x = 1.3
        base_pose.pose.position.y = -0.3
        base_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        base_pose = tf.transform_pose(apartment_setup.default_root, base_pose)
        base_pose.pose.position.z = 0
        # apartment_setup.allow_all_collisions()
        apartment_setup.move_base(base_pose)


class TestCollisionAvoidance:
    def test_avoid_all(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.avoid_collision(0.1, group1=zero_pose.robot_name)
        zero_pose.plan_and_execute()

    def test_attach_object(self, better_pose):
        parent_link = 'arm_left_tool_link'
        box_name = 'box'
        box_pose = PoseStamped()
        box_pose.header.frame_id = parent_link
        box_pose.pose.position.x = 0.2
        box_pose.pose.orientation.w = 1
        better_pose.add_box(box_name,
                            (0.1, 0.1, 0.1),
                            pose=box_pose,
                            parent_link=parent_link,
                            parent_link_group=better_pose.robot_name)
        better_pose.set_joint_goal(better_pose.default_pose)
        better_pose.plan_and_execute()

    def test_demo1(self, apartment_setup: TiagoTestWrapper):
        # setup
        apartment_name = apartment_setup.environment_name
        l_tcp = 'gripper_left_grasping_frame'
        r_tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'handle_cab1_top_door'
        cupboard_floor_frame = 'cabinet1_coloksu_level4'
        cupboard_floor = 'cabinet1_coloksu_level4'
        base_footprint = 'base_footprint'
        countertop_frame = 'island_countertop'
        countertop = 'island_countertop'
        grasp_offset = 0.1
        start_base_pose = apartment_setup.world.compute_fk_pose(apartment_setup.default_root, base_footprint)

        # spawn cup
        cup_name = 'cup'
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = cupboard_floor_frame
        cup_height = 0.1653
        cup_pose.pose.position.z = cup_height / 2
        cup_pose.pose.position.x = 0.15
        cup_pose.pose.position.y = 0.15
        cup_pose.pose.orientation.w = 1
        apartment_setup.add_box(name=cup_name,
                                size=(0.0753, 0.0753, cup_height),
                                pose=cup_pose,
                                parent_link=cupboard_floor,
                                parent_link_group=apartment_name)

        # open cupboard
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -grasp_offset
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root,
                                      weight=WEIGHT_ABOVE_CA * 10,
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'cabinet1_door_top_left'
        # apartment_setup.set_diff_drive_tangential_to_point(goal_point)
        apartment_setup.set_keep_hand_in_workspace(tip_link=l_tcp)
        # apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point, weight=WEIGHT_BELOW_CA)
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root,
                                      check=False)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=l_tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(
            joint_list=['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                        'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
                        'arm_left_7_joint'])
        apartment_setup.plan_and_execute()

    def test_self_collision_avoidance(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': -1.1069832458862692,
            'arm_left_2_joint': 1.4746164329656843,
            'arm_left_3_joint': 2.7736173839819602,
            'arm_left_4_joint': 1.6237723180496708,
            'arm_left_5_joint': -1.5975088318771629,
            'arm_left_6_joint': 1.3300843607103001,
            'arm_left_7_joint': -0.016546381784501657,
            'arm_right_1_joint': -1.0919070230703032,
            'arm_right_2_joint': 1.4928456221831905,
            'arm_right_3_joint': 2.740050318770805,
            'arm_right_4_joint': 1.6576417817518292,
            'arm_right_5_joint': -1.4619211253492215,
            'arm_right_6_joint': 1.2787860569647924,
            'arm_right_7_joint': 0.013613188642612156,
            'gripper_left_left_finger_joint': 0.0393669359310417,
            'gripper_left_right_finger_joint': 0.04396903656716549,
            'gripper_right_left_finger_joint': 0.03097991016001716,
            'gripper_right_right_finger_joint': 0.04384773311365822,
            'head_1_joint': -0.10322685494051058,
            'head_2_joint': -1.0027367693813412,
            'torso_lift_joint': 0.2499968644929236,
        }
        # zero_pose.set_joint_goal(js)
        # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute()
        zero_pose.set_seed_configuration(js)
        # zero_pose.set_joint_goal(zero_pose.better_pose2)
        js2 = {
            'torso_lift_joint': 0.3400000002235174,
        }
        zero_pose.set_joint_goal(js2)
        zero_pose.plan()

    def test_avoid_self_collision4(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': 0.21181287002285662,
            'arm_left_2_joint': -0.6151379734525764,
            'arm_left_3_joint': 0.769352860826213,
            'arm_left_4_joint': 1.5366410535725352,
            'arm_left_5_joint': 0.6692852960138725,
            'arm_left_6_joint': 0.8499649769704987,
            'arm_left_7_joint': 0.3934248653346525,
            'arm_right_1_joint': 0.2605757577669546,
            'arm_right_2_joint': -1.1145267872723925,
            'arm_right_3_joint': 1.4016496683543236,
            'arm_right_4_joint': 2.1447945222448572,
            'arm_right_5_joint': -2.0050615624226524,
            'arm_right_6_joint': 1.2321070888078671,
            'arm_right_7_joint': 0.5130944511015763,
            'gripper_left_left_finger_joint': 0.0016253199614312491,
            'gripper_left_right_finger_joint': 0.0015128278396200111,
            'gripper_right_left_finger_joint': 0.043748218050233635,
            'gripper_right_right_finger_joint': 0.04384773311365822,
            'head_1_joint': 1.2314613306445925,
            'head_2_joint': 0.4333201391595926,
        }

        zero_pose.set_seed_configuration(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(zero_pose.better_pose2)
        zero_pose.plan_and_execute()

    def test_left_arm(self, zero_pose: TiagoTestWrapper):
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'arm_left_3_link'
        box_pose.pose.position.z = 0.07
        box_pose.pose.position.x = 0.1
        box_pose.pose.orientation.w = 1
        # zero_pose.add_box('box',
        #                   size=(0.05,0.05,0.05),
        #                   pose=box_pose)
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'arm_left_5_link'
        box_pose.pose.position.z = 0.07
        box_pose.pose.position.y = -0.1
        box_pose.pose.orientation.w = 1
        # zero_pose.add_box('box2',
        #                   size=(0.05,0.05,0.05),
        #                   pose=box_pose)
        # zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()

    def test_load_negative_scale(self, zero_pose: TiagoTestWrapper):
        mesh_path = 'package://tiago_description/meshes/arm/arm_3_collision.dae'
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.z = 0.0
        box_pose.pose.orientation.w = 1
        zero_pose.add_mesh('meshy',
                           mesh=mesh_path,
                           pose=box_pose,
                           scale=(1, 1, -1),
                           )
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.z = -0.1
        box_pose.pose.orientation.w = 1
        zero_pose.add_box('box1',
                          size=(0.1, 0.1, 0.01),
                          pose=box_pose,
                          parent_link='base_link',
                          parent_link_group=zero_pose.robot_name)
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.y = 0.1
        box_pose.pose.position.z = 0.05
        box_pose.pose.orientation.w = 1
        zero_pose.add_box('box2',
                          size=(0.1, 0.01, 0.1),
                          pose=box_pose,
                          parent_link='base_link',
                          parent_link_group=zero_pose.robot_name)
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.position.y = -0.1
        box_pose.pose.position.z = 0.05
        box_pose.pose.orientation.w = 1
        zero_pose.add_box('box3',
                          size=(0.1, 0.01, 0.1),
                          pose=box_pose,
                          parent_link='base_link',
                          parent_link_group=zero_pose.robot_name)
        # box_pose = PoseStamped()
        # box_pose.header.frame_id = 'base_link'
        # box_pose.pose.position.x = 0.6
        # box_pose.pose.orientation.w = 1
        # zero_pose.add_mesh('meshy2',
        #                    mesh=mesh_path,
        #                    pose=box_pose,
        #                    scale=(1, 1, 1),
        #                    )
        zero_pose.plan_and_execute()

    def test_drive_into_kitchen(self, apartment_setup: TiagoTestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = 2
        base_goal.pose.orientation.w = 1
        apartment_setup.move_base(base_goal, check=False)

    def test_open_cabinet_left(self, apartment_setup: TiagoTestWrapper):
        tcp = 'gripper_left_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'handle_cab1_top_door'
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -0.1
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=apartment_setup.world.root_link_name,
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'cabinet1_door_top_left'
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.set_avoid_joint_limits_goal(50)
        apartment_setup.plan_and_execute()


class TestConstraints:
    def test_DiffDriveTangentialToPoint(self, apartment_setup):
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        apartment_setup.set_diff_drive_tangential_to_point(goal_point=goal_point)
        apartment_setup.plan_and_execute()


class TestJointGoals:
    def test_out_of_joint_soft_limits3(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_right_5_joint': -2.1031066629465776,
        }
        zero_pose.set_seed_configuration(js)
        zero_pose.set_joint_goal(zero_pose.better_pose2, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_out_of_joint_soft_limits6(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': 0.2999719152605501,
            'arm_left_2_joint': -1.0770103752381537,
            'arm_left_3_joint': 1.538071509678013,
            'arm_left_4_joint': 1.9890333283428183,
            'arm_left_5_joint': -1.9048274775275487,
            'arm_left_6_joint': 1.1571632014060012,
            'arm_left_7_joint': 0.6375186679031444,
            'arm_right_1_joint': 0.2716512459628158,
            'arm_right_2_joint': -1.0826349036723986,
            'arm_right_3_joint': 1.5215471970784948,
            'arm_right_4_joint': 2.0190831292184264,
            'arm_right_5_joint': -1.9978066473844511,
            'arm_right_6_joint': 1.206231348680554,
            'arm_right_7_joint': 0.4999018438525896,
            'gripper_left_left_finger_joint': 0.0016879097059911336,
            'gripper_left_right_finger_joint': 0.0017554347466345558,
            'gripper_right_left_finger_joint': 0.0009368327712725209,
            'gripper_right_right_finger_joint': 0.01085319375968001,
            'head_1_joint': -0.0033940217264349197,
            'head_2_joint': -0.9843060924802811,
            'torso_lift_joint': 0.3487927046680741,
        }

        zero_pose.set_seed_configuration(js)
        pointing_goal = PointStamped()
        pointing_goal.header.frame_id = 'base_footprint'
        pointing_goal.point.x = 2
        z = Vector3Stamped()
        z.header.frame_id = 'xtion_link'
        z.vector.z = 1
        zero_pose.set_pointing_goal(tip_link='xtion_link',
                                    root_link='base_footprint',
                                    goal_point=pointing_goal,
                                    pointing_axis=z)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_out_of_joint_soft_limits7(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_left_1_joint': 1.3505632726981545,
            'arm_left_2_joint': -1.1195635667275154,
            'arm_left_3_joint': 2.3115915820828667,
            'arm_left_4_joint': 0.9410957928690423,
            'arm_left_5_joint': -0.9138386896689713,
            'arm_left_6_joint': 0.32143255957945216,
            'arm_left_7_joint': 2.0158598934576375,
            'arm_right_1_joint': 0.3006285274060041,
            'arm_right_2_joint': -1.1107991645139517,
            'arm_right_3_joint': 1.4956685979283315,
            'arm_right_4_joint': 2.0495483917627206,
            'arm_right_5_joint': -1.9978918685519071,
            'arm_right_6_joint': 1.2007401412818746,
            'arm_right_7_joint': 0.49898664261947634,
            'gripper_left_left_finger_joint': 0.04418634626215283,
            'gripper_left_right_finger_joint': 0.04378708138690458,
            'gripper_right_left_finger_joint': 0.03060437169265785,
            'gripper_right_right_finger_joint': 0.04051188814220821,
            'head_1_joint': 0.018108434658135126,
            'head_2_joint': 0.09081672674822555,
            'torso_lift_joint': 0.3486754163734459,
        }

        zero_pose.set_seed_configuration(js)
        js2 = deepcopy(js)
        # del js2['arm_right_2_joint']
        zero_pose.set_joint_goal(js2, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_joint_goals(self, zero_pose: TiagoTestWrapper):
        js1 = {
            'arm_left_1_joint': - 1.0,
            'arm_left_2_joint': 0.0,
            'arm_left_3_joint': 1.5,
            'arm_left_4_joint': 2.2,
            'arm_left_5_joint': - 1.5,
            'arm_left_6_joint': 0.5,
            'arm_left_7_joint': 0.0,
        }
        js2 = {
            'arm_right_1_joint': - 1.0,
            'arm_right_2_joint': 0.0,
            'arm_right_3_joint': 1.5,
            'arm_right_4_joint': 2.2,
            'arm_right_5_joint': - 1.5,
            'arm_right_6_joint': 0.5,
            'arm_right_7_joint': 0.0,
        }
        zero_pose.set_joint_goal(js1)
        zero_pose.set_joint_goal(js2)

        zero_pose.plan_and_execute()

    def test_joint_goals_at_limits(self, zero_pose: TiagoTestWrapper):
        js1 = {
            'head_1_joint': 5,
            'head_2_joint': -5
        }
        zero_pose.set_joint_goal(js1, check=False, weight=WEIGHT_ABOVE_CA)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_get_out_of_joint_soft_limits(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 2,
            'head_2_joint': -2
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_get_out_of_joint_soft_limits_passive(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_right_5_joint': 3,
            'arm_left_5_joint': -3
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_get_out_of_joint_soft_limits_passive_with_velocity(self, zero_pose: TiagoTestWrapper):
        js = {
            'arm_right_5_joint': 3,
            'arm_left_5_joint': -3
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.world.state[PrefixName('arm_right_5_joint', 'tiago_dual')].velocity = 1
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_try_to_stay_out_of_soft_limits(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 2,
            'head_2_joint': -2
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(js, weight=WEIGHT_BELOW_CA, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.are_joint_limits_violated()

    def test_torso_goal(self, zero_pose: TiagoTestWrapper):
        # js1 = {
        #     'torso_lift_joint': 0.2,
        # }
        # js2 = {
        #     'torso_lift_joint': 0.25,
        #     'head_1_joint': 1
        # }
        # zero_pose.set_joint_goal(js1)
        # # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute()
        #
        # zero_pose.set_joint_goal(js2)
        # # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute()

        # zero_pose.set_json_goal('SetSeedConfiguration',
        #                         seed_configuration=js_start)
        zero_pose.allow_self_collision()
        js = deepcopy(zero_pose.default_pose)
        # del js['head_1_joint']
        # del js['head_2_joint']
        zero_pose.set_joint_goal(js)
        zero_pose.plan()
        zero_pose.allow_self_collision()
        js = deepcopy(zero_pose.default_pose)
        del js['gripper_right_left_finger_joint']
        # del js['gripper_right_right_finger_joint']
        zero_pose.set_joint_goal(js)
        zero_pose.plan()

    def test_get_out_of_joint_soft_limits2(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 2,
            'head_2_joint': -2
        }
        js2 = {
            'head_1_joint': 2.1,
            'head_2_joint': 0,
            'torso_lift_joint': 1.5
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(js2, check=False)
        zero_pose.plan()
        zero_pose.are_joint_limits_violated()
