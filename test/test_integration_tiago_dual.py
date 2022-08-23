import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PointStamped
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis, quaternion_from_matrix

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult
from giskardpy.configs.tiago import TiagoMujoco
from giskardpy.goals.goal import WEIGHT_BELOW_CA
from giskardpy.utils.utils import publish_pose
from utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def giskard(request, ros):
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
        'gripper_right_left_finger_joint': 0,
        'gripper_right_right_finger_joint': 0,
        'gripper_left_left_finger_joint': 0,
        'gripper_left_right_finger_joint': 0,
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
    }

    def __init__(self):
        self.mujoco_reset = rospy.ServiceProxy('reset', Trigger)
        super().__init__(TiagoMujoco)

    def move_base(self, goal_pose):
        tip_link = 'base_footprint'
        root_link = tf.get_tf_root()
        publish_pose(goal_pose)
        self.set_diff_drive_base_goal(goal_pose=goal_pose,
                                      tip_link=tip_link,
                                      root_link=root_link)
        self.set_max_traj_length(30)
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
        while not self.mujoco_reset().success:
            rospy.sleep(0.5)
        self.clear_world()
        self.reset_base()


class TestCartGoals:
    def test_drive(self, zero_pose: TiagoTestWrapper):
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

    def test_drive3(self, apartment_setup: TiagoTestWrapper):
        countertop_frame = 'iai_apartment/island_countertop'

        start_base_pose = PoseStamped()
        start_base_pose.header.frame_id = 'map'
        start_base_pose.pose.position = Point(1.295, 2.294, 0)
        start_base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.990, -0.139)
        apartment_setup.allow_all_collisions()
        apartment_setup.move_base(start_base_pose)

        countertip_P_goal = PointStamped()
        countertip_P_goal.header.frame_id = countertop_frame
        countertip_P_goal.point.x = 1.3
        countertip_P_goal.point.y = -0.3
        map_P_goal = tf.msg_to_homogeneous_matrix(tf.transform_point('map', countertip_P_goal))
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
        apartment_setup.allow_all_collisions()
        apartment_setup.move_base(base_pose)

    def test_drive_new(self, better_pose: TiagoTestWrapper):
        tip_link = 'gripper_left_grasping_frame'
        root_link = 'map'
        # map_T_eef = tf.lookup_pose(root_link, tip_link)
        # map_T_eef.pose.orientation = Quaternion(*quaternion_from_matrix([[1,0,0,0,],
        #                                                                  [0,0,1,0],
        #                                                                  [0,-1,0,0],
        #                                                                  [0,0,0,1]]))
        # better_pose.set_cart_goal(map_T_eef, tip_link, 'base_footprint', root_link2='map', check=False)
        #
        # # base_goal = PoseStamped()
        # # base_goal.header.frame_id = 'map'
        # # base_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0,-1,0,0,],
        # #                                                                  [1,0,0,0],
        # #                                                                  [0,0,1,0],
        # #                                                                  [0,0,0,1]]))
        # # better_pose.set_cart_goal(base_goal, 'base_footprint', 'map', check=False)
        # better_pose.plan_and_execute()

        # tip_link = 'base_footprint'
        goal = PoseStamped()
        goal.header.frame_id = tip_link
        # goal.pose.position.x = 1
        goal.pose.position.z = 1.3
        goal.pose.orientation.w = 1
        # goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))

        # better_pose.set_cart_goal(goal, tip_link=tip_link, root_link=root_link, weight=WEIGHT_BELOW_CA)
        better_pose.set_json_goal('KeepHandInWorkspace',
                                  map_frame='map',
                                  base_footprint='base_footprint',
                                  tip_link=tip_link)
        # gp = PointStamped()
        # gp.header.frame_id = tip_link
        # better_pose.set_pointing_goal(tip_link=tip_link,
        #                               goal_point=gp,
        #                               root_link='map',
        #                               )
        # better_pose.set_json_goal('PointingDiffDriveEEF',
        #                           base_tip='base_footprint',
        #                           base_root='map',
        #                           eef_tip=tip_link,
        #                           eef_root='base_footprint')
        better_pose.allow_all_collisions()
        better_pose.plan_and_execute()

    def test_drive2(self, zero_pose: TiagoTestWrapper):
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


class TestCollisionAvoidance:
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

    def test_demo1(self, apartment_setup: TiagoTestWrapper):
        # setup
        apartment_name = 'apartment'
        l_tcp = 'gripper_left_grasping_frame'
        r_tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'iai_apartment/handle_cab1_top_door'
        cupboard_floor_frame = 'iai_apartment/cabinet1_coloksu_level4'
        cupboard_floor = 'cabinet1_coloksu_level4'
        base_footprint = 'base_footprint'
        countertop_frame = 'iai_apartment/island_countertop'
        countertop = 'island_countertop'
        grasp_offset = 0.1
        start_base_pose = tf.lookup_pose(apartment_setup.default_root, base_footprint)

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
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        # apartment_setup.set_diff_drive_tangential_to_point(goal_point)
        apartment_setup.set_keep_hand_in_workspace(tip_link=r_tcp)
        # apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=l_tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        # grasp cup
        apartment_setup.set_joint_goal(apartment_setup.better_pose, check=False)
        apartment_setup.plan_and_execute()
        apartment_setup.move_base(start_base_pose)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cup_name
        grasp_pose.pose.position.x = grasp_offset
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                          [0, -1, 0, 0],
                                                                          [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(goal_pose=grasp_pose,
                                      tip_link=l_tcp,
                                      root_link=apartment_setup.default_root)
        apartment_setup.set_keep_hand_in_workspace(tip_link=l_tcp)
        apartment_setup.plan_and_execute()
        apartment_setup.update_parent_link_of_group(cup_name, l_tcp)

        # place cup
        apartment_setup.set_joint_goal(apartment_setup.better_pose, check=False)
        apartment_setup.plan_and_execute()
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
        apartment_setup.move_base(base_pose)

        cup_place_pose = PoseStamped()
        cup_place_pose.header.frame_id = countertop_frame
        cup_place_pose.pose.position.x = 0.25
        # cup_place_pose.pose.position.y = 0.
        cup_place_pose.pose.position.z = 0.02
        cup_place_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                              [0, -1, 0, 0],
                                                                              [0, 0, 1, 0],
                                                                              [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(goal_pose=cup_place_pose,
                                      tip_link=cup_name,
                                      root_link=apartment_setup.default_root)
        apartment_setup.plan_and_execute()

    def test_demo2(self, apartment_setup: TiagoTestWrapper):
        # setup
        apartment_name = 'apartment'
        l_tcp = 'gripper_left_grasping_frame'
        r_tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'iai_apartment/handle_cab1_top_door'
        cupboard_floor_frame = 'iai_apartment/cabinet1_coloksu_level4'
        cupboard_floor = 'cabinet1_coloksu_level4'
        base_footprint = 'base_footprint'
        grasp_offset = 0.1
        start_base_pose = tf.lookup_pose(apartment_setup.default_root, base_footprint)

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
                                      root_link=tf.get_tf_root(),
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=l_tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        # grasp cup
        apartment_setup.set_joint_goal(apartment_setup.better_pose)
        apartment_setup.plan_and_execute()
        apartment_setup.move_base(start_base_pose)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = cup_name
        grasp_pose.pose.position.x = grasp_offset
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                          [0, -1, 0, 0],
                                                                          [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(goal_pose=grasp_pose,
                                      tip_link=r_tcp,
                                      root_link=apartment_setup.default_root)
        apartment_setup.set_keep_hand_in_workspace(tip_link=r_tcp)
        apartment_setup.plan_and_execute()

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
                          parent_link_group=zero_pose.get_robot_name())
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
                          parent_link_group=zero_pose.get_robot_name())
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
                          parent_link_group=zero_pose.get_robot_name())
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
        apartment_setup.move_base(base_goal)

    def test_open_cabinet_left(self, apartment_setup: TiagoTestWrapper):
        tcp = 'gripper_left_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'iai_apartment/handle_cab1_top_door'
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
                                      root_link=tf.get_tf_root(),
                                      check=False)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

    def test_open_cabinet_right(self, apartment_setup: TiagoTestWrapper):
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'iai_apartment/side_B'
        base_pose.pose.position.x = 1.6
        base_pose.pose.position.y = 3.1
        base_pose.pose.orientation.w = 1
        base_pose = tf.transform_pose(tf.get_tf_root(), base_pose)
        apartment_setup.set_localization(base_pose)
        tcp = 'gripper_right_grasping_frame'
        handle_name = 'handle_cab1_top_door'
        handle_name_frame = 'iai_apartment/handle_cab1_top_door'
        goal_angle = np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = -0.1
        left_pose.pose.orientation.w = 1
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=tf.get_tf_root(),
                                      check=False)
        apartment_setup.set_json_goal('KeepHandInWorkspace',
                                      map_frame='map',
                                      base_footprint='base_footprint',
                                      tip_link=tcp)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        # apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
        #                               goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.avoid_joint_limits(50)
        apartment_setup.plan_and_execute()

    def test_dishwasher(self, apartment_setup: TiagoTestWrapper):
        dishwasher_middle = 'iai_apartment/dishwasher_drawer_middle'
        base_pose = PoseStamped()
        base_pose.header.frame_id = dishwasher_middle
        base_pose.pose.position.x = -1
        base_pose.pose.position.y = -0.25
        base_pose.pose.orientation.w = 1
        base_pose = tf.transform_pose(tf.get_tf_root(), base_pose)
        base_pose.pose.position.z = 0
        apartment_setup.set_localization(base_pose)

        tcp = 'gripper_left_grasping_frame'
        handle_name = 'handle_cab7'
        handle_name_frame = 'iai_apartment/handle_cab7'
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
                                      root_link=tf.get_tf_root(),
                                      check=False)
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        # apartment_setup.set_json_goal('KeepHandInWorkspace',
        #                               map_frame='map',
        #                               base_footprint='base_footprint',
        #                               tip_link=tcp)
        # apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()
        # apartment_setup.set_apartment_js({joint_name: goal_angle})

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        # apartment_setup.set_json_goal('KeepHandInWorkspace',
        #                               map_frame='map',
        #                               base_footprint='base_footprint',
        #                               tip_link=tcp)
        # apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()
        # apartment_setup.set_apartment_js({joint_name: 0})

    def test_hand_in_cabinet(self, apartment_setup: TiagoTestWrapper):
        tcp = 'gripper_left_grasping_frame'
        handle_name_frame = 'iai_apartment/cabinet1'
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        # left_pose.pose.position.x = 0.1
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=tf.get_tf_root(),
                                      check=False)
        apartment_setup.plan_and_execute()


class TestConstraints:
    def test_DiffDriveTangentialToPoint(self, apartment_setup):
        goal_point = PointStamped()
        goal_point.header.frame_id = 'iai_apartment/cabinet1_door_top_left'
        apartment_setup.set_json_goal('DiffDriveTangentialToPoint',
                                      goal_point=goal_point)
        apartment_setup.plan_and_execute()


class TestJointGoals:
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
            'head_1_joint': 99,
            'head_2_joint': 99
        }
        zero_pose.set_joint_goal(js1, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_SetSeedConfiguration(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan()

    def test_SetSeedConfiguration2(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

    def test_get_out_of_joint_soft_limits(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 1.3,
            'head_2_joint': -1
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan()

    def test_get_out_of_joint_limits(self, zero_pose: TiagoTestWrapper):
        js = {
            'head_1_joint': 2,
            'head_2_joint': -2
        }
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=js)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan(expected_error_codes=[MoveResult.OUT_OF_JOINT_LIMITS])
