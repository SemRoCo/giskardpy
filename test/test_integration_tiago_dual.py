import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis, quaternion_from_matrix

import giskardpy.utils.tfwrapper as tf
from giskardpy.configs.tiago import TiagoMujoco
from utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = TiagoTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TiagoTestWrapper(GiskardTestWrapper):
    default_pose = {
        'torso_lift_joint': 2.220446049250313e-16,
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
    }

    def __init__(self):
        self.mujoco_reset = rospy.ServiceProxy('reset', Trigger)
        super().__init__(TiagoMujoco)

    def move_base(self, goal_pose):
        tip_link = 'base_footprint'
        root_link = tf.get_tf_root()
        self.set_json_goal(constraint_type='DiffDriveBaseGoal',
                           tip_link=tip_link, root_link=root_link,
                           goal_pose=goal_pose)
        # self.allow_all_collisions()
        self.plan_and_execute()

    def reset(self):
        self.mujoco_reset()
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

        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()

    def test_drive2(self, zero_pose: TiagoTestWrapper):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(0.489, -0.598, 0.000)
        goal.pose.orientation.w = 1
        zero_pose.move_base(goal)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(-0.026, 0.569, 0.000)
        goal.pose.orientation = Quaternion(0, 0, 0.916530200374776, 0.3999654882623912)
        zero_pose.move_base(goal)


class TestCollisionAvoidance:
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

    def test_open_cabinet(self, apartment_setup: TiagoTestWrapper):
        tcp = 'gripper_left_grasping_frame'
        handle_name = 'cabinet1_handle_top'
        handle_name_frame = 'iai_apartment/cabinet1_handle_top'
        joint_name = 'cabinet1_door_top_left_joint'
        goal_angle = - np.pi / 2
        left_pose = PoseStamped()
        left_pose.header.frame_id = handle_name_frame
        left_pose.pose.position.x = 0.1
        left_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[-1, 0, 0, 0],
                                                                         [0, 0, -1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [0, 0, 0, 1]]))
        apartment_setup.set_cart_goal(left_pose,
                                      tip_link=tcp,
                                      root_link=tf.get_tf_root(),
                                      check=False)
        apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=goal_angle)
        apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()
        apartment_setup.set_apartment_js({joint_name: goal_angle})

        apartment_setup.set_json_goal('Open',
                                      tip_link=tcp,
                                      environment_link=handle_name,
                                      goal_joint_state=0)
        apartment_setup.allow_all_collisions()
        apartment_setup.plan_and_execute()
        apartment_setup.set_apartment_js({joint_name: 0})

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

    def test_SetSeedConfiguration(self, zero_pose: TiagoTestWrapper):
        zero_pose.set_json_goal('SetSeedConfiguration',
                                seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan()