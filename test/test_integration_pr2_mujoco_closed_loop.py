import math
from copy import deepcopy
from typing import Optional

import numpy as np
from numpy import pi
import pytest
import math
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf.transformations import quaternion_from_matrix, quaternion_about_axis, quaternion_multiply

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult, MoveGoal, GiskardError
from giskardpy.configs.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.pr2 import PR2CollisionAvoidance, PR2VelocityMujocoInterface, WorldWithPR2Config
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.data_types import JointStates
from giskardpy.god_map import god_map
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from test_integration_pr2 import PR2TestWrapper, TestJointGoals, pocky_pose
from giskardpy.goals.manipulability_goals import MaxManipulability
from giskardpy.goals.adaptive_goals import CloseGripper, PouringAdaptiveTilt, PouringAdaptiveTiltScraping
from neem_interface_python.neem_interface import NEEMInterface
import time


class PR2TestWrapperMujoco(PR2TestWrapper):
    better_pose = {'r_shoulder_pan_joint': -1.7125,
                   'r_shoulder_lift_joint': -0.25672,
                   'r_upper_arm_roll_joint': -1.46335,
                   'r_elbow_flex_joint': -2.12,
                   'r_forearm_roll_joint': 1.76632,
                   'r_wrist_flex_joint': -0.10001,
                   'r_wrist_roll_joint': 0.05106,
                   'l_shoulder_pan_joint': 1.9652,
                   'l_shoulder_lift_joint': - 0.26499,
                   'l_upper_arm_roll_joint': 1.3837,
                   'l_elbow_flex_joint': -2.12,
                   'l_forearm_roll_joint': 16.99,
                   'l_wrist_flex_joint': - 0.10001,
                   'l_wrist_roll_joint': 0,
                   'torso_lift_joint': 0.2,
                   # 'l_gripper_l_finger_joint': 0.55,
                   # 'r_gripper_l_finger_joint': 0.55,
                   'head_pan_joint': 0,
                   'head_tilt_joint': 0,
                   }

    def __init__(self):
        del self.default_pose['l_gripper_l_finger_joint']
        del self.default_pose['r_gripper_l_finger_joint']
        self.r_tip = 'r_gripper_tool_frame'
        self.l_tip = 'l_gripper_tool_frame'
        self.l_gripper_group = 'l_gripper'
        self.r_gripper_group = 'r_gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.mujoco_reset = rospy.ServiceProxy('mujoco/reset', Trigger)
        self.odom_root = 'odom_combined'
        giskard = Giskard(world_config=WorldWithPR2Config(),
                          collision_avoidance_config=PR2CollisionAvoidance(),
                          robot_interface_config=PR2VelocityMujocoInterface(),
                          behavior_tree_config=ClosedLoopBTConfig(debug_mode=True, control_loop_max_hz=50),
                          qp_controller_config=QPControllerConfig(qp_solver=SupportedQPSolver.gurobi))
        super().__init__(giskard)

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = tf.get_tf_root()
        p.pose.orientation.w = 1
        # self.set_localization(p)
        # self.wait_heartbeats()

    def set_localization(self, map_T_odom: PoseStamped):
        pass

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.allow_all_collisions()
        self.move_base(goal_pose)

    def reset(self):
        self.mujoco_reset()
        super().reset()


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR2TestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


class TestJointGoalsMujoco(TestJointGoals):
    def test_joint_goal(self, zero_pose: PR2TestWrapper):
        js = {
            # 'torso_lift_joint': 0.2999225173357618,
            'head_pan_joint': 0.041880780651479044,
            'head_tilt_joint': -0.37,
            'r_upper_arm_roll_joint': -0.9487714747527726,
            'r_shoulder_pan_joint': -1.0047307505973626,
            'r_shoulder_lift_joint': 0.48736790658811985,
            'r_forearm_roll_joint': -14.895833882874182,
            'r_elbow_flex_joint': -1.392377908925028,
            'r_wrist_flex_joint': -0.4548695149411013,
            'r_wrist_roll_joint': 0.11426798984097819,
            'l_upper_arm_roll_joint': 1.7383062350263658,
            'l_shoulder_pan_joint': 1.8799810286792007,
            'l_shoulder_lift_joint': 0.011627231224188975,
            'l_forearm_roll_joint': 312.67276414458695,
            'l_elbow_flex_joint': -2.0300928925694675,
            'l_wrist_flex_joint': -0.10014623223021513,
            'l_wrist_roll_joint': -6.062015047706399,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_joint_goal_projection(self, zero_pose: PR2TestWrapper):
        js = {
            'torso_lift_joint': 0.2999225173357618,
            'head_pan_joint': 0.041880780651479044,
            'head_tilt_joint': -0.37,
            'r_upper_arm_roll_joint': -0.9487714747527726,
            'r_shoulder_pan_joint': -1.0047307505973626,
            'r_shoulder_lift_joint': 0.48736790658811985,
            'r_forearm_roll_joint': -14.895833882874182,
            'r_elbow_flex_joint': -1.392377908925028,
            'r_wrist_flex_joint': -0.4548695149411013,
            'r_wrist_roll_joint': 0.11426798984097819,
            'l_upper_arm_roll_joint': 1.7383062350263658,
            'l_shoulder_pan_joint': 1.8799810286792007,
            'l_shoulder_lift_joint': 0.011627231224188975,
            'l_forearm_roll_joint': 312.67276414458695,
            'l_elbow_flex_joint': -2.0300928925694675,
            'l_wrist_flex_joint': -0.1,
            'l_wrist_roll_joint': -6.062015047706399,
        }
        zero_pose.set_joint_goal(goal_state=js)
        zero_pose.allow_all_collisions()
        zero_pose.projection()

        zero_pose.set_joint_goal(goal_state=js)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        zero_pose.set_seed_configuration(zero_pose.better_pose)
        zero_pose.set_joint_goal(goal_state=js)
        zero_pose.allow_all_collisions()
        zero_pose.projection()


class TestMoveBaseGoals:
    def test_left_1m(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.y = 1
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_left_1cm(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.y = 0.01
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_left_rotate(self, zero_pose: PR2TestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.set_localization(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = -1
        # base_goal.pose.orientation.w = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(base_goal)

    def test_circle(self, zero_pose: PR2TestWrapper):
        center = PointStamped()
        center.header.frame_id = zero_pose.default_root
        zero_pose.set_json_goal(constraint_type='Circle',
                                center=center,
                                radius=0.3,
                                tip_link='base_footprint',
                                scale=0.1)
        # zero_pose.set_json_goal('PR2CasterConstraints')
        zero_pose.set_max_traj_length(new_length=160)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_stay_put(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = zero_pose.default_root
        # base_goal.pose.position.y = 0.05
        base_goal.pose.orientation.w = 1
        # zero_pose.set_json_goal('PR2CasterConstraints')
        zero_pose.set_joint_goal(zero_pose.better_pose)
        zero_pose.move_base(base_goal)

    def test_forward_1m(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(base_goal)

    def test_carry_my_bs(self, zero_pose: PR2TestWrapper):
        # zero_pose.set_json_goal('CarryMyBullshit',
        #                         camera_link='head_mount_kinect_rgb_optical_frame',
        #                         laser_topic_name='/laser',
        #                         height_for_camera_target=1.5)
        # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute(expected_error_codes=[GiskardError.PREEMPTED], stop_after=10)
        #
        # zero_pose.set_json_goal('CarryMyBullshit',
        #                         camera_link='head_mount_kinect_rgb_optical_frame',
        #                         laser_topic_name='/laser',
        #                         clear_path=True,
        #                         height_for_camera_target=1.5)
        # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute(expected_error_codes=[GiskardError.PREEMPTED], stop_after=10)

        zero_pose.set_json_goal('CarryMyBullshit',
                                camera_link='head_mount_kinect_rgb_optical_frame',
                                # point_cloud_laser_topic_name='',
                                laser_frame_id='base_laser_link',
                                height_for_camera_target=1.5)
        zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute(expected_error_codes=[GiskardError.PREEMPTED], stop_after=30)
        zero_pose.plan_and_execute(expected_error_codes=[GiskardError.ERROR])

        # zero_pose.set_json_goal('CarryMyBullshit',
        #                         camera_link='head_mount_kinect_rgb_optical_frame',
        #                         laser_topic_name='/laser',
        #                         drive_back=True)
        # zero_pose.allow_all_collisions()
        # zero_pose.plan_and_execute(expected_error_codes=[GiskardError.PREEMPTED], stop_after=10)

        zero_pose.set_json_goal('CarryMyBullshit',
                                camera_link='head_mount_kinect_rgb_optical_frame',
                                # laser_topic_name='/laser',
                                laser_frame_id='base_laser_link',
                                drive_back=True)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_wave(self, zero_pose: PR2TestWrapper):
        center = PointStamped()
        center.header.frame_id = zero_pose.default_root
        zero_pose.allow_all_collisions()
        zero_pose.set_json_goal(constraint_type='Wave',
                                center=center,
                                radius=0.05,
                                tip_link='base_footprint',
                                scale=2)
        zero_pose.set_joint_goal(zero_pose.better_pose, check=False)
        zero_pose.plan_and_execute()

    def test_forward_1cm(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.01
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_left_1m_1m(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = 1
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_left_1cm_1cm(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0.01
        base_goal.pose.position.y = 0.01
        base_goal.pose.orientation.w = 1
        zero_pose.move_base(base_goal)

    def test_forward_right_and_rotate(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.position.y = -1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 4, [0, 0, 1]))
        zero_pose.move_base(base_goal)

    def test_forward_then_left(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 4, [0, 0, 1]))
        zero_pose.move_base(base_goal)
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 0
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi / 3, [0, 0, 1]))
        zero_pose.move_base(base_goal)

    def test_rotate_pi_half(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(-pi / 2, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(base_goal)

    def test_rotate_pi(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.move_base(base_goal)

    def test_rotate_0_001_rad(self, zero_pose: PR2TestWrapper):
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(0.001, [0, 0, 1]))
        zero_pose.move_base(base_goal)


class TestWorldManipulation:
    def test_add_urdf_body(self, kitchen_setup: PR2TestWrapper):
        assert god_map.tree.wait_for_goal.synchronization._number_of_synchronisation_behaviors() == 2
        joint_goal = 0.2
        object_name = kitchen_setup.default_env_name
        kitchen_setup.set_env_state({'sink_area_left_middle_drawer_main_joint': joint_goal})
        joint_state = rospy.wait_for_message('/kitchen/joint_states', JointState, rospy.Duration(1))
        joint_state = JointStates.from_msg(joint_state)
        assert joint_state['sink_area_left_middle_drawer_main_joint'].position == joint_goal
        kitchen_setup.clear_world()
        assert god_map.tree.wait_for_goal.synchronization._number_of_synchronisation_behaviors() == 1
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 1
        p.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        if god_map.is_standalone():
            js_topic = ''
            set_js_topic = ''
        else:
            js_topic = '/kitchen/joint_states'
            set_js_topic = '/kitchen/cram_joint_states'
        kitchen_setup.add_urdf_to_world(name=object_name,
                                        urdf=rospy.get_param('kitchen_description'),
                                        pose=p,
                                        js_topic=js_topic,
                                        set_js_topic=set_js_topic)
        kitchen_setup.wait_heartbeats(1)
        assert god_map.tree.wait_for_goal.synchronization._number_of_synchronisation_behaviors() == 2
        joint_state = kitchen_setup.get_group_info(object_name).joint_state
        joint_state = JointStates.from_msg(joint_state)
        assert joint_state['iai_kitchen/sink_area_left_middle_drawer_main_joint'].position == joint_goal

        joint_goal = 0.1
        kitchen_setup.set_env_state({'sink_area_left_middle_drawer_main_joint': joint_goal})
        kitchen_setup.remove_group(object_name)
        assert god_map.tree.wait_for_goal.synchronization._number_of_synchronisation_behaviors() == 1
        kitchen_setup.add_urdf_to_world(name=object_name,
                                        urdf=rospy.get_param('kitchen_description'),
                                        pose=p,
                                        js_topic=js_topic,
                                        set_js_topic=set_js_topic)
        kitchen_setup.wait_heartbeats(1)
        assert god_map.tree.wait_for_goal.synchronization._number_of_synchronisation_behaviors() == 2
        joint_state = kitchen_setup.get_group_info(object_name).joint_state
        joint_state = JointStates.from_msg(joint_state)
        assert joint_state['iai_kitchen/sink_area_left_middle_drawer_main_joint'].position == joint_goal


class TestConstraints:

    def test_SetSeedConfiguration_execute(self, zero_pose: PR2TestWrapper):
        zero_pose.set_seed_configuration(seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.execute(expected_error_code=GiskardError.GOAL_INITIALIZATION_ERROR)

    def test_SetSeedConfiguration_execute2(self, zero_pose: PR2TestWrapper):
        zero_pose.set_seed_configuration(seed_configuration=zero_pose.better_pose)
        zero_pose.execute(expected_error_code=GiskardError.CONSTRAINT_INITIALIZATION_ERROR)

    def test_SetSeedConfiguration_project(self, zero_pose: PR2TestWrapper):
        zero_pose.set_seed_configuration(seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.projection()

    def test_bowl_and_cup(self, kitchen_setup: PR2TestWrapper):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCollisionAvoidanceGoals::test_bowl_and_cup
        bowl_name = 'bowl'
        cup_name = 'cup'
        percentage = 50
        drawer_handle = 'sink_area_left_middle_drawer_handle'
        drawer_joint = 'sink_area_left_middle_drawer_main_joint'
        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        # cup_pose.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
        cup_pose.pose.position = Point(0.1, 0.2, -.05)
        cup_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder_to_world(name=cup_name, height=0.07, radius=0.04, pose=cup_pose,
                                            parent_link='sink_area_left_middle_drawer_main')

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        bowl_pose.pose.position = Point(0.1, -0.2, -.05)
        bowl_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.add_cylinder_to_world(name=bowl_name, height=0.05, radius=0.07, pose=bowl_pose,
                                            parent_link='sink_area_left_middle_drawer_main')

        # grasp drawer handle
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = drawer_handle
        bar_axis.vector.y = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = drawer_handle

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1

        kitchen_setup.set_grasp_bar_goal(bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=0.4,
                                         tip_link=kitchen_setup.l_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         root_link=kitchen_setup.default_root)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = drawer_handle
        x_goal.vector.x = -1

        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.l_tip,
                                            tip_normal=x_gripper,
                                            root_link=kitchen_setup.default_root,
                                            goal_normal=x_goal)
        # kitchen_setup.allow_all_collisions()
        kitchen_setup.allow_collision(kitchen_setup.l_gripper_group, kitchen_setup.default_env_name)
        kitchen_setup.plan_and_execute()

        # open drawer
        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.l_tip,
                                              environment_link=drawer_handle)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_env_state({drawer_joint: 0.48})

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.y = 1
        base_pose.pose.position.x = .1
        base_pose.pose.orientation.w = 1
        kitchen_setup.move_base(base_pose)

        # grasp bowl
        l_goal = deepcopy(bowl_pose)
        l_goal.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        l_goal.pose.position.z += .2
        l_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_cart_goal(goal_pose=l_goal,
                                    tip_link=kitchen_setup.l_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.allow_collision(kitchen_setup.l_gripper_group, bowl_name)

        # grasp cup
        r_goal = deepcopy(cup_pose)
        r_goal.header.frame_id = 'iai_kitchen/sink_area_left_middle_drawer_main'
        r_goal.pose.position.z += .2
        r_goal.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                                      [0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 0, 0, 1]]))
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.set_cart_goal(goal_pose=r_goal,
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root,
                                    weight=WEIGHT_BELOW_CA)
        kitchen_setup.plan_and_execute()

        l_goal.pose.position.z -= .2
        r_goal.pose.position.z -= .2
        kitchen_setup.set_cart_goal(goal_pose=l_goal,
                                    tip_link=kitchen_setup.l_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(goal_pose=r_goal,
                                    tip_link=kitchen_setup.r_tip,
                                    root_link=kitchen_setup.default_root)
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=bowl_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=cup_name)
        kitchen_setup.plan_and_execute()

        kitchen_setup.update_parent_link_of_group(name=bowl_name, parent_link=kitchen_setup.l_tip)
        kitchen_setup.update_parent_link_of_group(name=cup_name, parent_link=kitchen_setup.r_tip)

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'base_footprint'
        base_goal.pose.position.x = -.1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi, [0, 0, 1]))
        kitchen_setup.move_base(base_goal)

        # place bowl and cup
        bowl_goal = PoseStamped()
        bowl_goal.header.frame_id = 'kitchen_island_surface'
        bowl_goal.pose.position = Point(.2, 0, .05)
        bowl_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        cup_goal = PoseStamped()
        cup_goal.header.frame_id = 'kitchen_island_surface'
        cup_goal.pose.position = Point(.15, 0.25, .07)
        cup_goal.pose.orientation = Quaternion(0, 0, 0, 1)

        kitchen_setup.set_cart_goal(goal_pose=bowl_goal, tip_link=bowl_name, root_link=kitchen_setup.default_root)
        kitchen_setup.set_cart_goal(goal_pose=cup_goal, tip_link=cup_name, root_link=kitchen_setup.default_root)
        kitchen_setup.set_avoid_joint_limits_goal(percentage=percentage)
        kitchen_setup.avoid_all_collisions(0.05)
        kitchen_setup.plan_and_execute()

        kitchen_setup.detach_group(name=bowl_name)
        kitchen_setup.detach_group(name=cup_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=cup_name)
        kitchen_setup.allow_collision(group1=kitchen_setup.robot_name, group2=bowl_name)
        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.plan_and_execute()


class TestActionServerEvents:
    def test_interrupt1(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(goal_pose=p, tip_link='base_footprint', root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_code=GiskardError.PREEMPTED, stop_after=1)

    def test_interrupt2(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(goal_pose=p, tip_link='base_footprint', root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_code=GiskardError.PREEMPTED, stop_after=6)

    def test_undefined_type(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.send_goal(goal_type=MoveGoal.UNDEFINED, expected_error_code=GiskardError.INVALID_GOAL)

    def test_empty_goal(self, zero_pose: PR2TestWrapper):
        zero_pose.cmd_seq = []
        zero_pose.plan_and_execute(expected_error_code=GiskardError.INVALID_GOAL)


class TestManipulability:
    def test_manip1(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'map'
        p.pose.position = Point(0.8, -0.3, 1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'map')
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
                                               root_link='torso_lift_link',
                                               tip_link='r_gripper_tool_frame')
        zero_pose.plan_and_execute()


class TestPouring:
    def test_pour_tray2(self, zero_pose: PR2TestWrapper):
        tip_link = zero_pose.l_tip
        p = PoseStamped()
        p.header.frame_id = 'map'

        p.pose.position = Point(1.7, -0.2, 0.6)
        p.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                 [0, 0, -1, 0],
                                                                 [0, 1, 0, 0],
                                                                 [0, 0, 0, 1]]))
        zero_pose.set_cart_goal(p, tip_link, 'map')
        # p.pose.position = Point(1.7, -0.4, 0.8)
        # zero_pose.set_cart_goal(p, zero_pose.r_tip, 'map')
        zero_pose.execute()

        p.pose.position = Point(1.78, -0.2, 0.6)
        zero_pose.set_cart_goal(p, tip_link, 'map')
        zero_pose.execute()

        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=-200)
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperRight',
                                               pub_topic='/pr2/r_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='r_gripper_l_finger_joint',
                                               effort_threshold=-0.001,
                                               effort=-150)
        zero_pose.execute()

        p.pose.position = Point(1.78, -0.2, 0.8)
        zero_pose.set_cart_goal(p, tip_link, 'map')
        zero_pose.execute()

        pot_pose = PoseStamped()
        pot_pose.header.frame_id = 'tray'
        pot_pose.pose.position = Point(0, 0, 0)
        pot_pose.pose.orientation.w = 1

        # add a new object at the pose of the pot and attach it to the right tip
        zero_pose.add_box('dummy', (0.1, 0.1, 0.01), pose=pot_pose, parent_link=tip_link)

        pot_pose = PoseStamped()
        pot_pose.header.frame_id = 'map'
        pot_pose.pose.position = Point(1.9, 0.37, 1)  # Point(2, 0.3, 1)
        # pot_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
        #                                                                 [-1, 0, 0, 0],
        #                                                                 [0, 0, 1, 0],
        #                                                                 [0, 0, 0, 1]]))
        # zero_pose.set_cart_goal(pot_pose, 'dummy', 'map')
        # goal_normal = Vector3Stamped()
        # goal_normal.header.frame_id = 'map'
        # goal_normal.vector.z = 1
        # tip_normal = Vector3Stamped()
        # tip_normal.header.frame_id = 'dummy'
        # tip_normal.vector.z = 1
        # zero_pose.set_align_planes_goal(goal_normal=goal_normal, tip_link='dummy', tip_normal=tip_normal, root_link='map')
        # zero_pose.execute()

        p2 = PoseStamped()
        p2.header.frame_id = 'dummy'
        p2.pose.position = Point(-0.02, 0.07, 0.06)
        p2.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                  [1, 0, 0, 0],
                                                                  [0, 1, 0, 0],
                                                                  [0, 0, 0, 1]]))

        tilt_axis = Vector3Stamped()
        tilt_axis.header.frame_id = 'dummy'
        tilt_axis.vector.y = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=PouringAdaptiveTiltScraping.__name__,
                                               name='pouring',
                                               tip='dummy',
                                               root='map',
                                               tilt_angle=0.2,
                                               pouring_pose=pot_pose,
                                               tilt_axis=tilt_axis,
                                               pre_tilt=False,
                                               with_feedback=True,
                                               scrape_tip=zero_pose.r_tip,
                                               scrape_pose=p2)

        # zero_pose.set_cart_goal(p2, zero_pose.r_tip, 'dummy')
        # zero_pose.set_cart_goal(pot_pose, 'dummy', 'map')
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='r_gripper_tool_frame'
        #                                        )
        # zero_pose.allow_all_collisions()
        # zero_pose.avoid_collision(0.01, zero_pose.r_gripper_group, zero_pose.l_gripper_group)
        zero_pose.allow_collision(zero_pose.r_gripper_group, 'dummy')
        zero_pose.execute(add_local_minimum_reached=False)

    def test_pour_pot(self, zero_pose: PR2TestWrapper):
        record_neem = True
        if record_neem:
            ni = NEEMInterface()
        task_type = "http://www.ease-crc.org/ont/SOMA.owl#Pouring"
        env_owl = '/home/huerkamp/workspace/new_giskard_ws/environment-pot-bowl.owl'
        env_owl_ind_name = 'world'  # '/home/huerkamp/workspace/new_giskard_ws/environment.owl#world'
        env_urdf = '/home/huerkamp/workspace/new_giskard_ws/new_world_pot_bowl.urdf'
        agent_owl = '/home/huerkamp/workspace/new_giskard_ws/src/knowrob/owl/robots/PR2.owl'
        agent_owl_ind_name = 'pr2'
        agent_urdf = '/home/huerkamp/workspace/new_giskard_ws/src/iai_pr2/iai_pr2_description/robots/pr2_calibrated_with_ft2_without_virtual_joints.urdf'
        # agent_owl_ind_name, _ = ni.get_agent_and_ee_individual()
        if record_neem:
            parent_action = ni.start_episode(task_type=task_type, env_owl=env_owl, env_owl_ind_name=env_owl_ind_name,
                                             env_urdf=env_urdf,
                                             agent_owl=agent_owl, agent_owl_ind_name=agent_owl_ind_name,
                                             agent_urdf=agent_urdf)
        start_time_grasp = time.time()

        p = PoseStamped()
        p.header.stamp = rospy.get_rostime()
        p.header.frame_id = 'map'

        rotation = 0  # .9
        rot1 = quaternion_about_axis(rotation, (0, 1, 0))
        rot2 = quaternion_about_axis(rotation, (0, 1, 0))

        h = 0.9
        p.pose.position = Point(2, -0.39, h)
        p.pose.orientation = Quaternion(*quaternion_multiply(rot1, quaternion_from_matrix([[0, 0, 1, 0],
                                                                                           [1, 0, 0, 0],
                                                                                           [0, 1, 0, 0],
                                                                                           [0, 0, 0, 1]])))

        p2 = PoseStamped()
        p2.header.stamp = rospy.get_rostime()
        p2.header.frame_id = 'map'
        p2.pose.position = Point(2, -0.0, h)
        p2.pose.orientation = Quaternion(*quaternion_multiply(rot2, quaternion_from_matrix([[0, 0, -1, 0],
                                                                                            [-1, 0, 0, 0],
                                                                                            [0, 1, 0, 0],
                                                                                            [0, 0, 0, 1]])))

        # zero_pose.allow_all_collisions()
        # zero_pose.set_cart_goal(p, zero_pose.r_tip, 'map')
        # zero_pose.set_cart_goal(p2, zero_pose.l_tip, 'map')
        # zero_pose.set_avoid_joint_limits_goal()
        # zero_pose.execute()
        ####################################################
        h = 0.01
        p.header.frame_id = 'pot'
        p2.header.frame_id = 'pot'
        p.pose.position = Point(0, -0.20, h)
        p2.pose.position = Point(0, 0.20, h)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'map')
        zero_pose.set_cart_goal(p2, zero_pose.l_tip, 'map')
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='r_gripper_tool_frame',
        #                                        gain=50)
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='l_gripper_tool_frame',
        #                                        gain=50)
        zero_pose.execute()
        ###################################################
        p.pose.position = Point(0, -0.15, h)
        p2.pose.position = Point(0, 0.15, h)
        zero_pose.allow_all_collisions()
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'map')
        zero_pose.set_cart_goal(p2, zero_pose.l_tip, 'map')
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='r_gripper_tool_frame')
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='l_gripper_tool_frame')
        zero_pose.execute()
        ###################################################
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperRight',
                                               pub_topic='/pr2/r_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='r_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=-180)
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=-180)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, 'map')
        zero_pose.set_cart_goal(p2, zero_pose.l_tip, 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute(add_local_minimum_reached=True)
        if record_neem:
            action_iri = ni.add_subaction_with_task(parent_action=parent_action,
                                                    task_type="http://www.ease-crc.org/ont/SOMA.owl#Grasping",
                                                    start_time=start_time_grasp, end_time=time.time()
                                                    )
            ni.assert_task_and_roles(action_iri=action_iri, task_type='holding',
                                     source_iri='http://knowrob.org/kb/environment.owl#pot',
                                     dest_iri='http://knowrob.org/kb/environment.owl#bowl',
                                     agent_iri=agent_owl_ind_name)
        start_time_transporting = time.time()
        ####################################################
        # current pose of the pot in simulation
        pot_pose = PoseStamped()
        pot_pose.header.frame_id = 'pot'
        pot_pose.pose.position = Point(0, 0, 0)
        pot_pose.pose.orientation.w = 1

        # add a new object at the pose of the pot and attach it to the right tip
        zero_pose.add_box('dummy', (0.1, 0.1, 0.1), pose=pot_pose, parent_link=zero_pose.r_tip)

        # pose of the left gripper relative to the attached object
        l_pose = PoseStamped()
        l_pose.header.frame_id = 'dummy'
        l_pose.pose.position = Point(0, 0.15, 0)
        l_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, -1, 0],
                                                                      [-1, 0, 0, 0],
                                                                      [0, 1, 0, 0],
                                                                      [0, 0, 0, 1]]))

        # update pot pose to the desired pose of the pot
        # pot_pose.pose.position.z = 0.2
        #
        # move the dummy pot and the left gripper relative to it
        pot_pose.pose.position = Point(0, 0, 0.2)
        zero_pose.set_cart_goal(pot_pose, 'dummy', 'map')
        zero_pose.set_cart_goal(l_pose, zero_pose.l_tip, zero_pose.r_tip, add_monitor=False)
        zero_pose.execute(add_local_minimum_reached=True)
        if record_neem:
            action_iri = ni.add_subaction_with_task(parent_action=parent_action,
                                                    task_type="http://www.ease-crc.org/ont/SOMA.owl#Transporting",
                                                    start_time=start_time_transporting, end_time=time.time()
                                                    )
            ni.assert_task_and_roles(action_iri=action_iri, task_type='MovingTo',
                                     source_iri='http://knowrob.org/kb/environment.owl#pot',
                                     dest_iri='http://knowrob.org/kb/environment.owl#bowl',
                                     agent_iri=agent_owl_ind_name)
        start_time_draining = time.time()
        ###############################################################
        pot_pose = PoseStamped()
        pot_pose.header.frame_id = 'map'
        pot_pose.pose.position = Point(1.9, 0.3, 1.1)
        pot_pose.pose.orientation.w = 1  # Quaternion(*quaternion_about_axis(-0.5, (1, 0, 0)))
        # Todo: rotate pot pose around x axis
        tilt_axis = Vector3Stamped()
        tilt_axis.header.frame_id = 'dummy'
        tilt_axis.vector.y = 2 / math.sqrt(5)
        tilt_axis.vector.x = -1 / math.sqrt(5)
        # tilt_axis.vector.y = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=PouringAdaptiveTilt.__name__,
                                               name='pouring',
                                               tip='dummy',
                                               root='map',
                                               tilt_angle=0.7,
                                               pouring_pose=pot_pose,
                                               tilt_axis=tilt_axis,
                                               pre_tilt=False)
        # rot1 = quaternion_about_axis(0.2, (0, 1, 0))
        # pot_pose.pose.orientation = Quaternion(*rot1)
        # zero_pose.set_cart_goal(pot_pose, 'dummy', 'map')
        zero_pose.set_cart_goal(l_pose, zero_pose.l_tip, zero_pose.r_tip, add_monitor=False)
        zero_pose.allow_all_collisions()
        # zero_pose.set_avoid_joint_limits_goal()
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
                                               root_link='torso_lift_link',
                                               tip_link='r_gripper_tool_frame',
                                               gain=3)
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
                                               root_link='torso_lift_link',
                                               tip_link='l_gripper_tool_frame',
                                               gain=3)
        zero_pose.execute(add_local_minimum_reached=False)
        if record_neem:
            action_iri = ni.add_subaction_with_task(parent_action=parent_action,
                                                    task_type="http://www.ease-crc.org/ont/SOMA.owl#Draining",
                                                    start_time=start_time_draining, end_time=time.time()
                                                    )
            ni.assert_task_and_roles(action_iri=action_iri, task_type='TiltForward',
                                     source_iri='http://knowrob.org/kb/environment.owl#pot',
                                     dest_iri='http://knowrob.org/kb/environment.owl#bowl',
                                     agent_iri=agent_owl_ind_name)
            ni.stop_episode('/home/huerkamp/workspace/new_giskard_ws')

    def test_pour_two_cups(self, zero_pose: PR2TestWrapper):
        record_neem = True
        if record_neem:
            ni = NEEMInterface()
        task_type = "http://www.ease-crc.org/ont/SOMA.owl#Pouring"
        env_owl = '/home/huerkamp/workspace/new_giskard_ws/environment.owl'
        env_owl_ind_name = 'world'  # '/home/huerkamp/workspace/new_giskard_ws/environment.owl#world'
        env_urdf = '/home/huerkamp/workspace/new_giskard_ws/new_world_cups.urdf'
        agent_owl = '/home/huerkamp/workspace/new_giskard_ws/src/knowrob/owl/robots/PR2.owl'
        agent_owl_ind_name = 'pr2'
        agent_urdf = '/home/huerkamp/workspace/new_giskard_ws/src/iai_pr2/iai_pr2_description/robots/pr2_calibrated_with_ft2_without_virtual_joints.urdf'
        # agent_owl_ind_name, _ = ni.get_agent_and_ee_individual()
        if record_neem:
            parent_action = ni.start_episode(task_type=task_type, env_owl=env_owl, env_owl_ind_name=env_owl_ind_name,
                                             env_urdf=env_urdf,
                                             agent_owl=agent_owl, agent_owl_ind_name=agent_owl_ind_name,
                                             agent_urdf=agent_urdf)
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=100,
                                               as_open=True)
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperRight',
                                               pub_topic='/pr2/r_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='r_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=100,
                                               as_open=True)
        zero_pose.execute()

        start_time_grasp = time.time()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, 1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 2.015
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.5

        zero_pose.set_cart_goal(goal_pose, zero_pose.l_tip, 'map')
        goal_pose2 = PoseStamped()
        goal_pose2.header.frame_id = 'map'
        goal_pose2.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                          [0, 1, 0, 0],
                                                                          [0, 0, 1, 0],
                                                                          [0, 0, 0, 1]]))
        goal_pose2.pose.position.x = 2.02
        goal_pose2.pose.position.y = -0.6
        goal_pose2.pose.position.z = 0.5

        # zero_pose.set_cart_goal(goal_pose2, zero_pose.r_tip, 'map')
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='r_gripper_tool_frame',
        #                                        gain=3)
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=MaxManipulability.__name__,
        #                                        root_link='torso_lift_link',
        #                                        tip_link='l_gripper_tool_frame',
        #                                        gain=3)
        zero_pose.add_default_end_motion_conditions()
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperRight',
                                               pub_topic='/pr2/r_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='r_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=-180)
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=-180)
        zero_pose.execute()
        if record_neem:
            action_iri = ni.add_subaction_with_task(parent_action=parent_action,
                                                    task_type="http://www.ease-crc.org/ont/SOMA.owl#Grasping",
                                                    start_time=start_time_grasp, end_time=time.time()
                                                    )
            ni.assert_task_and_roles(action_iri=action_iri, task_type='holding',
                                     source_iri='http://knowrob.org/kb/environment.owl#free_cup',
                                     dest_iri='http://knowrob.org/kb/environment.owl#free_cup2',
                                     agent_iri=agent_owl_ind_name)
        start_time_transporting = time.time()

        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'free_cup'
        cup_pose.pose.position = Point(0, 0, 0)
        cup_pose.pose.orientation.w = 1

        # add a new object at the pose of the pot and attach it to the right tip
        zero_pose.add_box('cup1', (0.07, 0.07, 0.18), pose=cup_pose, parent_link=zero_pose.l_tip)
        cup_pose.header.frame_id = 'free_cup2'
        zero_pose.add_box('cup2', (0.07, 0.07, 0.18), pose=cup_pose, parent_link='map')

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, 1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 2
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.7

        zero_pose.set_cart_goal(goal_pose, zero_pose.l_tip, 'map')
        goal_pose2 = PoseStamped()
        goal_pose2.header.frame_id = 'map'
        goal_pose2.pose.orientation = Quaternion(*quaternion_about_axis(-np.pi / 4, [1, 0, 0]))
        goal_pose2.pose.position.x = 2.01
        goal_pose2.pose.position.y = -0.6
        goal_pose2.pose.position.z = 0.7

        # zero_pose.set_cart_goal(goal_pose2, zero_pose.r_tip, 'map')
        zero_pose.add_default_end_motion_conditions()
        zero_pose.allow_all_collisions()
        zero_pose.execute()
        if record_neem:
            action_iri = ni.add_subaction_with_task(parent_action=parent_action,
                                                    task_type="http://www.ease-crc.org/ont/SOMA.owl#Transporting",
                                                    start_time=start_time_transporting, end_time=time.time()
                                                    )
            ni.assert_task_and_roles(action_iri=action_iri, task_type='MovingTo',
                                     source_iri='http://knowrob.org/kb/environment.owl#free_cup',
                                     dest_iri='http://knowrob.org/kb/environment.owl#free_cup2',
                                     agent_iri=agent_owl_ind_name)
        start_time_pouring = time.time()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, 1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 2
        goal_pose.pose.position.y = -0.4
        goal_pose.pose.position.z = 0.7
        tilt_axis = Vector3Stamped()
        tilt_axis.header.frame_id = zero_pose.l_tip
        tilt_axis.vector.x = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=PouringAdaptiveTilt.__name__,
                                               name='pouring',
                                               tip=zero_pose.l_tip,
                                               root='map',
                                               tilt_angle=1.2,
                                               pouring_pose=goal_pose,
                                               tilt_axis=tilt_axis,
                                               pre_tilt=True,
                                               parent_action=None,
                                               agent_iri=None)
        zero_pose.allow_all_collisions()
        # zero_pose.avoid_collision(0.01, zero_pose.l_gripper_group, 'cup2')
        # zero_pose.set_cart_goal(goal_pose2, zero_pose.r_tip, 'map', add_monitor=False)
        zero_pose.execute(add_local_minimum_reached=True)

        if record_neem:
            action_iri = ni.add_subaction_with_task(parent_action=parent_action,
                                                    task_type="http://www.ease-crc.org/ont/SOMA.owl#Tilting",
                                                    start_time=start_time_pouring, end_time=time.time()
                                                    )
            ni.assert_task_and_roles(action_iri=action_iri, task_type='TiltForward',
                                     source_iri='http://knowrob.org/kb/environment.owl#free_cup',
                                     dest_iri='http://knowrob.org/kb/environment.owl#free_cup2',
                                     agent_iri=agent_owl_ind_name)
        # place cup back on table
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = 2
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.55

        zero_pose.set_cart_goal(goal_pose, zero_pose.l_tip, 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=100,
                                               as_open=True)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        if record_neem:
            ni.stop_episode('/home/huerkamp/workspace/new_giskard_ws')

    def test_complete_pouring(self, zero_pose: PR2TestWrapperMujoco):
        # first start related scripts for BB detection and scene action reasoning
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=100,
                                               as_open=True)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                         [0, 1, 0, 0],
                                                                         [0, 0, 1, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 2.01
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.5

        zero_pose.set_cart_goal(goal_pose, zero_pose.l_tip, 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripperLeft',
                                               pub_topic='/pr2/l_gripper_controller/command',
                                               joint_state_topic='pr2/joint_states',
                                               alibi_joint_name='l_gripper_l_finger_joint',
                                               effort_threshold=-0.14,
                                               effort=-180)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        goal_pose.pose.position.x = 2.01
        goal_pose.pose.position.y = -0.4
        goal_pose.pose.position.z = 0.7
        # goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
        #                                                                  [0, -1, 0, 0],
        #                                                                  [1, 0, 0, 0],
        #                                                                  [0, 0, 0, 1]]))
        tilt_axis = Vector3Stamped()
        tilt_axis.header.frame_id = zero_pose.l_tip
        tilt_axis.vector.x = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=PouringAdaptiveTilt.__name__,
                                               name='pouring',
                                               tip=zero_pose.l_tip,
                                               root='map',
                                               tilt_angle=1,
                                               pouring_pose=goal_pose,
                                               tilt_axis=tilt_axis,
                                               pre_tilt=True)
        zero_pose.allow_all_collisions()
        zero_pose.execute(add_local_minimum_reached=False)

        # goal_pose.pose.position.x = 1.93
        # goal_pose.pose.position.y = -0.2
        # goal_pose.pose.position.z = 0.3
        #
        # zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
        #                                        name='openGripper',
        #                                        as_open=True,
        #                                        velocity_threshold=100,
        #                                        effort_threshold=1,
        #                                        effort=100)
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()
        #
        # goal_pose.pose.position.x = 1.4
        # goal_pose.pose.position.y = -0.2
        # goal_pose.pose.position.z = 0.4
        #
        # zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()

# kernprof -lv py.test -s test/test_integration_pr2.py
# time: [1-9][1-9]*.[1-9]* s
# import pytest
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_goal'])
# pytest.main(['-s', __file__ + '::TestConstraints::test_open_dishwasher_apartment'])
# pytest.main(['-s', __file__ + '::TestConstraints::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_go_around_corner'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_box_between_boxes'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_self_collision'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_at_kitchen_corner'])
# pytest.main(['-s', __file__ + '::TestWayPoints::test_waypoints2'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_keep_position3'])
