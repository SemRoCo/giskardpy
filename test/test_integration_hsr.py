from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped, Vector3
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskard_msgs.msg import GiskardError
from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfig, HSRStandaloneInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.god_map import god_map
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import compare_poses, GiskardTestWrapper
import giskardpy.utils.tfwrapper as tf


class HSRTestWrapper(GiskardTestWrapper):
    default_pose = {
        'arm_flex_joint': -0.03,
        'arm_lift_joint': 0.01,
        'arm_roll_joint': 0.0,
        'head_pan_joint': 0.0,
        'head_tilt_joint': 0.0,
        'wrist_flex_joint': 0.0,
        'wrist_roll_joint': 0.0,
        'hand_motor_joint': 0.8
    }
    better_pose = default_pose

    def __init__(self, giskard=None):
        self.tip = 'hand_gripper_tool_frame'
        if giskard is None:
            giskard = Giskard(world_config=WorldWithHSRConfig(),
                              collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                              robot_interface_config=HSRStandaloneInterface(),
                              behavior_tree_config=StandAloneBTConfig(debug_mode=True,
                                                                      publish_tf=True,
                                                                      publish_js=False,
                                                                      simulation_max_hz=20),
                              qp_controller_config=QPControllerConfig())
        super().__init__(giskard)
        self.gripper_group = 'gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom'
        self.robot = god_map.world.groups[self.robot_name]

    def open_gripper(self):
        self.command_gripper(1.23)

    def close_gripper(self):
        self.command_gripper(0)

    def command_gripper(self, width):
        js = {'hand_motor_joint': width}
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset(self):
        self.register_group('gripper',
                            root_link_group_name=self.robot_name,
                            root_link_name='hand_palm_link')


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://hsr_description/launch/upload_hsrb.launch')
    c = HSRTestWrapper()
    # c = HSRTestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def box_setup(zero_pose: HSRTestWrapper) -> HSRTestWrapper:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.1
    p.pose.orientation.w = 1
    zero_pose.add_box_to_world(name='box', size=(1, 1, 1), pose=p)
    return zero_pose


class TestJointGoals:

    def test_mimic_joints(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = god_map.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()
        hand_T_finger_current = god_map.world.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
        hand_T_finger_expected = PoseStamped()
        hand_T_finger_expected.header.frame_id = 'hand_palm_link'
        hand_T_finger_expected.pose.position.x = -0.01675
        hand_T_finger_expected.pose.position.y = -0.0907
        hand_T_finger_expected.pose.position.z = 0.0052
        hand_T_finger_expected.pose.orientation.x = -0.0434
        hand_T_finger_expected.pose.orientation.y = 0.0
        hand_T_finger_expected.pose.orientation.z = 0.0
        hand_T_finger_expected.pose.orientation.w = 0.999
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        js = {'torso_lift_joint': 0.1}
        zero_pose.set_joint_goal(js, add_monitor=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state[arm_lift_joint].position, 0.2, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = god_map.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints2(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = god_map.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()

        tip = 'hand_gripper_tool_frame'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.2
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state[arm_lift_joint].position, 0.2, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = god_map.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints3(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = god_map.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()
        tip = 'head_pan_link'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state[arm_lift_joint].position, 0.3, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.902
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = god_map.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints4(self, zero_pose: HSRTestWrapper):
        ll, ul = god_map.world.get_joint_velocity_limits('hsrb/arm_lift_joint')
        assert ll == -0.15
        assert ul == 0.15
        ll, ul = god_map.world.get_joint_velocity_limits('hsrb/torso_lift_joint')
        assert ll == -0.075
        assert ul == 0.075
        joint_goal = {'torso_lift_joint': 0.25}
        zero_pose.set_joint_goal(joint_goal, add_monitor=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(god_map.world.state['hsrb/arm_lift_joint'].position, 0.5, decimal=2)


class TestCartGoals:
    def test_save_graph_pdf(self, kitchen_setup):
        box1_name = 'box1'
        pose = PoseStamped()
        pose.header.frame_id = kitchen_setup.default_root
        pose.pose.orientation.w = 1
        kitchen_setup.add_box_to_world(name=box1_name,
                                       size=(1, 1, 1),
                                       pose=pose,
                                       parent_link='hand_palm_link',
                                       parent_link_group='hsrb')
        god_map.world.save_graph_pdf()

    def test_move_base(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.teleport_base(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position.x = 1
        base_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(goal_pose=base_goal, tip_link='base_footprint', root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

    def test_move_base_1m_forward(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_1m_left(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_1m_diagonal(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.position.y = 1
        map_T_odom.pose.orientation.w = 1
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_rotate(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_move_base_forward_rotate(self, zero_pose: HSRTestWrapper):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = 'map'
        map_T_odom.pose.position.x = 1
        map_T_odom.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 3, [0, 0, 1]))
        zero_pose.allow_all_collisions()
        zero_pose.move_base(map_T_odom)

    def test_rotate_gripper(self, zero_pose: HSRTestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(goal_pose=r_goal, tip_link=zero_pose.tip, root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()


class TestConstraints:

    def test_open_fridge(self, kitchen_setup: HSRTestWrapper):
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position = Point(0.3, -0.5, 0)
        base_goal.pose.orientation.w = 1
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1

        kitchen_setup.set_grasp_bar_goal(root_link=kitchen_setup.default_root,
                                         tip_link=kitchen_setup.tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.tip
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.tip,
                                            tip_normal=x_gripper,
                                            goal_normal=x_goal,
                                            root_link='map')
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.execute()
        current_pose = god_map.world.compute_fk_pose(root='map', tip=kitchen_setup.tip)

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
                                              environment_link=handle_name,
                                              goal_joint_state=1.5)
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.execute()
        kitchen_setup.set_env_state({'iai_fridge_door_joint': 1.5})

        pose_reached = kitchen_setup.monitors.add_cartesian_pose('map',
                                                                 tip_link=kitchen_setup.tip,
                                                                 goal_pose=current_pose)
        kitchen_setup.monitors.add_end_motion(start_condition=pose_reached)

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
                                              environment_link=handle_name,
                                              goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)

        kitchen_setup.execute(add_local_minimum_reached=False)

        kitchen_setup.set_env_state({'iai_fridge_door_joint': 0})

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.allow_self_collision()
        kitchen_setup.plan_and_execute()

    def test_open_fridge_sequence(self, kitchen_setup: HSRTestWrapper):
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = 'map'
        base_goal.pose.position = Point(0.3, -0.5, 0)
        base_goal.pose.orientation.w = 1
        kitchen_setup.allow_all_collisions()
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1

        # %% phase 1
        bar_grasped = kitchen_setup.monitors.add_distance_to_line(name='bar grasped',
                                                                  root_link=kitchen_setup.default_root,
                                                                  tip_link=kitchen_setup.tip,
                                                                  center_point=bar_center,
                                                                  line_axis=bar_axis,
                                                                  line_length=.4)
        kitchen_setup.motion_goals.add_grasp_bar(root_link=kitchen_setup.default_root,
                                                 tip_link=kitchen_setup.tip,
                                                 tip_grasp_axis=tip_grasp_axis,
                                                 bar_center=bar_center,
                                                 bar_axis=bar_axis,
                                                 bar_length=.4,
                                                 name='grasp bar',
                                                 end_condition=bar_grasped)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.tip
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.motion_goals.add_align_planes(tip_link=kitchen_setup.tip,
                                                    tip_normal=x_gripper,
                                                    goal_normal=x_goal,
                                                    root_link='map',
                                                    name='orient to door',
                                                    end_condition=bar_grasped)

        # %% phase 2 open door
        door_open = kitchen_setup.monitors.add_local_minimum_reached(name='door open',
                                                                     start_condition=bar_grasped)
        kitchen_setup.motion_goals.add_open_container(tip_link=kitchen_setup.tip,
                                                      environment_link=handle_name,
                                                      goal_joint_state=1.5,
                                                      name='open door',
                                                      start_condition=bar_grasped,
                                                      end_condition=door_open)

        kitchen_setup.allow_all_collisions()
        kitchen_setup.monitors.add_end_motion(start_condition=door_open)
        kitchen_setup.execute(add_local_minimum_reached=False)


class TestCollisionAvoidanceGoals:

    def test_self_collision_avoidance_empty(self, zero_pose: HSRTestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_code=GiskardError.EMPTY_PROBLEM)
        current_state = god_map.world.state.to_position_dict()
        current_state = {k.short_name: v for k, v in current_state.items()}
        zero_pose.compare_joint_state(current_state, zero_pose.default_pose)

    def test_self_collision_avoidance(self, zero_pose: HSRTestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.position.z = 0.5
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=r_goal, tip_link=zero_pose.tip, root_link='map')
        zero_pose.plan_and_execute()

    def test_self_collision_avoidance2(self, zero_pose: HSRTestWrapper):
        js = {
            'arm_flex_joint': 0.0,
            'arm_lift_joint': 0.0,
            'arm_roll_joint': -1.52,
            'head_pan_joint': -0.09,
            'head_tilt_joint': -0.62,
            'wrist_flex_joint': -1.55,
            'wrist_roll_joint': 0.11,
        }
        zero_pose.set_seed_configuration(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'hand_palm_link'
        goal_pose.pose.position.x = 0.5
        goal_pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=goal_pose, tip_link=zero_pose.tip, root_link='map')
        zero_pose.plan_and_execute()

    def test_attached_collision1(self, box_setup: HSRTestWrapper):
        box_name = 'asdf'
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'map'
        box_pose.pose.position = Point(0.85, 0.3, .66)
        box_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        box_setup.add_box_to_world(box_name, (0.07, 0.04, 0.1), box_pose)
        box_setup.open_gripper()

        grasp_pose = deepcopy(box_pose)
        # grasp_pose.pose.position.x -= 0.05
        grasp_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                          [0, -1, 0, 0],
                                                                          [1, 0, 0, 0],
                                                                          [0, 0, 0, 1]]))
        box_setup.set_cart_goal(goal_pose=grasp_pose, tip_link=box_setup.tip, root_link='map')
        box_setup.plan_and_execute()
        box_setup.update_parent_link_of_group(box_name, box_setup.tip)

        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x -= 0.5
        base_goal.pose.orientation.w = 1
        box_setup.move_base(base_goal)

    def test_collision_avoidance(self, zero_pose: HSRTestWrapper):
        js = {'arm_flex_joint': -np.pi / 2}
        zero_pose.set_joint_goal(js)
        zero_pose.plan_and_execute()

        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.9
        p.pose.position.y = 0
        p.pose.position.z = 0.5
        p.pose.orientation.w = 1
        zero_pose.add_box_to_world(name='box', size=(1, 1, 0.01), pose=p)

        js = {'arm_flex_joint': 0}
        zero_pose.set_joint_goal(js, add_monitor=False)
        zero_pose.plan_and_execute()


class TestAddObject:
    def test_add(self, zero_pose):
        box1_name = 'box1'
        pose = PoseStamped()
        pose.header.frame_id = zero_pose.default_root
        pose.pose.orientation.w = 1
        pose.pose.position.x = 1
        zero_pose.add_box_to_world(name=box1_name,
                                   size=(1, 1, 1),
                                   pose=pose,
                                   parent_link='hand_palm_link')

        zero_pose.set_joint_goal({'arm_flex_joint': -0.7})
        zero_pose.plan_and_execute()

    def test_grasp_from_table(self, zero_pose: HSRTestWrapper):
        z_offset = 0.3
        x_offset = -1
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'map'
        box_pose.pose.position = Point(2 + x_offset, 0, 0.8 + z_offset)
        box_pose.pose.orientation.w = 1
        zero_pose.add_box_to_world('box', size=(0.2, 0.05, 0.3), pose=box_pose, parent_link='map')

        table_pose = PoseStamped()
        table_pose.header.frame_id = 'map'
        table_pose.pose.position = Point(2 + x_offset, 0, 0.625 + z_offset)
        table_pose.pose.orientation.w = 1
        zero_pose.add_box_to_world('table', size=(0.5, 0.5, 0.05), pose=table_pose, parent_link='map')

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'box'
        goal_pose.pose.position = Point(-0.15, 0, -0.05)
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, 0, 1]]))
        zero_pose.motion_goals.add_cartesian_pose(goal_pose=goal_pose, tip_link='hand_palm_link', root_link='map')

        zero_pose.motion_goals.avoid_all_collisions(min_distance=0.07)
        zero_pose.motion_goals.allow_collision(group1='gripper')
        zero_pose.set_max_traj_length(30)
        zero_pose.execute()

    def test_gpt(self, zero_pose: HSRTestWrapper):
        giskard = zero_pose
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = 'map'
        bowl_pose.pose.orientation.w = 1
        bowl_pose.pose.position = Point(1, 0, 0.4)  # Set bowl position (e.g., on a table)
        giskard.world.add_cylinder(name='bowl', height=0.1, radius=0.15, pose=bowl_pose, parent_link='map')

        # Define cup pose (initially above the bowl)
        # cup_pose = PoseStamped()
        # cup_pose.header.frame_id = 'map'
        # cup_pose.pose.orientation.w = 1
        # cup_pose.pose.position = Point(0, 0, 0.7)  # Set the cup 20 cm above the bowl
        # giskard.world.add_cylinder(name='cup', height=0.15, radius=0.05, pose=cup_pose, parent_link='map')

        # Define the robot's cup feature (position and orientation)
        robot_cup_feature = PointStamped()
        robot_cup_feature.header.frame_id = 'hand_palm_link'
        robot_cup_feature.point = Point(0, 0, 0)

        robot_gripper_z_axis_feature = Vector3Stamped()
        robot_gripper_z_axis_feature.header.frame_id = 'hand_palm_link'
        robot_gripper_z_axis_feature.vector = Vector3(1, 0, 0)

        # Define the world features (center and z-axis of the bowl)
        world_bowl_center_feature = PointStamped()
        world_bowl_center_feature.header.frame_id = 'bowl'
        world_bowl_center_feature.point = Point(0, 0, 0)

        world_bowl_z_axis_feature = Vector3Stamped()
        world_bowl_z_axis_feature.header.frame_id = 'bowl'
        world_bowl_z_axis_feature.vector = Vector3(0, 0, 1)

        # Maintain height between the cup and the bowl (along the z-axis)
        mon_initial_height = giskard.monitors.add_height(
            root_link='map',
            tip_link='hand_palm_link',
            reference_point=world_bowl_center_feature,
            tip_point=robot_cup_feature,
            lower_limit=0.1,  # Maintain 10 cm above the bowl initially
            upper_limit=0.1+0.01
        )
        giskard.motion_goals.add_height(
            root_link='map',
            tip_link='hand_palm_link',
            reference_point=world_bowl_center_feature,
            tip_point=robot_cup_feature,
            lower_limit=0.1,
            upper_limit=0.1,
            end_condition=mon_initial_height
        )

        # Maintain distance between the cup and the bowl in the x-y plane
        mon_distance = giskard.monitors.add_distance(
            root_link='map',
            tip_link='hand_palm_link',
            reference_point=world_bowl_center_feature,
            tip_point=robot_cup_feature,
            lower_limit=0,  # Align cup exactly over the bowl horizontally
            upper_limit=0+0.01
        )
        giskard.motion_goals.add_distance(
            root_link='map',
            tip_link='hand_palm_link',
            reference_point=world_bowl_center_feature,
            tip_point=robot_cup_feature,
            lower_limit=0,
            upper_limit=0,
            end_condition=mon_distance
        )

        # Align the cup's z-axis with the bowl's z-axis for pouring
        mon_align = giskard.monitors.add_vectors_aligned(
            root_link='map',
            tip_link='hand_palm_link',
            goal_normal=world_bowl_z_axis_feature,
            tip_normal=robot_gripper_z_axis_feature
        )
        giskard.motion_goals.add_align_planes(
            root_link='map',
            tip_link='hand_palm_link',
            goal_normal=world_bowl_z_axis_feature,
            tip_normal=robot_gripper_z_axis_feature,
            end_condition=mon_align
        )

        # --- New Motion Goals to Raise Cup ---
        # Monitor to check if the new height (20 cm above the bowl) is met
        mon_new_height = giskard.monitors.add_height(
            root_link='map',
            tip_link='hand_palm_link',
            reference_point=world_bowl_center_feature,
            tip_point=robot_cup_feature,
            lower_limit=0.2-0.01,  # Desired new height 20 cm
            upper_limit=0.2+0.01,
            start_condition=f'{mon_initial_height} and {mon_distance} and {mon_align}'
        )
        giskard.motion_goals.add_height(
            root_link='map',
            tip_link='hand_palm_link',
            reference_point=world_bowl_center_feature,
            tip_point=robot_cup_feature,
            lower_limit=0.2,
            upper_limit=0.2,
            start_condition=f'{mon_initial_height} and {mon_distance} and {mon_align}',
            # Start when initial conditions are met
            end_condition=mon_new_height  # End motion when the cup is 20 cm above the bowl
        )

        # Monitor and execute the motion
        giskard.monitors.add_end_motion(
            start_condition=mon_new_height  # End motion when the new height is reached
        )
        giskard.monitors.add_cancel_motion(
            giskard.monitors.add_local_minimum_reached(),
            error_message='local minimum reached while monitors are not satisfied'
        )

        giskard.motion_goals.allow_all_collisions()
        # Execute the motion
        result = giskard.execute(add_local_minimum_reached=False)
        if result.error.code != GiskardError.SUCCESS:
            print(giskard.get_end_motion_reason(move_result=result, show_all=False))
        else:
            print("Cup successfully positioned above the bowl for pouring.")
