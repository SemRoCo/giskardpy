from copy import deepcopy

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskard_msgs.msg import LinkName
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.motion_statechart.goals.test import GraspSequence, Cutting
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy_ros.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfig, HSRStandaloneInterface
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.god_map import god_map
from giskardpy_ros.utils.utils_for_tests import launch_launchfile
from giskardpy_ros.utils.utils_for_tests import compare_poses, GiskardTestWrapper


class HSRTestWrapper(GiskardTestWrapper):
    default_pose = {
        'arm_flex_joint': -0.03,
        'arm_lift_joint': 0.01,
        'arm_roll_joint': 0.0,
        'head_pan_joint': 0.0,
        'head_tilt_joint': 0.0,
        'wrist_flex_joint': 0.0,
        'wrist_roll_joint': 0.0,
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
                                                                      publish_js=False),
                              qp_controller_config=QPControllerConfig(mpc_dt=0.0125))
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
        self.monitors.add_set_seed_configuration(seed_configuration=js,
                                                 name='move gripper')
        self.execute()

    def reset(self):
        self.register_group(new_group_name='gripper',
                            root_link_name=LinkName(name='hand_palm_link',
                                                    group_name=self.robot_name))


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
        hand_T_finger_current = zero_pose.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
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
        zero_pose.execute()
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
        base_T_torso2 = zero_pose.compute_fk_pose('base_footprint', 'torso_lift_link')
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
        zero_pose.execute()
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
        base_T_torso2 = zero_pose.compute_fk_pose('base_footprint', 'torso_lift_link')
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
        zero_pose.execute()
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
        base_T_torso2 = zero_pose.compute_fk_pose('base_footprint', 'torso_lift_link')
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
        zero_pose.execute()
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
                                       parent_link='hand_palm_link')
        god_map.world.save_graph_pdf(god_map.tmp_folder)

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
        zero_pose.execute()

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
        zero_pose.execute()


class TestConstraints:

    def test_PointingCone(self, zero_pose: HSRTestWrapper):
        tip_link = 'head_center_camera_frame'
        goal_point = PointStamped()
        goal_point.header.frame_id = 'map'
        goal_point.point.x = 0.5
        goal_point.point.y = -0.5
        goal_point.point.z = 1

        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip_link
        pointing_axis.vector.z = 1

        zero_pose.motion_goals.add_motion_goal(class_name=Pointing.__name__,
                                               name='pointy_cone',
                                               tip_link=LinkName(tip_link, ''),
                                               root_link=LinkName('map', ''),
                                               goal_point=goal_point,
                                               pointing_axis=pointing_axis)
        zero_pose.allow_all_collisions()
        zero_pose.add_default_end_motion_conditions()
        zero_pose.execute(add_local_minimum_reached=False)

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
        current_pose = kitchen_setup.compute_fk_pose(root_link='map', tip_link=kitchen_setup.tip)

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
        kitchen_setup.execute()

    def test_open_fridge_sequence_simple(self, kitchen_setup: HSRTestWrapper):
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'
        camera_link = 'head_rgbd_sensor_link'
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

        # %% phase 1 grasp handle
        bar_grasped = kitchen_setup.motion_goals.add_grasp_bar(root_link=kitchen_setup.default_root,
                                                               tip_link=kitchen_setup.tip,
                                                               tip_grasp_axis=tip_grasp_axis,
                                                               bar_center=bar_center,
                                                               bar_axis=bar_axis,
                                                               bar_length=.4,
                                                               name='grasp handle')

        # %% close gripper
        gripper_closed = kitchen_setup.motion_goals.add_joint_position(name='close gripper',
                                                                       goal_state={'hand_motor_joint': 0})
        gripper_opened = kitchen_setup.motion_goals.add_joint_position(name='open gripper',
                                                                       goal_state={'hand_motor_joint': 1.23})

        # %% phase 2 open door
        door_open = kitchen_setup.motion_goals.add_open_container(tip_link=kitchen_setup.tip,
                                                                  environment_link=handle_name,
                                                                  goal_joint_state=1.5,
                                                                  name='open door')

        kitchen_setup.update_end_condition(node_name=bar_grasped, condition=bar_grasped)

        kitchen_setup.update_start_condition(node_name=gripper_closed, condition=bar_grasped)
        kitchen_setup.update_end_condition(node_name=gripper_closed, condition=gripper_closed)

        kitchen_setup.update_start_condition(node_name=door_open, condition=gripper_closed)
        kitchen_setup.update_start_condition(node_name=gripper_opened, condition=f'{door_open}')

        kitchen_setup.update_end_condition(node_name=door_open, condition=f'{door_open}')

        kitchen_setup.allow_all_collisions()
        kitchen_setup.monitors.add_end_motion(start_condition=f'{gripper_opened}')
        kitchen_setup.execute(add_local_minimum_reached=False)


class TestCollisionAvoidanceGoals:

    def test_self_collision_avoidance_empty(self, zero_pose: HSRTestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.execute(expected_error_type=EmptyProblemException, add_local_minimum_reached=False)
        current_state = god_map.world.state.to_position_dict()
        current_state = {k.short_name: v for k, v in current_state.items()}
        zero_pose.compare_joint_state(current_state, zero_pose.default_pose)

    def test_self_collision_avoidance(self, zero_pose: HSRTestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.position.z = 0.5
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=r_goal, tip_link=zero_pose.tip, root_link='map')
        zero_pose.execute()

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
        zero_pose.execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'hand_palm_link'
        goal_pose.pose.position.x = 0.5
        goal_pose.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=goal_pose, tip_link=zero_pose.tip, root_link='map')
        zero_pose.execute()

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
        grasp = box_setup.motion_goals.add_motion_goal(class_name=GraspSequence.__name__,
                                                       name='pick up',
                                                       tip_link=LinkName(name=box_setup.tip),
                                                       root_link='map',
                                                       gripper_joint='hand_motor_joint',
                                                       goal_pose=grasp_pose)
        detected = box_setup.monitors.add_pulse(name='Detect Object',
                                                after_ticks=5)
        success = box_setup.monitors.add_time_above(name='Obj in Hand?',
                                                    threshold=10)
        stop_retry = box_setup.monitors.add_pulse(name='Above 5 Retries',
                                                  after_ticks=100000)

        not_obj_in_hand = f'not {success}'
        box_setup.update_end_condition(node_name=detected, condition=detected)
        box_setup.update_reset_condition(node_name=detected, condition=not_obj_in_hand)

        box_setup.update_start_condition(node_name=grasp, condition=detected)
        box_setup.update_end_condition(node_name=grasp, condition=grasp)
        box_setup.update_reset_condition(node_name=grasp, condition=not_obj_in_hand)

        box_setup.update_start_condition(node_name=success, condition=grasp)
        box_setup.update_end_condition(node_name=success, condition=success)
        box_setup.update_reset_condition(node_name=success, condition=not_obj_in_hand)

        box_setup.update_start_condition(node_name=stop_retry, condition=f'{grasp} and not {success}')
        box_setup.update_reset_condition(node_name=stop_retry, condition=f'not {stop_retry}')

        box_setup.monitors.add_end_motion(start_condition=success)
        box_setup.monitors.add_cancel_motion(start_condition=stop_retry, error=Exception('too many retries'))
        box_setup.allow_all_collisions()
        box_setup.execute(add_local_minimum_reached=False)
        box_setup.update_parent_link_of_group(box_name, box_setup.tip)

        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x -= 0.5
        base_goal.pose.orientation.w = 1
        box_setup.move_base(base_goal)

    def test_schnibbeln_sequence(self, box_setup: HSRTestWrapper):
        box_name = 'Schnibbler'
        box_pose = PoseStamped()
        box_pose.header.frame_id = box_setup.tip
        box_pose.pose.position = Point(0.0, 0.0, 0.06)
        box_pose.pose.orientation.w = 1.0
        bread_name = 'Bernd'
        bread_pose = PoseStamped()
        bread_pose.header.frame_id = 'map'
        bread_pose.pose.position = Point(0.91, 0.25, .62)
        bread_pose.pose.orientation.w = 1.0

        box_setup.add_box_to_world(name=box_name, size=(0.05, 0.01, 0.15), pose=box_pose, parent_link=box_setup.tip)
        box_setup.add_box_to_world(name=bread_name, size=(0.1, 0.2, 0.06), pose=bread_pose, parent_link='box')
        box_setup.dye_group(group_name=box_name, rgba=(0.0, 0.588, 0.784, 1.0))
        box_setup.dye_group(group_name=bread_name, rgba=(0.784, 0.588, 0.0, 1.0))
        box_setup.close_gripper()

        pre_schnibble_pose = PoseStamped()
        pre_schnibble_pose.header.frame_id = 'map'
        pre_schnibble_pose.pose.position = Point(0.85, 0.2, .75)
        pre_schnibble_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                                  [0, -1, 0, 0],
                                                                                  [1, 0, 0, 0],
                                                                                  [0, 0, 0, 1]]))
        pre_schnibble = 'Position Knife'
        box_setup.motion_goals.add_cartesian_pose(name=pre_schnibble,
                                                  goal_pose=pre_schnibble_pose,
                                                  tip_link=box_setup.tip,
                                                  root_link='map',
                                                  end_condition=pre_schnibble)
        human_close = box_setup.monitors.add_pulse(name='Human Close?',
                                                   after_ticks=50,
                                                   true_for_ticks=50,
                                                   start_condition=pre_schnibble,
                                                   end_condition='')

        cut = box_setup.motion_goals.add_motion_goal(class_name=Cutting.__name__,
                                                     name='Cut',
                                                     root_link=LinkName(name='map'),
                                                     tip_link=LinkName(name=box_name),
                                                     depth=0.1,
                                                     right_shift=-0.1,
                                                     start_condition=pre_schnibble)

        # no_contact = box_setup.monitors.add_const_true(name='Made Contact?',
        #                                                start_condition=schnibble_down)

        schnibbel_done = box_setup.monitors.add_time_above(name='Done?',
                                                           threshold=5,
                                                           start_condition=cut)

        reset = f'not {schnibbel_done}'
        box_setup.update_reset_condition(node_name=cut, condition=reset)
        box_setup.update_reset_condition(node_name=schnibbel_done, condition=reset)
        box_setup.update_end_condition(node_name=human_close, condition=schnibbel_done)

        box_setup.update_pause_condition(node_name=cut, condition=human_close)

        box_setup.monitors.add_end_motion(start_condition=schnibbel_done)
        # box_setup.monitors.add_cancel_motion(start_condition=f'not {no_contact}', error=Exception('no contact'))
        box_setup.allow_all_collisions()
        box_setup.execute(add_local_minimum_reached=False)
        # box_setup.update_parent_link_of_group(box_name, box_setup.tip)

    def test_collision_avoidance(self, zero_pose: HSRTestWrapper):
        js = {'arm_flex_joint': -np.pi / 2}
        zero_pose.set_joint_goal(js)
        zero_pose.execute()

        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = 0.9
        p.pose.position.y = 0
        p.pose.position.z = 0.5
        p.pose.orientation.w = 1
        zero_pose.add_box_to_world(name='box', size=(1, 1, 0.01), pose=p)

        js = {'arm_flex_joint': 0}
        zero_pose.set_joint_goal(js, add_monitor=False)
        zero_pose.execute()

    #
    # def test_avoid_collision_touch_hard_threshold(self, box_setup: HSRTestWrapper):
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = box_setup.default_root
    #     base_goal.pose.position.x = 0.2
    #     base_goal.pose.orientation.z = 1
    #     box_setup.teleport_base(base_goal)
    #
    #     box_setup.avoid_collision(min_distance=0.05, group1=box_setup.robot_name)
    #     box_setup.allow_self_collision()
    #
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = 'base_footprint'
    #     base_goal.pose.position.x = -0.3
    #     base_goal.pose.orientation.w = 1
    #     box_setup.set_cart_goal(base_goal, tip_link='base_footprint', root_link='map', weight=WEIGHT_ABOVE_CA)
    #     box_setup.set_max_traj_length(30)
    #     box_setup.execute(add_local_minimum_reached=False)
    #     box_setup.check_cpi_geq(['base_link'], 0.048)
    #     box_setup.check_cpi_leq(['base_link'], 0.07)


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
        zero_pose.execute()
