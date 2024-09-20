from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from numpy import pi
from std_srvs.srv import Trigger
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig, ClosedLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.hsr import WorldWithHSRConfig, HSRCollisionAvoidanceConfig, HSRStandaloneInterface, \
    HSRMujocoVelocityInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.god_map import god_map
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import compare_poses, GiskardTestWrapper
from giskardpy.goals.adaptive_goals import CloseGripper, PouringAdaptiveTilt
from giskard_msgs.msg import GiskardError


class HSRTestWrapper(GiskardTestWrapper):
    default_pose = {
        'arm_flex_joint': 0.0,
        'arm_lift_joint': 0.0,
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
                              robot_interface_config=HSRMujocoVelocityInterface(),
                              behavior_tree_config=ClosedLoopBTConfig(debug_mode=True),
                              qp_controller_config=QPControllerConfig())
        super().__init__(giskard)
        self.gripper_group = 'gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom'
        self.robot = god_map.world.groups[self.robot_name]

    def open_gripper(self):
        self.command_gripper(1.24)

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
        arm_lift_joint = zero_pose.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()
        hand_T_finger_current = zero_pose.world.compute_fk_pose('hand_palm_link', 'hand_l_distal_link')
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
        zero_pose.set_joint_goal(js, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state[arm_lift_joint].position, 0.2, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = zero_pose.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints2(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = zero_pose.world.search_for_joint_name('arm_lift_joint')
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
        np.testing.assert_almost_equal(zero_pose.world.state[arm_lift_joint].position, 0.2, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = zero_pose.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints3(self, zero_pose: HSRTestWrapper):
        arm_lift_joint = zero_pose.world.search_for_joint_name('arm_lift_joint')
        zero_pose.open_gripper()
        tip = 'head_pan_link'
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.15
        p.pose.orientation.w = 1
        zero_pose.set_cart_goal(goal_pose=p, tip_link=tip,
                                root_link='base_footprint')
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state[arm_lift_joint].position, 0.3, decimal=2)
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = 'base_footprint'
        base_T_torso.pose.position.x = 0
        base_T_torso.pose.position.y = 0
        base_T_torso.pose.position.z = 0.902
        base_T_torso.pose.orientation.x = 0
        base_T_torso.pose.orientation.y = 0
        base_T_torso.pose.orientation.z = 0
        base_T_torso.pose.orientation.w = 1
        base_T_torso2 = zero_pose.world.compute_fk_pose('base_footprint', 'torso_lift_link')
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints4(self, zero_pose: HSRTestWrapper):
        ll, ul = zero_pose.world.get_joint_velocity_limits('hsrb/arm_lift_joint')
        assert ll == -0.2
        assert ul == 0.2
        ll, ul = zero_pose.world.get_joint_velocity_limits('hsrb/torso_lift_joint')
        assert ll == -0.1
        assert ul == 0.1
        joint_goal = {'torso_lift_joint': 0.25}
        zero_pose.set_joint_goal(joint_goal, check=False)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        np.testing.assert_almost_equal(zero_pose.world.state['hsrb/arm_lift_joint'].position, 0.5, decimal=2)


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
        kitchen_setup.world.save_graph_pdf()

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
        zero_pose.set_cart_goal(base_goal, 'base_footprint')
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
        zero_pose.set_cart_goal(r_goal, zero_pose.tip)
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

        kitchen_setup.set_json_goal('GraspBar',
                                    root_link=kitchen_setup.default_root,
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
                                            goal_normal=x_goal)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.plan_and_execute()

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
                                              environment_link=handle_name,
                                              goal_joint_state=1.5)
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_env_state({'iai_fridge_door_joint': 1.5})

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
                                              environment_link=handle_name,
                                              goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_env_state({'iai_fridge_door_joint': 0})

        kitchen_setup.plan_and_execute()

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.allow_self_collision()
        kitchen_setup.plan_and_execute()


class TestCollisionAvoidanceGoals:

    def test_self_collision_avoidance_empty(self, zero_pose: HSRTestWrapper):
        zero_pose.plan_and_execute()
        current_state = zero_pose.world.state.to_position_dict()
        current_state = {k.short_name: v for k, v in current_state.items()}
        zero_pose.compare_joint_state(current_state, zero_pose.default_pose)

    def test_self_collision_avoidance(self, zero_pose: HSRTestWrapper):
        r_goal = PoseStamped()
        r_goal.header.frame_id = zero_pose.tip
        r_goal.pose.position.z = 0.5
        r_goal.pose.orientation.w = 1
        zero_pose.set_cart_goal(r_goal, zero_pose.tip)
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
        zero_pose.set_cart_goal(goal_pose, zero_pose.tip)
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
        box_setup.set_cart_goal(grasp_pose, box_setup.tip)
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
        zero_pose.set_joint_goal(js, check=False)
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
                                   parent_link='hand_palm_link',
                                   parent_link_group='hsrb4s')

        zero_pose.set_joint_goal({'arm_flex_joint': -0.7})
        zero_pose.plan_and_execute()


class TestAdaptiveGoals:
    def test_complete_pouring(self, zero_pose):
        zero_pose.set_avoid_name_conflict(False)
        # first start related scripts for BB detection and scene action reasoning
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='openGripper',
                                               as_open=True,
                                               velocity_threshold=100,
                                               effort_threshold=1,
                                               effort=100)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 1.95
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.3

        zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='closeGripper', effort=-220)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        cup_pose = PoseStamped()
        cup_pose.header.frame_id = 'free_cup'
        cup_pose.pose.position = Point(0, 0, 0)
        cup_pose.pose.orientation.w = 1

        # add a new object at the pose of the pot and attach it to the right tip
        zero_pose.add_box('cup1', (0.07, 0.07, 0.28), pose=cup_pose, parent_link='hand_palm_link')
        cup_pose.header.frame_id = 'free_cup2'
        zero_pose.add_box('cup2', (0.07, 0.07, 0.18), pose=cup_pose, parent_link='map')

        # goal_pose.pose.position.x = 1.85
        # goal_pose.pose.position.y = -0.7
        # goal_pose.pose.position.z = 0.54
        # goal_pose.pose.position.x = 1.75
        # goal_pose.pose.position.y = -0.4
        # goal_pose.pose.position.z = 0.6
        goal_pose.header.frame_id = 'cup2'
        goal_pose.pose.position = Point(-0.3, 0.2, 0.3)
        # goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
        #                                                                  [0, -1, 0, 0],
        #                                                                  [1, 0, 0, 0],
        #                                                                  [0, 0, 0, 1]]))
        tilt_axis = Vector3Stamped()
        tilt_axis.header.frame_id = 'hand_palm_link'
        tilt_axis.vector.z = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class=PouringAdaptiveTilt.__name__,
                                               name='Pouring',
                                               tip='hand_palm_link',
                                               root='map',
                                               tilt_angle=0.3,
                                               pouring_pose=goal_pose,
                                               tilt_axis=tilt_axis,
                                               pre_tilt=False)
        zero_pose.allow_all_collisions()
        zero_pose.avoid_collision(0.01, 'cup1', 'cup2')
        zero_pose.set_max_traj_length(30)
        zero_pose.execute(add_local_minimum_reached=False)

        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = 1.93
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.3

        zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                               name='openGripper',
                                               as_open=True,
                                               velocity_threshold=100,
                                               effort_threshold=1,
                                               effort=100)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        goal_pose.pose.position.x = 1.4
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.4

        zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()

    def test_mixing(self, zero_pose):
        zero_pose.set_avoid_name_conflict(False)
        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
        #                                        name='openGripper',
        #                                        as_open=True,
        #                                        velocity_threshold=100,
        #                                        effort_threshold=1,
        #                                        effort=100)
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 1.95
        goal_pose.pose.position.y = -0.6
        goal_pose.pose.position.z = 0.35

        # zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()

        # zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
        #                                        name='closeGripper', effort=-220)
        # zero_pose.allow_all_collisions()
        # zero_pose.execute()

        goal_pose.pose.position.z = 0.5
        zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()

        #######
        robot_feature = PointStamped()
        robot_feature.header.frame_id = 'hand_palm_link'
        robot_feature.point = Point(0, 0, 0)

        robot_up_axis = Vector3Stamped()
        robot_up_axis.header.frame_id = 'hand_palm_link'
        robot_up_axis.vector.x = 1

        robot_pointing_axis = Vector3Stamped()
        robot_pointing_axis.header.frame_id = 'hand_palm_link'
        robot_pointing_axis.vector.z = 1

        # Define the world features (center and z-axis of the bowl)

        world_feature2 = PointStamped()
        world_feature2.header.frame_id = 'map'
        world_feature2.point = Point(2, -0.2, 0.29)

        world_up_axis = Vector3Stamped()
        world_up_axis.header.frame_id = 'map'
        world_up_axis.vector.z = 1

        start_spiral = PointStamped()
        start_spiral.header.frame_id = 'map'
        start_spiral.point = Point(2, -0.2, 0.45)

        seq = [
            [
                ['distance', world_feature2, robot_feature, 0.00, 0.02],
                ['height', world_feature2, robot_feature, 0.25, 0.30],
                ['align', world_up_axis, robot_up_axis]
            ],
            [
                ['height', world_feature2, robot_feature, 0.15, 0.16],
                ['mixing', start_spiral, world_up_axis, 0.05, 25],
                ['angle', world_up_axis, robot_up_axis, 0.0, 0.4],
                ['force', 2, '/mujoco/sensors_3D']
            ],
            [
                ['distance', world_feature2, robot_feature, 0.00, 0.05],
                ['height', world_feature2, robot_feature, 0.25, 0.3],
                ['align', world_up_axis, robot_up_axis]
            ],
        ]

        zero_pose.create_tcmp_controller(tip_link='hand_palm_link', root_link='map',
                                         sequence=seq, max_vel=0.2)

        zero_pose.set_max_traj_length(60)
        zero_pose.execute(add_local_minimum_reached=False)

    def test_force(self, zero_pose):

        #######
        robot_feature = PointStamped()
        robot_feature.header.frame_id = 'hand_palm_link'
        robot_feature.point = Point(0, 0, 0)

        robot_up_axis = Vector3Stamped()
        robot_up_axis.header.frame_id = 'hand_palm_link'
        robot_up_axis.vector.x = 1

        robot_pointing_axis = Vector3Stamped()
        robot_pointing_axis.header.frame_id = 'hand_palm_link'
        robot_pointing_axis.vector.z = 1

        # Define the world features (center and z-axis of the bowl)

        world_feature2 = PointStamped()
        world_feature2.header.frame_id = 'map'
        world_feature2.point = Point(2, -0.6, 0.3)

        world_up_axis = Vector3Stamped()
        world_up_axis.header.frame_id = 'map'
        world_up_axis.vector.z = 1

        start_spiral = PointStamped()
        start_spiral.header.frame_id = 'map'
        start_spiral.point = Point(2, -0.2, 0.49)

        seq = [
            [
                ['distance', world_feature2, robot_feature, 0.00, 0.02],
                ['height', world_feature2, robot_feature, 0.0, 0.01],
                ['align', world_up_axis, robot_up_axis],
                ['force', 10, '/mujoco/sensors_3D']
            ]
        ]

        zero_pose.create_tcmp_controller(tip_link='hand_palm_link', root_link='map',
                                         sequence=seq, max_vel=0.2)

        zero_pose.set_max_traj_length(60)
        zero_pose.execute(add_local_minimum_reached=False)
