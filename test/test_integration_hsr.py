import time
from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped, QuaternionStamped
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig, ClosedLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfig, WorldWithHSRConfigMujoco, \
    HSRStandaloneInterface, HSRMujocoVelocityInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import compare_poses, GiskardTestWrapper
import rospy
from std_srvs.srv import Trigger
from giskardpy.hand_model import Hand, Finger
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA


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
        self.robot_name = 'hsr'
        if giskard is None:
            giskard = Giskard(world_config=WorldWithHSRConfig(),
                              collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                              robot_interface_config=HSRStandaloneInterface(),
                              behavior_tree_config=StandAloneBTConfig(),
                              qp_controller_config=QPControllerConfig())
        super().__init__(giskard)
        self.gripper_group = 'gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom'
        self.robot = self.world.groups[self.robot_name]

    def move_base(self, goal_pose):
        self.set_cart_goal(goal_pose, tip_link='base_footprint', root_link=self.world.root_link_name)
        self.plan_and_execute()

    def open_gripper(self):
        self.command_gripper(1.24)

    def close_gripper(self):
        self.command_gripper(0)

    def command_gripper(self, width):
        js = {'hand_motor_joint': width}
        self.set_joint_goal(js)
        self.plan_and_execute()

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        if self.is_standalone():
            self.teleport_base(p)
        else:
            self.move_base(p)

    def reset(self):
        self.clear_world()
        # self.close_gripper()
        self.reset_base()
        self.register_group('gripper',
                            root_link_group_name=self.robot_name,
                            root_link_name='hand_palm_link')

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.set_seed_odometry(base_pose=goal_pose, group_name=group_name)
        self.allow_all_collisions()
        self.plan_and_execute()


class HSRTestWrapperMujoco(HSRTestWrapper):
    better_pose = {
        'arm_flex_joint': -0.7,
        'arm_lift_joint': 0.2,
        'arm_roll_joint': 0.0,
        'head_pan_joint': -0.1,
        'head_tilt_joint': 0.1,
        'wrist_flex_joint': -0.9,
        'wrist_roll_joint': -0.4,
    }

    def __init__(self, giskard=None):
        self.mujoco_reset = rospy.ServiceProxy('mujoco/reset', Trigger)
        if giskard is None:
            giskard = Giskard(world_config=WorldWithHSRConfigMujoco(),
                              collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                              robot_interface_config=HSRMujocoVelocityInterface(),
                              behavior_tree_config=ClosedLoopBTConfig(),
                              qp_controller_config=QPControllerConfig())
        super().__init__(giskard)

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        # self.move_base(p)

    def set_localization(self, map_T_odom: PoseStamped):
        pass
        # super(HSRTestWrapper, self).set_localization(map_T_odom)

    def reset(self):
        self.clear_world()
        # self.close_gripper()
        self.reset_base()

    def command_gripper(self, width):
        pass


@pytest.fixture(scope='module')
def giskard(request, ros):
    # launch_launchfile('package://hsr_description/launch/upload_hsrb.launch')
    # c = HSRTestWrapper()
    c = HSRTestWrapperMujoco()
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
    zero_pose.add_box(name='box', size=(1, 1, 1), pose=p)
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
        kitchen_setup.add_box(name=box1_name,
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
        r_goal.pose.position.x = 2
        r_goal.pose.orientation = Quaternion(*quaternion_about_axis(pi, [0, 0, 1]))
        zero_pose.set_cart_goal(r_goal, zero_pose.tip, 'map')
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
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 1.5})

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.tip,
                                              environment_link=handle_name,
                                              goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.plan_and_execute()
        kitchen_setup.set_kitchen_js({'iai_fridge_door_joint': 0})

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

        box_setup.add_box(box_name, (0.07, 0.04, 0.1), box_pose)
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
        zero_pose.add_box(name='box', size=(1, 1, 0.01), pose=p)

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
        zero_pose.add_box(name=box1_name,
                          size=(1, 1, 1),
                          pose=pose,
                          parent_link='hand_palm_link')

        zero_pose.set_joint_goal({'arm_flex_joint': -0.7})
        zero_pose.plan_and_execute()

    def test_add_and_delete(self, zero_pose: HSRTestWrapper):
        box1_name = 'box'
        pose = PoseStamped()
        pose.header.frame_id = zero_pose.default_root
        pose.pose.orientation.w = 1
        pose.pose.position.x = 3
        pose.pose.position.z = 1

        zero_pose.add_box(box1_name, (0.2, 0.2, 0.2), pose)

        arm_pose = PoseStamped()
        arm_pose.header.frame_id = box1_name
        arm_pose.pose.position.x = -0.5

        zero_pose.set_cart_goal(arm_pose, 'hand_palm_link', 'map')
        zero_pose.plan_and_execute()


class TestPouring:
    def test_pouring2(self, better_pose):
        better_pose.update_parent_link_of_group('sync_create_cup2233', 'hand_palm_link', 'hsrb4s')

    def test_pouring(self, better_pose):
        # TODO: there seems to be a bug in the mujoco sync. it slows down start up and the base is not moved
        # in the pouring goal. Somehow the plugins even cause QP not solvable errors in other test that don't rely on
        # them, and avoid base moevemnt there
        zero_pose = better_pose
        containerPose = PoseStamped()
        containerPose.header.frame_id = 'map'
        containerPose.pose.position.x = 2
        containerPose.pose.position.y = -0.2
        containerPose.pose.position.z = 0.3
        containerPose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                             [0, 1, 0, 0],
                                                                             [0, 0, 1, 0],
                                                                             [0, 0, 0, 1]]))
        cupPose = PoseStamped()
        cupPose.header.frame_id = 'hand_palm_link'
        cupPose.pose.position.x = -0.02
        cupPose.pose.position.y = 0
        cupPose.pose.position.z = 0.1
        cupPose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                       [0, -1, 0, 0],
                                                                       [1, 0, 0, 0],
                                                                       [0, 0, 0, 1]]))
        edgePose = PoseStamped()
        edgePose.header.frame_id = 'cup'
        edgePose.pose.position.x = 0
        edgePose.pose.position.y = 0.03
        edgePose.pose.position.z = 0.1
        edgePose.pose.orientation = Quaternion(*quaternion_from_matrix([[1, 0, 0, 0],
                                                                        [0, 1, 0, 0],
                                                                        [0, 0, 1, 0],
                                                                        [0, 0, 0, 1]]))

        zero_pose.add_box('container', (0.06, 0.06, 0.19), containerPose)
        # zero_pose.add_mesh('container', mesh='package://giskardpy/test/urdfs/meshes/bowl_21.obj',
        #                    pose=containerPose)
        zero_pose.add_cylinder('cup', 0.19, 0.03, cupPose, parent_link=cupPose.header.frame_id)
        zero_pose.add_box('edge', (0.02, 0.01, 0.01), edgePose, 'cup', 'cup')

        # held object
        object_link = 'cup'
        # object axis for keep upright
        object_axis = Vector3Stamped()
        object_axis.header.frame_id = object_link
        object_axis.vector.z = 1

        # goal object and keep above parameter
        container_plane = PointStamped()
        container_plane.header.frame_id = 'container'  # 'sync_bowl1/sync_bowl1'
        container_plane.point.y = -0.06
        lower_distance = 0.25
        upper_distance = 0.25
        plane_radius = 0.0

        # reference axis for keep upright
        reference_axis = Vector3Stamped()
        reference_axis.header.frame_id = 'map'
        reference_axis.vector.z = 1

        reference_axis2 = Vector3Stamped()
        reference_axis2.header.frame_id = 'map'
        reference_axis2.vector.x = 1

        # rotation axis for tilting
        rotation_axis = Vector3Stamped()
        rotation_axis.header.frame_id = 'edge'
        rotation_axis.vector.x = 1
        tilt_angle = -1.9
        tilt_velocity = 0.8

        # zero_pose.update_parent_link_of_group('sync_create_cup2233', 'hand_palm_link', 'hsrb4s')
        # First phase KeepObjectUpright & KeepObjectAbovePlane
        # zero_pose.set_json_goal('KeepObjectAbovePlane',
        #                         object_link='cup',  # 'sync_create_cup2223/sync_create_cup2223',
        #                         plane_center_point=container_plane,
        #                         lower_distance=lower_distance,
        #                         upper_distance=upper_distance,
        #                         plane_radius=plane_radius,
        #                         root_link='map')
        #
        # zero_pose.set_json_goal('KeepObjectUpright',
        #                         object_link_axis=object_axis,
        #                         reference_link_axis=reference_axis,
        #                         root_link='map')
        # # align x planes of object and map
        # zero_pose.set_json_goal('KeepObjectUpright',
        #                         object_link_axis=rotation_axis,
        #                         reference_link_axis=reference_axis2,
        #                         root_link='map')
        goal_pose = containerPose
        goal_pose.pose.position.y -= 0.12
        goal_pose.pose.position.z += 0.1
        zero_pose.set_cart_goal(goal_pose=goal_pose,
                                tip_link='cup',  # sync_create_cup2233/sync_create_cup2233',
                                root_link='map')

        # zero_pose.set_avoid_joint_limits_goal(10)
        # zero_pose.add_cmd()
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        # # Second phase TiltObject & KeepObjectAbovePlane
        goal_point = PointStamped()
        goal_point.header.frame_id = 'map'
        goal_point.point.x = 2
        goal_point.point.y = -0.26
        goal_point.point.z = 0.42
        zero_pose.set_translation_goal(goal_point=goal_point,
                                       tip_link='edge',
                                       root_link='map',
                                       max_velocity=0.2)
        # zero_pose.set_json_goal('KeepObjectAbovePlane',
        #                         object_link='edge',# 'sync_create_cup2223/sync_create_cup2223',
        #                         plane_center_point=container_plane,
        #                         lower_distance=0.1,
        #                         upper_distance=0.1,
        #                         plane_radius=plane_radius,
        #                         root_link='map')

        zero_pose.set_json_goal('TiltObject',
                                object_link='edge',
                                reference_link='map',
                                rotation_velocity=tilt_velocity,
                                root_link='map',
                                lower_angle=tilt_angle,
                                rotation_axis=rotation_axis)
        # align x planes of object and map
        # zero_pose.set_json_goal('KeepObjectUpright',
        #                         object_link_axis=rotation_axis,
        #                         reference_link_axis=reference_axis2,
        #                         root_link='map')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        # zero_pose.add_cmd()
        #
        # # # Third phase KeepObjectUpright & KeepObjectAbovePlane
        # zero_pose.set_json_goal('KeepObjectAbovePlane',
        #                         object_link='sync_create_cup2223/sync_create_cup2223',
        #                         plane_center_point=container_plane,
        #                         lower_distance=lower_distance,
        #                         upper_distance=upper_distance,
        #                         plane_radius=plane_radius,
        #                         root_link='map')
        # zero_pose.set_json_goal('KeepObjectUpright',
        #                         object_link_axis=object_axis,
        #                         reference_link_axis=reference_axis,
        #                         root_link='map')
        # # align x planes of object and map
        # zero_pose.set_json_goal('KeepObjectUpright',
        #                         object_link_axis=rotation_axis,
        #                         reference_link_axis=reference_axis2,
        #                         root_link='map')
        #
        # zero_pose.plan_and_execute()


class TestActions:
    def test_pouring_action(self, zero_pose: HSRTestWrapper):
        # TODO: add move to constraints and make the others more stable
        orientation = QuaternionStamped()
        orientation.header.frame_id = 'map'
        orientation.quaternion = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                     [0, -1, 0, 0],
                                                                     [1, 0, 0, 0],
                                                                     [0, 0, 0, 1]]))

        down_orientation = QuaternionStamped()
        down_orientation.header.frame_id = 'map'
        down_orientation.quaternion = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                          [0, 1, 0, 0],
                                                                          [-1, 0, 0, 0],
                                                                          [0, 0, 0, 1]]))

        container_plane = PointStamped()
        container_plane.header.frame_id = 'map'
        container_plane.point.x = 2
        container_plane.point.z = 1

        zero_pose.set_json_goal('PouringAction',
                                tip_link='hand_palm_link',
                                root_link='map',
                                upright_orientation=orientation,
                                down_orientation=down_orientation,
                                container_plane=container_plane,
                                tilt_joint='wrist_roll_joint')
        zero_pose.set_avoid_joint_limits_goal()
        zero_pose.plan_and_execute()

    def test_adaptive_pouring(self, zero_pose):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = 1.9
        goal_pose.pose.position.y = - 0.2
        goal_pose.pose.position.z = 0.65
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, 0, 1]]))
        zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        zero_pose.set_joint_goal({'arm_flex_joint': -0.8}, weight=WEIGHT_ABOVE_CA)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()

        zero_pose.set_json_goal('PouringAdaptiveTilt',
                                tip='hand_palm_link',
                                root='map',
                                tilt_angle=-1.5)
        goal_point = PointStamped()
        goal_point.header.frame_id = 'map'
        goal_point.point = goal_pose.pose.position
        zero_pose.set_json_goal('CartesianPosition',
                                root_link='map',
                                tip_link='hand_palm_link',
                                goal_point=goal_point)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()


class TestGrasping:
    def test_gripper(self, better_pose):
        # range="-0.798 1.24"
        better_pose.set_joint_goal({'hand_motor_joint': 1})
        better_pose.plan_and_execute()
        # Do Giang or Simon have some best practices for controlling the gripper in mujoco sim?

    def test_box_grasp(self, better_pose: HSRTestWrapper):
        better_pose2 = {
            'arm_flex_joint': -0.7,
            'arm_lift_joint': 0.2,
            'arm_roll_joint': 0.0,
            'head_pan_joint': -0.1,
            'head_tilt_joint': 0.1,
            'wrist_flex_joint': -0.9,
            'wrist_roll_joint': -0.4,
        }

        hand = Hand(hand_tool_frame='hsrb4s/hand_tool_frame',
                    palm_link='hsrb4s/hand_palm_link',
                    thumb=Finger(tip_tool_frame='hsrb4s/thumb_tool_frame',
                                 collision_links=['hsrb4s/hand_l_distal_link',
                                                  'hsrb4s/hand_l_proximal_link']),
                    fingers=[Finger(tip_tool_frame='hsrb4s/finger_tool_frame',
                                    collision_links=['hsrb4s/hand_l_distal_link',
                                                     'hsrb4s/hand_l_proximal_link'])
                             ],
                    finger_js={'hand_motor_joint': 0.7},
                    opening_width=0.06)

        box_pose = PoseStamped()
        box_pose.header.frame_id = 'map'
        box_pose.pose.position.z = 0.3
        box_pose.pose.position.x = 2
        box_pose.pose.position.y = 0
        box_pose.pose.orientation.w = 1
        better_pose.add_box('box', (0.03, 0.2, 0.3), box_pose)

        better_pose.set_json_goal('GraspBox',
                                  hand=hand,
                                  object_name='box',
                                  root_link='map',
                                  blocked_directions=[0, 0, 0, 0, 0, 0],
                                  group='hsrb4s',
                                  max_linear_velocity=0.2)

        view = 'hsrb4s/head_center_camera_frame'
        goal_point = PointStamped()
        goal_point.point = box_pose.pose.position
        pointingAxis = Vector3Stamped()
        pointingAxis.header.frame_id = view
        pointingAxis.vector.z = 1
        better_pose.set_pointing_goal(goal_point, view, pointingAxis, 'map')

        better_pose.allow_all_collisions()
        better_pose.plan_and_execute()


class TestServo:
    def test_point_feature(self, zero_pose):
        goal_point = PointStamped()
        goal_point.header.frame_id = 'map'
        goal_point.point.x = 1
        goal_point.point.y = 0.5
        goal_point.point.z = 0.8

        zero_pose.set_json_goal('VisualServoPointGoal',
                                root='map',
                                tip='hand_palm_link',
                                goal_point=goal_point)

        goal_vector = Vector3Stamped()
        goal_vector.header.frame_id = 'map'
        goal_vector.vector.y = 1
        zero_pose.set_json_goal('VisualServoVectorGoal',
                                root='map',
                                tip='hand_palm_link',
                                goal_vector=goal_vector)

        tip_normal = Vector3Stamped()
        tip_normal.header.frame_id = 'hand_palm_link'
        tip_normal.vector.x = 1

        goal_normal = Vector3Stamped()
        goal_normal.header.frame_id = 'map'
        goal_normal.vector.z = 1

        zero_pose.set_align_planes_goal(tip_link='hand_palm_link',
                                        tip_normal=tip_normal,
                                        root_link='map',
                                        goal_normal=goal_normal)
        zero_pose.set_joint_goal({'arm_flex_joint': -1})
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
