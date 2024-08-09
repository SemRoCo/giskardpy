from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from numpy import pi
from std_srvs.srv import Trigger
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskardpy.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfig, HSRStandaloneInterface \
    , HSRMujocoVelocityInterface, HSRVelocityInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.configs.behavior_tree_config import StandAloneBTConfig, ClosedLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import compare_poses, GiskardTestWrapper
from giskardpy.goals.manipulability_goals import MaxManipulability
import giskardpy.utils.tfwrapper as tf
from giskardpy.goals.action_goals import PouringAction
from giskardpy.goals.adaptive_goals import CloseGripper, PouringAdaptiveTilt
from neem_interface_python.neem_interface import NEEMInterface


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
                              robot_interface_config=HSRStandaloneInterface(),
                              behavior_tree_config=StandAloneBTConfig(debug_mode=True),
                              qp_controller_config=QPControllerConfig(qp_solver=SupportedQPSolver.qpSWIFT))
        self.gripper_group = 'gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom'
        super().__init__(giskard)
        # self.robot = self.world.groups[self.robot_name]

    def move_base(self, goal_pose):
        self.set_cart_goal(goal_pose, tip_link='base_footprint', root_link='map')
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
        p.header.frame_id = tf.get_tf_root()
        p.pose.orientation.w = 1

    def reset(self):
        super().reset()

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.allow_all_collisions()
        self.move_base(goal_pose)

    def set_localization(self, map_T_odom: PoseStamped):
        pass


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
    default_pose = better_pose

    def __init__(self):
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.mujoco_reset = rospy.ServiceProxy('mujoco/reset', Trigger)
        self.odom_root = 'odom'
        giskard = Giskard(world_config=WorldWithHSRConfig(description_name='/hsrb4s/robot_description'),
                          collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                          robot_interface_config=HSRMujocoVelocityInterface(),
                          behavior_tree_config=ClosedLoopBTConfig(debug_mode=True),
                          qp_controller_config=QPControllerConfig(max_trajectory_length=200,
                                                                  qp_solver=SupportedQPSolver.qpSWIFT))
        # real hsr closed loop config
        # giskard = Giskard(world_config=WorldWithHSRConfig(),
        #                   collision_avoidance_config=HSRCollisionAvoidanceConfig(),
        #                   robot_interface_config=HSRVelocityInterface(),
        #                   behavior_tree_config=ClosedLoopBTConfig())
        super().__init__(giskard)

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.orientation.w = 1
        # self.move_base(p)

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.move_base(goal_pose)

    def set_localization(self, map_T_odom: PoseStamped):
        pass
        # super(HSRTestWrapper, self).set_localization(map_T_odom)

    def reset(self):
        # self.mujoco_reset()
        super().reset()

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


class TestActionGoals:
    def test_pouring_action(self, zero_pose):
        zero_pose.motion_goals.add_motion_goal(motion_goal_class='PouringAction2',
                                               tip_link='hand_palm_link',
                                               root_link='map')
        zero_pose.execute(add_local_minimum_reached=False)

    def test_complete_pouring(self, zero_pose):
        ni = NEEMInterface()
        task_type = 'soma:Pouring'
        env_owl = '/home/huerkamp/workspace/new_giskard_ws/environment.owl'
        env_owl_ind_name = 'world'
        env_urdf = '/home/huerkamp/workspace/new_giskard_ws/new_world_cups.urdf'
        agent_owl = '/home/huerkamp/workspace/new_giskard_ws/src/knowrob/owl/robots/PR2.owl'
        agent_owl_ind_name = 'robotHSR'
        agent_urdf = '/home/huerkamp/workspace/new_giskard_ws/src/mujoco_robots/hsr_robot/hsr_description/hsr_description/robots/hsrb4s.urdf'
        ni.start_episode(task_type=task_type, env_owl=env_owl, env_owl_ind_name=env_owl_ind_name, env_urdf=env_urdf,
                         agent_owl=agent_owl, agent_owl_ind_name=agent_owl_ind_name, agent_urdf=agent_urdf)

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
        zero_pose.motion_goals.add_motion_goal(motion_goal_class='PouringAdaptiveTilt',
                                               name='pouring',
                                               tip='hand_palm_link',
                                               root='map',
                                               tilt_angle=0.3,
                                               pouring_pose=goal_pose,
                                               tilt_axis=tilt_axis,
                                               pre_tilt=False)
        zero_pose.allow_all_collisions()
        zero_pose.avoid_collision(0.01, 'cup1', 'cup2')
        # ni.start_episode(task_type=task_type, env_owl=env_owl, env_owl_ind_name=env_owl_ind_name, env_urdf=env_urdf,
        #                  agent_owl=agent_owl, agent_owl_ind_name=agent_owl_ind_name, agent_urdf=agent_urdf)
        zero_pose.execute(add_local_minimum_reached=False)
        ni.stop_episode('/home/huerkamp/workspace/new_giskard_ws')
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

    def test_align_gripper_to_object(self, zero_pose):
        goal_normal = Vector3Stamped()
        goal_normal.header.frame_id = 'free_cup'
        goal_normal.vector.y = -1
        tip_normal = Vector3Stamped()
        tip_normal.header.frame_id = 'hand_palm_link'
        tip_normal.vector.y = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class='AlignGripperToObject',
                                               name='align',
                                               tip_link='hand_palm_link',
                                               root_link='map',
                                               goal_normal=goal_normal,
                                               tip_normal=tip_normal)
        # publish goal_normal again to /align_goal to change the alignement
        zero_pose.execute(add_local_minimum_reached=False)

    def test_pickup(self, zero_pose):
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
        # add the pickup action
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'hand_palm_link'
        goal_pose.pose.position.x = 0.1
        goal_pose.pose.orientation.w = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class='PickUp',
                                               name='pickup',
                                               root_link='map',
                                               tip_link='hand_palm_link',
                                               goal_pose=goal_pose)
        zero_pose.monitors.add_end_motion(
            start_condition=zero_pose.monitors.add_cartesian_pose('map', 'hand_palm_link', goal_pose))
        zero_pose.execute(add_local_minimum_reached=False)

        zero_pose.motion_goals.add_motion_goal(motion_goal_class='PutDown',
                                               name='putdown')
        zero_pose.execute()

    def test_hand_cam_servo(self, zero_pose: HSRTestWrapperMujoco):
        optical_axis = Vector3Stamped()
        optical_axis.header.frame_id = 'hand_palm_link'
        optical_axis.vector.z = 1
        zero_pose.motion_goals.add_motion_goal(motion_goal_class='HandCamServoGoal',
                                               name='hnadcamservo',
                                               root_link='map',
                                               cam_link='hand_palm_link',
                                               optical_axis=optical_axis,
                                               transform_from_image_coordinates=True)
        zero_pose.execute(add_local_minimum_reached=False)

    def test_abhijit(self, zero_pose):
        def openGripper():
            zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                                   name='openGripper',
                                                   as_open=True,
                                                   velocity_threshold=100,
                                                   effort_threshold=1,
                                                   effort=100)
            zero_pose.allow_all_collisions()
            zero_pose.execute()

        def closeGripper():
            zero_pose.motion_goals.add_motion_goal(motion_goal_class=CloseGripper.__name__,
                                                   name='closeGripper')
            zero_pose.allow_all_collisions()
            zero_pose.execute()

        def align_to(side: str, axis_align_to_z: Vector3Stamped, object_frame: str, control_frame: str,
                     axis_align_to_x: Vector3Stamped = None, distance=0.3, height_offset=0.0, second_distance=0.0):
            # TODO: add a tilt method with angle and velocity parametrization
            goal_normal = Vector3Stamped()
            goal_normal.header.frame_id = object_frame
            goal_normal.vector.z = 1
            zero_pose.motion_goals.add_align_planes(goal_normal, control_frame, axis_align_to_z, 'map')
            if second_axis:
                second_goal_normal = Vector3Stamped()
                second_goal_normal.header.frame_id = object_frame
                second_goal_normal.vector.x = 1
                zero_pose.motion_goals.add_align_planes(second_goal_normal, control_frame, axis_align_to_x, 'map')

            goal_position = PointStamped()
            goal_position.header.frame_id = object_frame
            if side == 'front':
                goal_position.point.x = -distance
                goal_position.point.y = second_distance
                goal_position.point.z = height_offset
            elif side == 'left':
                goal_position.point.x = second_distance
                goal_position.point.y = distance
                goal_position.point.z = height_offset
            elif side == 'right':
                goal_position.point.x = second_distance
                goal_position.point.y = -distance
                goal_position.point.z = height_offset
            zero_pose.motion_goals.add_cartesian_position(goal_position, control_frame, 'map')
            zero_pose.execute()

        def tilt(angle: float, velocity: float, rotation_axis: Vector3Stamped, controlled_frame: str):
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = controlled_frame
            goal_pose.pose.orientation = Quaternion(
                *quaternion_about_axis(angle, [rotation_axis.vector.x, rotation_axis.vector.y, rotation_axis.vector.z]))
            zero_pose.motion_goals.add_cartesian_pose(goal_pose, controlled_frame, 'map')
            zero_pose.motion_goals.add_limit_cartesian_velocity(tip_link=controlled_frame, root_link='map',
                                                                max_angular_velocity=velocity)
            zero_pose.execute()

        upright_axis = Vector3Stamped()
        upright_axis.header.frame_id = 'hand_palm_link'
        upright_axis.vector.x = 1

        second_axis = Vector3Stamped()
        second_axis.header.frame_id = 'hand_palm_link'
        second_axis.vector.z = 1

        # Here starts the control
        openGripper()

        align_to('front', axis_align_to_z=upright_axis, object_frame='free_cup', control_frame='hand_palm_link',
                 axis_align_to_x=second_axis, distance=0.04)

        closeGripper()

        #############move to random position################
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 0, 1, 0],
                                                                         [0, -1, 0, 0],
                                                                         [1, 0, 0, 0],
                                                                         [0, 0, 0, 1]]))
        goal_pose.pose.position.x = 1.7
        goal_pose.pose.position.y = -0.4
        goal_pose.pose.position.z = 0.7
        zero_pose.set_cart_goal(goal_pose, 'hand_palm_link', 'map')
        zero_pose.allow_all_collisions()
        zero_pose.execute()
        #############################

        # align_to('front', axis_align_to_z=upright_axis, object_frame='free_cup2', control_frame='hand_palm_link',
        #          axis_align_to_x=second_axis, distance=0.3, height_offset=0.2)

        align_to('left', axis_align_to_z=upright_axis, object_frame='free_cup2', control_frame='hand_palm_link',
                 axis_align_to_x=second_axis, distance=0.13, height_offset=0.2)

        rotation_axis = Vector3Stamped()
        rotation_axis.header.frame_id = 'hand_palm_link'
        rotation_axis.vector.z = 1
        tilt(angle=1.7, velocity=1.0, rotation_axis=rotation_axis, controlled_frame='hand_palm_link')
