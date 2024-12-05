from copy import deepcopy
from typing import Optional

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from numpy import pi
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

from giskard_msgs.msg import LinkName, GiskardError
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.motion_graph.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.utils.math import quaternion_from_rotation_matrix
from giskardpy_ros.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfig, HSRStandaloneInterface
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.god_map import god_map
from giskardpy_ros.configs.other_robots.justin import WorldWithJustinConfig, JustinStandaloneInterface, \
    JustinCollisionAvoidanceConfig
from giskardpy_ros.ros1.visualization_mode import VisualizationMode
from utils_for_tests import launch_launchfile
from utils_for_tests import compare_poses, GiskardTestWrapper


class JustinTestWrapper(GiskardTestWrapper):
    default_pose = {
        "torso1_joint": 0.0,
        "torso2_joint": -0.9,
        "torso3_joint": 1.26,
        "head1_joint": 0.0,
        "head2_joint": 0.0,
        "left_arm1_joint": 0.41,
        "left_arm2_joint": -1.64,
        "left_arm3_joint": 0.12,
        "left_arm4_joint": 0.96,
        "left_arm5_joint": 0.71,
        "left_arm6_joint": -0.02,
        "left_arm7_joint": 0.43,
        "right_arm1_joint": 0.6,
        "right_arm2_joint": -1.59,
        "right_arm3_joint": 2.97,
        "right_arm4_joint": -0.99,
        "right_arm5_joint": -2.44,
        "right_arm6_joint": 0.0,
        "right_arm7_joint": 0.0,
    }

    better_pose = default_pose

    def __init__(self, giskard=None):
        self.r_tip = 'r_gripper_tool_frame'
        self.l_tip = 'l_gripper_tool_frame'
        if giskard is None:
            giskard = Giskard(world_config=WorldWithJustinConfig(),
                              collision_avoidance_config=JustinCollisionAvoidanceConfig(),
                              robot_interface_config=JustinStandaloneInterface(),
                              behavior_tree_config=StandAloneBTConfig(publish_tf=True, debug_mode=True,
                                                                      visualization_mode=VisualizationMode.VisualsFrameLocked),
                              qp_controller_config=QPControllerConfig(mpc_dt=0.0125,
                                                                      control_dt=0.0125))
        super().__init__(giskard)
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.odom_root = 'odom'
        self.robot = god_map.world.groups[self.robot_name]

    def reset(self):
        pass


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://iai_dlr_rollin_justin/launch/justin_upload.launch')
    c = JustinTestWrapper()
    # c = JustinTestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


@pytest.fixture()
def box_setup(zero_pose: JustinTestWrapper) -> JustinTestWrapper:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.pose.position.x = 1.2
    p.pose.position.y = 0
    p.pose.position.z = 0.1
    p.pose.orientation.w = 1
    zero_pose.add_box_to_world(name='box', size=(1, 1, 1), pose=p)
    return zero_pose


class TestJointGoals:

    def test_joint_goal(self, zero_pose: JustinTestWrapper):
        js = {
            "torso1_joint": 0.0,
            "torso2_joint": -0.9,
            "torso3_joint": 1.26,
            "head1_joint": 0.0,
            "head2_joint": 0.0,
            "left_arm1_joint": 0.0,
            "left_arm2_joint": 0,
            "left_arm3_joint": 0.0,
            "left_arm4_joint": 0.0,
            "left_arm5_joint": 0.0,
            "left_arm6_joint": -0.0,
            "left_arm7_joint": 0.0,
            "right_arm1_joint": 0.6,
            "right_arm2_joint": -1.59,
            "right_arm3_joint": 2.97,
            "right_arm4_joint": -0.99,
            "right_arm5_joint": -2.44,
            "right_arm6_joint": 0.0,
            "right_arm7_joint": 0.0,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.execute()

    def test_torso4(self, zero_pose: JustinTestWrapper):
        js = {
            "torso2_joint": -1,
            "torso3_joint": 0.2,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.execute()
        js = {
            "torso2_joint": 0,
            "torso3_joint": 2,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.execute()


class TestEuRobin:

    def test_open_fridge(self, kitchen_setup: JustinTestWrapper):
        handle_frame_id = 'iai_kitchen/iai_fridge_door_handle'
        handle_name = 'iai_fridge_door_handle'
        # base_goal = PoseStamped()
        # base_goal.header.frame_id = 'map'
        # base_goal.pose.position = Point(0.3, -0.5, 0)
        # base_goal.pose.orientation.w = 1
        # kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.r_tip
        tip_grasp_axis.vector.y = 1

        kitchen_setup.set_grasp_bar_goal(root_link=kitchen_setup.default_root,
                                         tip_link=kitchen_setup.r_tip,
                                         tip_grasp_axis=tip_grasp_axis,
                                         bar_center=bar_center,
                                         bar_axis=bar_axis,
                                         bar_length=.4)
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.r_tip
        x_gripper.vector.z = 1

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1
        kitchen_setup.set_align_planes_goal(tip_link=kitchen_setup.r_tip,
                                            tip_normal=x_gripper,
                                            goal_normal=x_goal,
                                            root_link='map')
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.execute()
        current_pose = kitchen_setup.compute_fk_pose(root_link='map', tip_link=kitchen_setup.r_tip)

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.r_tip,
                                              environment_link=handle_name,
                                              goal_joint_state=1.5)
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.execute()
        kitchen_setup.set_env_state({'iai_fridge_door_joint': 1.5})

        pose_reached = kitchen_setup.monitors.add_cartesian_pose('map',
                                                                 tip_link=kitchen_setup.r_tip,
                                                                 goal_pose=current_pose)
        kitchen_setup.monitors.add_end_motion(start_condition=pose_reached)

        kitchen_setup.set_open_container_goal(tip_link=kitchen_setup.r_tip,
                                              environment_link=handle_name,
                                              goal_joint_state=0)
        kitchen_setup.allow_all_collisions()
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)

        kitchen_setup.execute(add_local_minimum_reached=False)

        kitchen_setup.set_env_state({'iai_fridge_door_joint': 0})

        kitchen_setup.set_joint_goal(kitchen_setup.better_pose)
        kitchen_setup.allow_all_collisions()
        kitchen_setup.execute()

    def test_open_fridge_dlr(self, dlr_kitchen_setup: JustinTestWrapper):
        handle_frame_id = 'dlr_kitchen/fridge_door_handle'
        handle_name = 'fridge_door_handle'
        door_joint = 'fridge_door_joint'
        fridge = 'dlr_kitchen/fridge'
        # base_goal = PoseStamped()
        box_name = 'box'
        kitchenette = 'dlr_kitchen/kitchenette'
        box_pose = PoseStamped()
        box_pose.header.frame_id = kitchenette
        box_pose.pose.position.z = 0.22
        box_pose.pose.position.x = -0.1
        box_pose.pose.orientation.w = 1.0
        dlr_kitchen_setup.world.add_box(name=box_name, size=(0.03, 0.15, 0.2), pose=box_pose, parent_link=kitchenette)
        dlr_kitchen_setup.dye_group(group_name=box_name, rgba=(0.0, 0.0, 1.0, 1.0))

        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header.frame_id = box_name
        pre_grasp_pose.pose.orientation = Quaternion(*quaternion_from_rotation_matrix([[-1, 0, 0, 0],
                                                                                   [0, 1, 0, 0],
                                                                                   [0, 0, -1, 0],
                                                                                   [0, 0, 0, 1]]))
        pre_grasp_pose.pose.position.z = 0.25
        box_pre_grasped = dlr_kitchen_setup.tasks.add_cartesian_pose(name='pregrasp pose',
                                                                 goal_pose=pre_grasp_pose,
                                                                 tip_link=dlr_kitchen_setup.r_tip,
                                                                 root_link='map')

        grasp_pose = deepcopy(pre_grasp_pose)
        grasp_pose.pose.position.z -= 0.1
        box_grasped = dlr_kitchen_setup.tasks.add_cartesian_pose(name='grasp box',
                                                                 goal_pose=grasp_pose,
                                                                 tip_link=dlr_kitchen_setup.r_tip,
                                                                 root_link='map',
                                                                 start_condition=box_pre_grasped)
        dlr_kitchen_setup.monitors.add_end_motion(start_condition=box_grasped)
        dlr_kitchen_setup.avoid_all_collisions()
        dlr_kitchen_setup.allow_self_collision()
        dlr_kitchen_setup.execute(add_local_minimum_reached=False)

        # %%
        dlr_kitchen_setup.world.update_parent_link_of_group(name=box_name, parent_link=dlr_kitchen_setup.r_tip)

        # %%
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'base_footprint'
        base_pose.pose.orientation.w = 1.0
        base_pose.pose.position.x = -1
        drove_back = dlr_kitchen_setup.tasks.add_cartesian_pose(name='drive back',
                                                                goal_pose=base_pose,
                                                                tip_link='base_footprint',
                                                                root_link='map')

        hand_over_pose = PoseStamped()
        hand_over_pose.header.frame_id = dlr_kitchen_setup.l_tip
        hand_over_pose.pose.orientation = Quaternion(*quaternion_from_rotation_matrix([[-1, 0, 0, 0],
                                                                                       [0, 1, 0, 0],
                                                                                       [0, 0, -1, 0],
                                                                                       [0, 0, 0, 1]]))
        hand_over_pose.pose.position.z = 0.3
        handed_over = dlr_kitchen_setup.tasks.add_cartesian_pose(name='hand over',
                                                                 goal_pose=hand_over_pose,
                                                                 tip_link=dlr_kitchen_setup.r_tip,
                                                                 root_link=dlr_kitchen_setup.l_tip)

        done = f'{handed_over} and {drove_back}'
        dlr_kitchen_setup.monitors.add_end_motion(start_condition=done)
        dlr_kitchen_setup.execute(add_local_minimum_reached=False)

        # %%
        dlr_kitchen_setup.world.update_parent_link_of_group(name=box_name, parent_link=dlr_kitchen_setup.l_tip)
        # %%
        in_default_pose = dlr_kitchen_setup.tasks.add_joint_position(name='default joint pose',
                                                                     goal_state=dlr_kitchen_setup.better_pose)
        handle_grasp_pose = PoseStamped()
        handle_grasp_pose.header.frame_id = handle_frame_id
        handle_grasp_pose.pose.orientation = Quaternion(*quaternion_from_rotation_matrix([[0, 0, 1, 0],
                                                                                          [1, 0, 0, 0],
                                                                                          [0, 1, 0, 0],
                                                                                          [0, 0, 0, 1]]))
        handle_grasp_pose.pose.position.x = -0.2
        handle_graped = dlr_kitchen_setup.tasks.add_cartesian_pose(name='grasp handle',
                                                                   goal_pose=handle_grasp_pose,
                                                                   tip_link=dlr_kitchen_setup.r_tip,
                                                                   root_link='map',
                                                                   start_condition=in_default_pose)

        door_open = dlr_kitchen_setup.motion_goals.add_open_container(name='open fridge',
                                                                      tip_link=dlr_kitchen_setup.r_tip,
                                                                      environment_link=handle_name,
                                                                      start_condition=handle_graped)
        door_half_open = dlr_kitchen_setup.monitors.add_joint_position(name='is door half open?',
                                                                       goal_state={door_joint: np.pi / 4},
                                                                       start_condition=handle_graped)

        place_pose = PoseStamped()
        place_pose.header.frame_id = fridge
        place_pose.pose.orientation = Quaternion(*quaternion_from_rotation_matrix([[0, 0, 1, 0],
                                                                                   [-1, 0, 0, 0],
                                                                                   [0, -1, 0, 0],
                                                                                   [0, 0, 0, 1]]))
        place_pose.pose.position.z = 0.35
        place_pose.pose.position.x = 0.
        box_placed = dlr_kitchen_setup.tasks.add_cartesian_pose(name='place box',
                                                                goal_pose=place_pose,
                                                                tip_link=box_name,
                                                                root_link='map',
                                                                start_condition=door_half_open)



        done = f'{door_open} and {box_placed}'
        dlr_kitchen_setup.monitors.add_end_motion(start_condition=done)
        dlr_kitchen_setup.execute(add_local_minimum_reached=False)

        # %%
        dlr_kitchen_setup.world.update_parent_link_of_group(name=box_name, parent_link=fridge)

        # %%
        retract_pose = PoseStamped()
        retract_pose.header.frame_id = dlr_kitchen_setup.l_tip
        retract_pose.pose.orientation.w = 1.0
        retract_pose.pose.position.z = -0.5
        retracted = dlr_kitchen_setup.tasks.add_cartesian_pose(name='retract left hand',
                                                               goal_pose=retract_pose,
                                                               tip_link=dlr_kitchen_setup.l_tip,
                                                               root_link='map')

        door_closed = dlr_kitchen_setup.motion_goals.add_close_container(name='close fridge',
                                                                         tip_link=dlr_kitchen_setup.r_tip,
                                                                         environment_link=handle_name)
        dlr_kitchen_setup.monitors.add_end_motion(start_condition=f'{door_closed} and {retracted}')
        dlr_kitchen_setup.execute(add_local_minimum_reached=False)
