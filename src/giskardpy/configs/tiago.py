from typing import Optional

import numpy as np

from giskardpy.configs.data_types import SupportedQPSolver, TfPublishingModes
from giskardpy.configs.default_giskard import Giskard, ControlModes
from giskardpy.my_types import Derivatives


class TiagoBase(Giskard):
    localization_joint_name = 'localization'
    drive_joint_name = 'brumbrum'
    odom_link_name = 'odom'
    map_name = 'map'

    def configure_world(self):
        self.world.set_default_color(1, 1, 1, 0.7)
        self.world.set_default_limits({Derivatives.velocity: 1,
                                       Derivatives.acceleration: np.inf,
                                       Derivatives.jerk: 30})
        self.world.add_empty_link(self.map_name)
        self.world.add_empty_link(self.odom_link_name)
        self.world.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                                  joint_name=self.localization_joint_name)
        robot_group_name = self.world.add_robot_from_parameter_server()
        root_link_name = self.world.get_root_link_of_group(robot_group_name)
        self.world.add_diff_drive_joint(name=self.drive_joint_name,
                                        parent_link_name=self.odom_link_name,
                                        child_link_name=root_link_name,
                                        translation_limits={
                                            Derivatives.velocity: 0.4,
                                            Derivatives.acceleration: 1,
                                            Derivatives.jerk: 5,
                                        },
                                        rotation_limits={
                                            Derivatives.velocity: 0.2,
                                            Derivatives.acceleration: 1,
                                            Derivatives.jerk: 5
                                        },
                                        robot_group_name=robot_group_name)

    def configure_collision_avoidance(self):
        self.collision_avoidance.load_moveit_self_collision_matrix('package://giskardpy/config/tiago.srdf')
        self.collision_avoidance.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                                        number_of_repeller=2,
                                                                        soft_threshold=0.2,
                                                                        hard_threshold=0.1)
        self.ignored_collisions = ['wheel_left_link',
                                   'wheel_right_link',
                                   'caster_back_left_2_link',
                                   'caster_back_right_2_link',
                                   'caster_front_left_2_link',
                                   'caster_front_right_2_link']
        self.collision_avoidance.fix_joints_for_self_collision_avoidance(['head_1_joint',
                                                                          'head_2_joint',
                                                                          'gripper_left_left_finger_joint',
                                                                          'gripper_left_right_finger_joint',
                                                                          'gripper_right_left_finger_joint',
                                                                          'gripper_right_right_finger_joint'])
        self.collision_avoidance.fix_joints_for_external_collision_avoidance(['gripper_left_left_finger_joint',
                                                                              'gripper_left_right_finger_joint',
                                                                              'gripper_right_left_finger_joint',
                                                                              'gripper_right_right_finger_joint'])
        self.collision_avoidance.overwrite_external_collision_avoidance('arm_right_7_joint',
                                                                        number_of_repeller=4,
                                                                        soft_threshold=0.05,
                                                                        hard_threshold=0.0,
                                                                        max_velocity=0.2)
        self.collision_avoidance.overwrite_external_collision_avoidance('arm_left_7_joint',
                                                                        number_of_repeller=4,
                                                                        soft_threshold=0.05,
                                                                        hard_threshold=0.0,
                                                                        max_velocity=0.2)
        self.collision_avoidance.set_default_self_collision_avoidance(hard_threshold=0.04,
                                                                      soft_threshold=0.08)
        self.collision_avoidance.set_default_external_collision_avoidance(hard_threshold=0.03,
                                                                          soft_threshold=0.08)


class TiagoMujoco(TiagoBase):

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.open_loop)

    def configure_behavior_tree(self):
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=True)

    def configure_robot_interface(self):
        self.robot_interface.sync_joint_state_topic('/tiago/joint_states')
        self.robot_interface.sync_odometry_topic(odometry_topic='/tiago/base_footprint',
                                                 joint_name=self.drive_joint_name)
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/tiago/arm_left_controller/follow_joint_trajectory',
            state_topic='/tiago/arm_left_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/tiago/arm_right_controller/follow_joint_trajectory',
            state_topic='/tiago/arm_right_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/tiago/head_controller/follow_joint_trajectory',
            state_topic='/tiago/head_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/tiago/left_gripper_controller/follow_joint_trajectory',
            state_topic='/tiago/left_gripper_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/tiago/right_gripper_controller/follow_joint_trajectory',
            state_topic='/tiago/right_gripper_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/tiago/torso_controller/follow_joint_trajectory',
            state_topic='/tiago/torso_controller/state')
        self.robot_interface.add_base_cmd_velocity(cmd_vel_topic='/tiago/cmd_vel',
                                                   joint_name=self.drive_joint_name)


class Tiago_Standalone(TiagoBase):

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.stand_alone)

    def configure_behavior_tree(self):
        self.behavior_tree.add_tf_publisher(mode=TfPublishingModes.all)
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True, add_to_control_loop=True)

    def configure_robot_interface(self):
        self.robot_interface.register_controlled_joints(
            ['torso_lift_joint', 'head_1_joint', 'head_2_joint', self.drive_joint_name])
        self.robot_interface.register_controlled_joints(
            ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
             'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint'])
        self.robot_interface.register_controlled_joints(['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                                                         'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                                                         'arm_right_7_joint'])
        self.robot_interface.register_controlled_joints(
            ['gripper_right_left_finger_joint', 'gripper_right_right_finger_joint',
             'gripper_left_left_finger_joint', 'gripper_left_right_finger_joint'])
