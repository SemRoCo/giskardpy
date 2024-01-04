#!/usr/bin/env python
import numpy as np
import rospy

from giskardpy.configs.data_types import ControlModes, TfPublishingModes
from giskardpy.configs.giskard import Giskard
from giskardpy.data_types import Derivatives


class MyPR2Setup(Giskard):
    map_name = 'map'
    localization_joint_name = 'localization'
    odom_link_name = 'odom_combined'
    drive_joint_name = 'brumbrum'

    def configure_world(self):
        self.world_config.set_default_color(1, 1, 1, 0.8)
        self.world_config.set_default_limits({Derivatives.velocity: 1,
                                              Derivatives.acceleration: np.inf,
                                              Derivatives.jerk: 30})
        self.world_config.add_empty_link(self.map_name)
        self.world_config.add_empty_link(self.odom_link_name)
        self.world_config.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                                         joint_name=self.localization_joint_name)
        pr2_group_name = self.world_config.add_robot_from_parameter_server()
        root_link_name = self.world_config.get_root_link_of_group(pr2_group_name)
        self.world_config.add_omni_drive_joint(name=self.drive_joint_name,
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
                                               robot_group_name=pr2_group_name)
        self.world_config.set_joint_limits(limit_map={Derivatives.velocity: 3,
                                                      Derivatives.jerk: 60},
                                           joint_name='head_pan_joint')

    def configure_collision_avoidance(self):
        self.collision_avoidance_config.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/pr2.srdf')
        self.collision_avoidance_config.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                                                 hard_threshold=0.0)
        for joint_name in ['r_wrist_roll_joint', 'l_wrist_roll_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   number_of_repeller=4,
                                                                                   soft_threshold=0.05,
                                                                                   hard_threshold=0.0,
                                                                                   max_velocity=0.2)
        for joint_name in ['r_wrist_flex_joint', 'l_wrist_flex_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   number_of_repeller=2,
                                                                                   soft_threshold=0.05,
                                                                                   hard_threshold=0.0,
                                                                                   max_velocity=0.2)
        for joint_name in ['r_elbow_flex_joint', 'l_elbow_flex_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   soft_threshold=0.05,
                                                                                   hard_threshold=0.0)
        for joint_name in ['r_forearm_roll_joint', 'l_forearm_roll_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   soft_threshold=0.025,
                                                                                   hard_threshold=0.0)
        self.collision_avoidance_config.fix_joints_for_collision_avoidance([
            'r_gripper_l_finger_joint',
            'l_gripper_l_finger_joint'
        ])
        self.collision_avoidance_config.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                                               number_of_repeller=2,
                                                                               soft_threshold=0.2,
                                                                               hard_threshold=0.1)

    def configure_execution(self):
        self.qp_controller_config.set_control_mode(ControlModes.standalone)
        self.qp_controller_config.set_max_trajectory_length(length=30)

    def configure_robot_interface(self):
        self.robot_interface_config.register_controlled_joints([
            'torso_lift_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'r_shoulder_pan_joint',
            'r_shoulder_lift_joint',
            'r_upper_arm_roll_joint',
            'r_forearm_roll_joint',
            'r_elbow_flex_joint',
            'r_wrist_flex_joint',
            'r_wrist_roll_joint',
            'l_shoulder_pan_joint',
            'l_shoulder_lift_joint',
            'l_upper_arm_roll_joint',
            'l_forearm_roll_joint',
            'l_elbow_flex_joint',
            'l_wrist_flex_joint',
            'l_wrist_roll_joint',
            self.drive_joint_name,
        ])

    def configure_behavior_tree(self):
        self.behavior_tree_config.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False,
                                                                     add_to_control_loop=True)
        self.behavior_tree_config.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)


rospy.init_node('giskard')

# Start Giskard.
MyPR2Setup().live()
