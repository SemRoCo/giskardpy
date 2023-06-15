from typing import Optional

import numpy as np
from std_msgs.msg import ColorRGBA

from giskardpy.configs.data_types import ControlModes, TfPublishingModes
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import Derivatives


class Donbot_Base(Giskard):
    localization_joint_name = 'localization'
    map_name = 'map'

    def configure_world(self):
        self.world.set_default_color(r=1, g=1, b=1, a=1)
        self.world.set_default_limits({Derivatives.velocity: 0.5,
                                       Derivatives.acceleration: np.inf,
                                       Derivatives.jerk: 15})
        self.world.add_empty_link(self.map_name)
        pr2_group_name = self.world.add_robot_from_parameter_server()
        root_link_name = self.world.get_root_link_of_group(pr2_group_name)
        self.world.add_6dof_joint(parent_link=self.map_name, child_link=root_link_name,
                                  joint_name=self.localization_joint_name)
        self.world.set_joint_limits(limit_map={Derivatives.velocity: 0.1}, joint_name='odom_x_joint')
        self.world.set_joint_limits(limit_map={Derivatives.velocity: 0.1}, joint_name='odom_y_joint')
        self.world.set_joint_limits(limit_map={Derivatives.velocity: 0.05}, joint_name='odom_z_joint')

    def configure_collision_avoidance(self):
        self.collision_avoidance.load_moveit_self_collision_matrix('package://giskardpy/config/iai_donbot.srdf')
        self.collision_avoidance.ignore_self_collisions_of_pair('ur5_forearm_link', 'ur5_wrist_3_link')
        self.collision_avoidance.ignore_self_collisions_of_pair('ur5_base_link', 'ur5_upper_arm_link')
        self.collision_avoidance.add_self_collision('plate', 'ur5_upper_arm_link')
        self.collision_avoidance.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                                          hard_threshold=0.0)
        self.collision_avoidance.overwrite_external_collision_avoidance('odom_z_joint',
                                                                        number_of_repeller=2,
                                                                        soft_threshold=0.1,
                                                                        hard_threshold=0.05)
        close_links = ['ur5_wrist_1_link', 'ur5_wrist_2_link', 'ur5_wrist_3_link', 'ur5_forearm_link',
                       'ur5_upper_arm_link']
        for link_name in close_links:
            self.collision_avoidance.overwrite_self_collision_avoidance(link_name,
                                                                        soft_threshold=0.02,
                                                                        hard_threshold=0.005)
        super_close_links = ['gripper_gripper_left_link', 'gripper_gripper_right_link']
        for link_name in super_close_links:
            self.collision_avoidance.overwrite_self_collision_avoidance(link_name,
                                                                        soft_threshold=0.00001,
                                                                        hard_threshold=0.0)


class Donbot_IAI(Donbot_Base):

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.open_loop)

    def configure_behavior_tree(self):
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=True,
                                                              add_to_control_loop=False)

    def configure_robot_interface(self):
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/whole_body_controller/base/follow_joint_trajectory',
            state_topic='/whole_body_controller/base/state',
            fill_velocity_values=True)
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            state_topic='/scaled_pos_joint_traj_controller/state',
            fill_velocity_values=True)


class Donbot_Standalone(Donbot_Base):

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.stand_alone)
        self.execution.set_max_trajectory_length(length=30)

    def configure_behavior_tree(self):
        super().configure_behavior_tree()
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False,
                                                              add_to_control_loop=True)
        self.behavior_tree.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)

    def configure_robot_interface(self):
        self.robot_interface.register_controlled_joints([
            'ur5_elbow_joint',
            'ur5_shoulder_lift_joint',
            'ur5_shoulder_pan_joint',
            'ur5_wrist_1_joint',
            'ur5_wrist_2_joint',
            'ur5_wrist_3_joint',
            'odom_x_joint',
            'odom_y_joint',
            'odom_z_joint',
        ])
