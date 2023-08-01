from typing import Optional

import numpy as np

from giskardpy.configs.data_types import ControlModes, SupportedQPSolver, TfPublishingModes
from giskardpy.configs.giskard import Giskard
from giskardpy.my_types import PrefixName, Derivatives


class HSR_Base(Giskard):
    localization_joint_name = 'localization'
    drive_joint_name = 'brumbrum'
    odom_link_name = 'odom'
    map_name = 'map'

    def configure_world(self, robot_description: str = 'robot_description'):
        self.world_config.set_default_color(1, 1, 1, 1)
        self.world_config.set_default_limits({Derivatives.velocity: 1,
                                              Derivatives.acceleration: np.inf,
                                              Derivatives.jerk: 30})
        self.world_config.add_empty_link(self.map_name)
        self.world_config.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                                         joint_name=self.localization_joint_name)
        self.world_config.add_empty_link(self.odom_link_name)
        pr2_group_name = self.world_config.add_robot_from_parameter_server(robot_description)
        root_link_name = self.world_config.get_root_link_of_group(pr2_group_name)
        self.world_config.add_omni_drive_joint(parent_link_name=self.odom_link_name,
                                               child_link_name=root_link_name,
                                               name=self.drive_joint_name,
                                               x_name=PrefixName('odom_x', pr2_group_name),
                                               y_name=PrefixName('odom_y', pr2_group_name),
                                               yaw_vel_name=PrefixName('odom_t', pr2_group_name),
                                               translation_limits={
                                            Derivatives.velocity: 0.2,
                                            Derivatives.acceleration: 1,
                                            Derivatives.jerk: 6,
                                        },
                                               rotation_limits={
                                            Derivatives.velocity: 0.2,
                                            Derivatives.acceleration: 1,
                                            Derivatives.jerk: 6
                                        },
                                               robot_group_name=pr2_group_name)

    def configure_collision_avoidance(self):
        # self.collision_avoidance.set_collision_checker(CollisionCheckerLib.none)
        self.collision_avoidance_config.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/hsrb.srdf')
        self.collision_avoidance_config.set_default_external_collision_avoidance(soft_threshold=0.05,
                                                                                 hard_threshold=0.0)
        self.collision_avoidance_config.overwrite_external_collision_avoidance('wrist_roll_joint',
                                                                               number_of_repeller=4,
                                                                               soft_threshold=0.05,
                                                                               hard_threshold=0.0,
                                                                               max_velocity=0.2)
        self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name='brumbrum',
                                                                               number_of_repeller=2,
                                                                               soft_threshold=0.1,
                                                                               hard_threshold=0.03)
        self.collision_avoidance_config.overwrite_self_collision_avoidance(link_name='head_tilt_link',
                                                                           soft_threshold=0.03)

    def configure_robot_interface(self):
        pass


class HSR_Realtime(HSR_Base):
    def configure_execution(self):
        self.execution_config.set_control_mode(ControlModes.close_loop)

    def configure_world(self, robot_description: str = 'robot_description'):
        super().configure_world('robot_description')
        self.world_config.set_default_color(1, 1, 1, 0.7)

    def configure_behavior_tree(self):
        self.behavior_tree_config.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=False,
                                                                     add_to_control_loop=False)
        self.behavior_tree_config.add_debug_marker_publisher()
        self.behavior_tree_config.add_qp_data_publisher(publish_debug=True, add_to_base=False)

    def configure_robot_interface(self):
        self.robot_interface_config.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                                                  tf_parent_frame=self.map_name,
                                                                  tf_child_frame=self.odom_link_name)
        self.robot_interface_config.sync_joint_state_topic('/hsrb/joint_states')
        self.robot_interface_config.sync_odometry_topic('/hsrb/odom', self.drive_joint_name)

        self.robot_interface_config.add_joint_velocity_group_controller(namespace='/hsrb/realtime_body_controller_real')

        self.robot_interface_config.add_base_cmd_velocity(cmd_vel_topic='/hsrb/command_velocity',
                                                          joint_name=self.drive_joint_name)


class HSR_StandAlone(HSR_Base):

    def configure_execution(self):
        self.execution_config.set_control_mode(ControlModes.standalone)

    def configure_behavior_tree(self):
        self.behavior_tree_config.add_visualization_marker_publisher(add_to_sync=True,
                                                                     add_to_control_loop=True)
        self.behavior_tree_config.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)

    def configure_robot_interface(self):
        super().configure_robot_interface()
        self.robot_interface_config.register_controlled_joints([
            'arm_flex_joint',
            'arm_lift_joint',
            'arm_roll_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            self.drive_joint_name,
        ])
