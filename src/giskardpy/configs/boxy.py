import numpy as np

from giskardpy.configs.data_types import ControlModes, TfPublishingModes
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import Derivatives


class Boxy_StandAlone(Giskard):
    localization_joint_name = 'localization'
    map_name = 'map'

    def configure_world(self):
        self.world.set_default_color(1, 1, 1, 1)
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

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.stand_alone)

    def configure_collision_avoidance(self):
        self.collision_avoidance.overwrite_external_collision_avoidance('odom_z_joint',
                                                                        number_of_repeller=2,
                                                                        soft_threshold=0.2,
                                                                        hard_threshold=0.1)

    def configure_behavior_tree(self):
        self.behavior_tree.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True,
                                                              add_to_control_loop=True)

    def configure_robot_interface(self):
        self.robot_interface.register_controlled_joints([
            'neck_shoulder_pan_joint',
            'neck_shoulder_lift_joint',
            'neck_elbow_joint',
            'neck_wrist_1_joint',
            'neck_wrist_2_joint',
            'neck_wrist_3_joint',
            'triangle_base_joint',
            'left_arm_0_joint',
            'left_arm_1_joint',
            'left_arm_2_joint',
            'left_arm_3_joint',
            'left_arm_4_joint',
            'left_arm_5_joint',
            'left_arm_6_joint',
            'right_arm_0_joint',
            'right_arm_1_joint',
            'right_arm_2_joint',
            'right_arm_3_joint',
            'right_arm_4_joint',
            'right_arm_5_joint',
            'right_arm_6_joint',
            'odom_x_joint',
            'odom_y_joint',
            'odom_z_joint',
        ])
