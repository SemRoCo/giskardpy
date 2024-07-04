from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy_ros.configs.robot_interface_config import StandAloneRobotInterfaceConfig


class BoxyCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def setup(self):
        self.load_self_collision_matrix(
            'package://giskardpy/self_collision_matrices/iai/boxy_description.srdf')
        self.overwrite_external_collision_avoidance('odom_z_joint',
                                                                               number_of_repeller=2,
                                                                               soft_threshold=0.2,
                                                                               hard_threshold=0.1)


class BoxyStandaloneInterfaceConfig(StandAloneRobotInterfaceConfig):
    def __init__(self):
        super().__init__([
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
