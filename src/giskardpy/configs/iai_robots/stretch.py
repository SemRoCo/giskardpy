from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig, RobotInterfaceConfig


class StretchCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/stretch.srdf')
        self.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)


class StretchStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            'torso_lift_joint', 'head_1_joint', 'head_2_joint', drive_joint_name,
            'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
            'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint', 'arm_right_1_joint', 'arm_right_2_joint',
            'arm_right_3_joint',
            'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
            'arm_right_7_joint', 'gripper_right_left_finger_joint', 'gripper_right_right_finger_joint',
            'gripper_left_left_finger_joint', 'gripper_left_right_finger_joint'
        ])
