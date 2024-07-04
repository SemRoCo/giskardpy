from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy_ros.configs.robot_interface_config import StandAloneRobotInterfaceConfig


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
            drive_joint_name,
            'joint_gripper_finger_left',
            'joint_gripper_finger_right',
            'joint_right_wheel',
            'joint_left_wheel',
            'joint_lift',
            'joint_arm_l3',
            'joint_arm_l2',
            'joint_arm_l1',
            'joint_arm_l0',
            'joint_wrist_yaw',
            'joint_head_pan',
            'joint_head_tilt',
        ])
