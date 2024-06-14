from giskardpy.configs.giskard import CollisionAvoidanceConfig
from giskardpy.configs.world_config import WorldWithOmniDriveRobot
from giskardpy.data_types import Derivatives
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig
from giskardpy.god_map import god_map


class KevinCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        # self.collision_scene.collision_checker_id = self.collision_scene.collision_checker_id.none
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/kevin.srdf')
        self.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)

        god_map.world.register_group('gripper', 'kevin/robot_arm_wrist_link', actuated=True)


class KevinStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            drive_joint_name,
            'robot_left_rocker_joint',
            'robot_right_rocker_joint',
            'robot_arm_inner_joint',
            'robot_arm_outer_joint',
            'robot_arm_wrist_joint',
            'robot_arm_column_joint',
            'robot_arm_gripper_joint',
            'robot_arm_gripper_mirror_joint'
        ])
