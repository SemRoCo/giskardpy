from giskardpy.configs.giskard import CollisionAvoidanceConfig
from giskardpy.configs.world_config import WorldWithOmniDriveRobot, WorldWithDiffDriveRobot
from giskardpy.data_types import Derivatives
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig
from giskardpy.god_map import god_map
import numpy as np


class WorldWithKevinAndFixedArmsRobot(WorldWithDiffDriveRobot):
    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        super().__init__(map_name=map_name, localization_joint_name=localization_joint_name,
                         odom_link_name=odom_link_name, drive_joint_name=drive_joint_name)

    def setup(self):
        self.set_default_limits({Derivatives.velocity: 1,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: 30})
        self.add_empty_link(self.map_name)
        self.add_empty_link(self.odom_link_name)
        self.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                            joint_name=self.localization_joint_name)
        self.add_robot_from_parameter_server(parameter_name='kevin/robot_description', group_name='kevin')
        root_link_name = self.get_root_link_of_group('kevin')
        self.add_diff_drive_joint(name=self.drive_joint_name,
                                  parent_link_name=self.odom_link_name,
                                  child_link_name=root_link_name,
                                  translation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  },
                                  robot_group_name='kevin')

        self.add_robot_from_parameter_server('tracy/robot_description', group_name='tracy')
        root_link_tracy = self.get_root_link_of_group('tracy')
        transform = np.eye(4)
        transform[0,3] = 3
        transform[2, 3] = 0.9
        transform[0, 0] = -1
        transform[1, 1] = -1
        self.add_fixed_joint(self.map_name, root_link_tracy, transform)


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
            'robot_arm_inner_joint',
            'robot_arm_outer_joint',
            'robot_arm_wrist_joint',
            'robot_arm_column_joint',
            'robot_arm_gripper_joint',
            'robot_arm_gripper_mirror_joint'
        ])


class KevinAndArmsStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            drive_joint_name,
            'robot_arm_inner_joint',
            'robot_arm_outer_joint',
            'robot_arm_wrist_joint',
            'robot_arm_column_joint',
            'robot_arm_gripper_joint',
            'robot_arm_gripper_mirror_joint',
            'left_shoulder_pan_joint',
            'left_shoulder_lift_joint',
            'left_elbow_joint',
            'left_wrist_1_joint',
            'left_wrist_2_joint',
            'left_wrist_3_joint',
            'right_shoulder_pan_joint',
            'right_shoulder_lift_joint',
            'right_elbow_joint',
            'right_wrist_1_joint',
            'right_wrist_2_joint',
            'right_wrist_3_joint'
        ])
