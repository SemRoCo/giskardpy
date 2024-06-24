import numpy as np

from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig, RobotInterfaceConfig
from giskardpy.configs.world_config import WorldConfig
from giskardpy.my_types import PrefixName, Derivatives


class WorldWithArmarConfig(WorldConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.set_default_color(1, 1, 1, 1)
        self.set_default_limits({Derivatives.velocity: 1,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: 30})
        self.add_empty_link(self.map_name)
        self.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                            joint_name=self.localization_joint_name)
        self.add_empty_link(self.odom_link_name)
        self.add_robot_from_parameter_server()
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_omni_drive_joint(parent_link_name=self.odom_link_name,
                                  child_link_name=root_link_name,
                                  name=self.drive_joint_name,
                                  x_name=PrefixName('odom_x', self.robot_group_name),
                                  y_name=PrefixName('odom_y', self.robot_group_name),
                                  yaw_vel_name=PrefixName('odom_t', self.robot_group_name),
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
                                  robot_group_name=self.robot_group_name)


class ArmarCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/other_robots/armar.srdf')
        self.set_default_external_collision_avoidance(soft_threshold=0.05,
                                                      hard_threshold=0.0)


class ArmarStandaloneInterface(StandAloneRobotInterfaceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            'ArmL1_Cla1',
            'ArmL2_Sho1',
            'ArmL3_Sho2',
            'ArmL4_Sho3',
            'ArmL5_Elb1',
            'ArmL6_Elb2',
            'ArmL7_Wri1',
            'ArmL8_Wri2',
            'ArmR1_Cla1',
            'ArmR2_Sho1',
            'ArmR3_Sho2',
            'ArmR4_Sho3',
            'ArmR5_Elb1',
            'ArmR6_Elb2',
            'ArmR7_Wri1',
            'ArmR8_Wri2',
            # 'Index_L_1_Joint',
            # 'Index_L_2_Joint',
            # 'Index_L_3_Joint',
            # 'Index_R_1_Joint',
            # 'Index_R_2_Joint',
            # 'Index_R_3_Joint',
            # 'LeftHandFingers',
            # 'LeftHandThumb',
            # 'Middle_L_1_Joint',
            # 'Middle_L_2_Joint',
            # 'Middle_L_3_Joint',
            # 'Middle_R_1_Joint',
            # 'Middle_R_2_Joint',
            # 'Middle_R_3_Joint',
            'Neck_1_Yaw',
            'Neck_2_Pitch',
            # 'Pinky_L_1_Joint',
            # 'Pinky_L_2_Joint',
            # 'Pinky_L_3_Joint',
            # 'Pinky_R_1_Joint',
            # 'Pinky_R_2_Joint',
            # 'Pinky_R_3_Joint',
            # 'RightHandFingers',
            # 'RightHandThumb',
            # 'Ring_L_1_Joint',
            # 'Ring_L_2_Joint',
            # 'Ring_L_3_Joint',
            # 'Ring_R_1_Joint',
            # 'Ring_R_2_Joint',
            # 'Ring_R_3_Joint',
            # 'Thumb_L_1_Joint',
            # 'Thumb_L_2_Joint',
            # 'Thumb_R_1_Joint',
            # 'Thumb_R_2_Joint',
            'TorsoJoint',
            # 'VirtualCentralGaze',
            # 'VirtualCentralGazeDepthCamera',
            drive_joint_name])
