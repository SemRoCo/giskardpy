from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig, RobotInterfaceConfig
from giskardpy.configs.world_config import WorldConfig
from giskardpy.data_types import PrefixName, Derivatives
import numpy as np


class WorldWithJustinConfig(WorldConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum',
                 description_name: str = 'robot_description'):
        super().__init__()
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name
        self.robot_description_name = description_name

    def setup(self):
        self.set_default_color(1, 1, 1, 1)
        self.set_default_limits({Derivatives.velocity: 1,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: 30})
        self.add_empty_link(self.map_name)
        self.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                            joint_name=self.localization_joint_name)
        self.add_empty_link(self.odom_link_name)
        self.add_robot_from_parameter_server(parameter_name=self.robot_description_name)
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

class JustinStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            drive_joint_name,
            "torso1_joint",
            "torso2_joint",
            "torso3_joint",
            "torso4_joint",
            "head1_joint",
            "head2_joint",
            "left_arm1_joint",
            "left_arm2_joint",
            "left_arm3_joint",
            "left_arm4_joint",
            "left_arm5_joint",
            "left_arm6_joint",
            "left_arm7_joint",
            "left_1thumb1_joint",
            "left_1thumb2_joint",
            "left_1thumb3_joint",
            "left_1thumb4_joint",
            "left_2tip1_joint",
            "left_2tip2_joint",
            "left_2tip3_joint",
            "left_2tip4_joint",
            "left_3middle1_joint",
            "left_3middle2_joint",
            "left_3middle3_joint",
            "left_3middle4_joint",
            "left_4ring1_joint",
            "left_4ring2_joint",
            "left_4ring3_joint",
            "left_4ring4_joint",
            "right_arm1_joint",
            "right_arm2_joint",
            "right_arm3_joint",
            "right_arm4_joint",
            "right_arm5_joint",
            "right_arm6_joint",
            "right_arm7_joint",
            "right_1thumb1_joint",
            "right_1thumb2_joint",
            "right_1thumb3_joint",
            "right_1thumb4_joint",
            "right_3middle1_joint",
            "right_3middle2_joint",
            "right_3middle3_joint",
            "right_3middle4_joint",
            "right_4ring1_joint",
            "right_4ring2_joint",
            "right_4ring3_joint",
            "right_4ring4_joint",
            "right_2tip1_joint",
            "right_2tip2_joint",
            "right_2tip3_joint",
            "right_2tip4_joint"
        ])


class JustinCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/justin.srdf')
