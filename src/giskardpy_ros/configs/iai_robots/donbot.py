import numpy as np

from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.model.world_config import WorldConfig
from giskardpy_ros.configs.robot_interface_config import RobotInterfaceConfig, StandAloneRobotInterfaceConfig
from giskardpy.data_types.data_types import Derivatives


class WorldWithBoxyBaseConfig(WorldConfig):

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization'):
        super().__init__()
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name

    def setup(self):
        self.set_default_limits({Derivatives.velocity: 0.5,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: 15})
        self.add_empty_link(self.map_name)
        self.add_robot_from_parameter_server()
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_6dof_joint(parent_link=self.map_name, child_link=root_link_name,
                            joint_name=self.localization_joint_name)
        self.set_joint_limits(limit_map={Derivatives.velocity: 0.1}, joint_name='odom_x_joint')
        self.set_joint_limits(limit_map={Derivatives.velocity: 0.1}, joint_name='odom_y_joint')
        self.set_joint_limits(limit_map={Derivatives.velocity: 0.05}, joint_name='odom_z_joint')


class DonbotCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def setup(self):
        self.load_self_collision_matrix(
            'package://giskardpy/self_collision_matrices/iai/iai_donbot.srdf')
        self.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                      hard_threshold=0.0)
        self.overwrite_external_collision_avoidance('odom_z_joint',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.1,
                                                    hard_threshold=0.05)
        close_links = ['ur5_wrist_1_link', 'ur5_wrist_2_link', 'ur5_wrist_3_link', 'ur5_forearm_link',
                       'ur5_upper_arm_link']
        for link_name in close_links:
            self.overwrite_self_collision_avoidance(link_name,
                                                    soft_threshold=0.02,
                                                    hard_threshold=0.005)
        super_close_links = ['gripper_gripper_left_link', 'gripper_gripper_right_link']
        for link_name in super_close_links:
            self.overwrite_self_collision_avoidance(link_name,
                                                    soft_threshold=0.00001,
                                                    hard_threshold=0.0)


class DonbotStandaloneInterfaceConfig(StandAloneRobotInterfaceConfig):
    def __init__(self):
        super().__init__([
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


class DonbotJointTrajInterfaceConfig(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name

    def setup(self):
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                           tf_parent_frame=self.map_name,
                                           tf_child_frame=root_link_name.short_name)
        self.sync_joint_state_topic('/joint_states')
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/base',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/scaled_pos_joint_traj_controller',
                                                fill_velocity_values=True)
