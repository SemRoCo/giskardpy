import numpy as np

from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig, RobotInterfaceConfig
from giskardpy.configs.world_config import WorldConfig
from giskardpy.data_types import PrefixName, Derivatives


class WorldWithHSRConfig(WorldConfig):
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
        self.set_joint_limits(limit_map={
                                      Derivatives.jerk: 10,
                                  },
            joint_name='arm_lift_joint')


class HSRCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/hsrb.srdf')
        self.set_default_external_collision_avoidance(soft_threshold=0.05,
                                                      hard_threshold=0.0)
        self.overwrite_external_collision_avoidance('wrist_roll_joint',
                                                    number_of_repeller=4,
                                                    soft_threshold=0.05,
                                                    hard_threshold=0.0,
                                                    max_velocity=0.2)
        self.overwrite_external_collision_avoidance(joint_name=self.drive_joint_name,
                                                    number_of_repeller=2,
                                                    soft_threshold=0.1,
                                                    hard_threshold=0.03)
        self.overwrite_self_collision_avoidance(link_name='head_tilt_link',
                                                soft_threshold=0.03)


class HSRStandaloneInterface(StandAloneRobotInterfaceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            'arm_flex_joint',
            'arm_lift_joint',
            'arm_roll_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            drive_joint_name])


class HSRVelocityInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                           tf_parent_frame=self.map_name,
                                           tf_child_frame=self.odom_link_name)
        self.sync_joint_state_topic('/hsrb/joint_states')
        self.sync_odometry_topic('/hsrb/odom', self.drive_joint_name)

        self.add_joint_velocity_group_controller(namespace='/hsrb/realtime_body_controller_real')

        self.add_base_cmd_velocity(cmd_vel_topic='/hsrb/command_velocity',
                                   joint_name=self.drive_joint_name)


class HSRJointTrajInterfaceConfig(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                           tf_parent_frame=self.map_name,
                                           tf_child_frame=self.odom_link_name)
        self.sync_joint_state_topic('/hsrb/joint_states')
        self.sync_odometry_topic('/hsrb/odom', self.drive_joint_name)

        self.add_follow_joint_trajectory_server(namespace='/hsrb/head_trajectory_controller',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/hsrb/arm_trajectory_controller',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/hsrb/omni_base_controller',
                                                fill_velocity_values=True,
                                                path_tolerance={
                                                    Derivatives.position: 1,
                                                    Derivatives.velocity: 1,
                                                    Derivatives.acceleration: 100})
        # self.add_base_cmd_velocity(cmd_vel_topic='/hsrb/command_velocity',
        #                            track_only_velocity=True,
        #                            joint_name=self.drive_joint_name)
