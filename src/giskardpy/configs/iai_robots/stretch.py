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


class StretchTrajectoryInterface(RobotInterfaceConfig):

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
        self.sync_joint_state_topic('/joint_states')
        self.sync_odometry_topic('/odom', self.drive_joint_name)
        fill_velocity_values = False
        self.add_follow_joint_trajectory_server(namespace='/stretch_controller',
                                                fill_velocity_values=fill_velocity_values,
                                                goal_time_tolerance=100,
                                                controlled_joints=['joint_gripper_finger_left',
                                                                   'joint_lift',
                                                                   'joint_arm_l3',
                                                                   'joint_arm_l2',
                                                                   'joint_arm_l1',
                                                                   'joint_arm_l0',
                                                                   'joint_wrist_yaw',
                                                                   'joint_wrist_pitch',
                                                                   'joint_wrist_roll',
                                                                   'joint_head_pan',
                                                                   'joint_head_tilt'])

        self.add_base_cmd_velocity(cmd_vel_topic='/stretch/cmd_vel',
                                   track_only_velocity=True,
                                   joint_name=self.drive_joint_name)
