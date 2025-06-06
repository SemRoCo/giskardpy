from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig, RobotInterfaceConfig


class TiagoCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/tiago_dual.srdf')
        self.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)
        self.fix_joints_for_collision_avoidance(['head_1_joint',
                                                 'head_2_joint',
                                                 'gripper_left_left_finger_joint',
                                                 'gripper_left_right_finger_joint',
                                                 'gripper_right_left_finger_joint',
                                                 'gripper_right_right_finger_joint'])
        self.overwrite_external_collision_avoidance('arm_right_7_joint',
                                                    number_of_repeller=4,
                                                    soft_threshold=0.05,
                                                    hard_threshold=0.0,
                                                    max_velocity=0.2)
        self.overwrite_external_collision_avoidance('arm_left_7_joint',
                                                    number_of_repeller=4,
                                                    soft_threshold=0.05,
                                                    hard_threshold=0.0,
                                                    max_velocity=0.2)
        self.set_default_self_collision_avoidance(hard_threshold=0.04,
                                                  soft_threshold=0.08)
        self.set_default_external_collision_avoidance(hard_threshold=0.03,
                                                      soft_threshold=0.08)


class TiagoJointTrajServerIAIInterface(RobotInterfaceConfig):
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
        self.sync_joint_state_topic('/tiago/joint_states')
        self.sync_odometry_topic(odometry_topic='/tiago/base_footprint',
                                 joint_name=self.drive_joint_name)
        self.add_follow_joint_trajectory_server(namespace='/tiago/arm_left_controller')
        self.add_follow_joint_trajectory_server(namespace='/tiago/arm_right_controller')
        self.add_follow_joint_trajectory_server(namespace='/tiago/head_controller')
        self.add_follow_joint_trajectory_server(namespace='/tiago/left_gripper_controller')
        self.add_follow_joint_trajectory_server(namespace='/tiago/right_gripper_controller')
        self.add_follow_joint_trajectory_server(namespace='/tiago/torso_controller')
        self.add_base_cmd_velocity(cmd_vel_topic='/tiago/cmd_vel',
                                   joint_name=self.drive_joint_name)


class TiagoStandaloneInterface(StandAloneRobotInterfaceConfig):

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
