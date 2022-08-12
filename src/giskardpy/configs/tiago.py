from giskardpy.configs.default_config import Giskard, ControlModes
from giskardpy.data_types import PrefixName


class TiagoMujoco(Giskard):
    def __init__(self):
        super().__init__()
        self.add_fixed_joint(parent_link=PrefixName('world', 'giskard'), child_link='map')
        self.add_sync_tf_frame('map', 'odom')
        self.add_odometry_topic('/tiago/base_footprint')
        self.add_robot_from_parameter_server()
        self.add_follow_joint_trajectory_server(namespace='/arm_left_controller/follow_joint_trajectory',
                                                state_topic='/arm_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/arm_right_controller/follow_joint_trajectory',
                                                state_topic='/arm_right_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/head_controller/follow_joint_trajectory',
                                                state_topic='/head_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/left_gripper_controller/follow_joint_trajectory',
                                                state_topic='/left_gripper_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/right_gripper_controller/follow_joint_trajectory',
                                                state_topic='/right_gripper_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state')
        # self.add_diff_drive_interface(cmd_vel_topic='/mobile_base_controller/cmd_vel',
        self.add_diff_drive_interface(cmd_vel_topic='/tiago/cmd_vel',
                                      parent_link_name='odom',
                                      child_link_name='base_footprint')


class IAI_Tiago(Giskard):
    def __init__(self):
        super().__init__()
        self.add_sync_tf_frame('map', 'odom')
        self.add_odometry_topic('/mobile_base_controller/odom')
        self.add_follow_joint_trajectory_server(namespace='/arm_left_controller/follow_joint_trajectory',
                                                state_topic='/arm_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/arm_right_controller/follow_joint_trajectory',
                                                state_topic='/arm_right_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/gripper_left_controller/follow_joint_trajectory',
                                                state_topic='/gripper_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/gripper_right_controller/follow_joint_trajectory',
                                                state_topic='/gripper_right_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/head_controller/follow_joint_trajectory',
                                                state_topic='/head_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state')
        self.add_diff_drive_interface(cmd_vel_topic='/mobile_base_controller/cmd_vel',
                                      parent_link_name='odom',
                                      child_link_name='base_footprint')


class TiagoStandAlone(Giskard):
    def __init__(self):
        super().__init__()
        self.general_config.control_mode = ControlModes.stand_alone
        self.add_fixed_joint(parent_link=PrefixName('world', 'giskard'), child_link='map')
        self.add_fixed_joint(parent_link='map', child_link='odom')
        self.add_robot_from_parameter_server('robot_description')
        self.register_controlled_joints(['torso_lift_joint', 'head_1_joint', 'head_2_joint'])
        self.register_controlled_joints(['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
                                         'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint'])
        self.register_controlled_joints(['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                                         'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                                         'arm_right_7_joint'])
        self.register_controlled_joints(['gripper_right_left_finger_joint', 'gripper_right_right_finger_joint',
                                         'gripper_left_left_finger_joint', 'gripper_left_right_finger_joint'])
        self.add_diff_drive_interface(parent_link_name='odom',
                                      child_link_name='base_footprint')
