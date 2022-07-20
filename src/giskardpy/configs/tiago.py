from giskardpy.configs.default_config import Giskard


class Tiago(Giskard):
    def __init__(self):
        super().__init__()
        self.robot_interface_config.joint_state_topic = 'joint_states'
        self.add_sync_tf_frame('map', 'odom')
        # self.set_odometry_topic('/mobile_base_controller/odom')
        self.set_odometry_topic('/tiago/base_footprint')
        self.add_follow_joint_trajectory_server(namespace='/arm_left_controller/follow_joint_trajectory',
                                                state_topic='/arm_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/arm_right_controller/follow_joint_trajectory',
                                                state_topic='/arm_right_controller/state')
        # self.add_follow_joint_trajectory_server(namespace='/head_controller/follow_joint_trajectory',
        # state_topic='/head_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state')
        # self.add_diff_drive_interface(cmd_vel_topic='/mobile_base_controller/cmd_vel',
        self.add_diff_drive_interface(cmd_vel_topic='/tiago/cmd_vel',
                                      parent_link_name='odom',
                                      child_link_name='base_footprint')
