from giskardpy.configs.default_config import Giskard


class Tiago(Giskard):

    def __init__(self):
        super().__init__()
        self.robot_interface_config.joint_state_topic = 'joint_states'
        self.add_sync_tf_frame('map', 'odom')
        self.set_odometry_topic('/mobile_base_controller/odom')
        self.add_follow_joint_trajectory_server(namespace='/arm_left_controller/follow_joint_trajectory',
                                                state_topic='/arm_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/arm_right_controller/follow_joint_trajectory',
                                                state_topic='/arm_right_controller/state')
        # self.add_follow_joint_trajectory_server(namespace='/head_controller/follow_joint_trajectory',
                                                # state_topic='/head_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state')
        self.add_diff_drive_interface(cmd_vel_topic='/mobile_base_controller/cmd_vel',
                                      parent_link_name='odom',
                                      child_link_name='base_footprint')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('ur5_forearm_link', 'ur5_wrist_3_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('ur5_base_link', 'ur5_upper_arm_link')
        # self.collision_avoidance_config.add_self_collision('plate', 'ur5_upper_arm_link')
        # self.collision_avoidance_config.set_default_external_collision_avoidance(soft_threshold=0.1,
        #                                                                          hard_threshold=0.0)
        # self.collision_avoidance_config.overwrite_external_collision_avoidance('brumbrum',
        #                                                                        number_of_repeller=2,
        #                                                                        soft_threshold=0.1,
        #                                                                        hard_threshold=0.05)
        # close_links = ['ur5_wrist_1_link', 'ur5_wrist_2_link', 'ur5_wrist_3_link', 'ur5_forearm_link',
        #                'ur5_upper_arm_link']
        # for link_name in close_links:
        #     self.collision_avoidance_config.overwrite_self_collision_avoidance(link_name,
        #                                                                        soft_threshold=0.02,
        #                                                                        hard_threshold=0.005)
        # super_close_links = ['gripper_gripper_left_link', 'gripper_gripper_right_link']
        # for link_name in super_close_links:
        #     self.collision_avoidance_config.overwrite_self_collision_avoidance(link_name,
        #                                                                        soft_threshold=0.00001,
        #                                                                        hard_threshold=0.0)
