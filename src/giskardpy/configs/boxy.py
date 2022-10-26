from collections import defaultdict

from std_msgs.msg import ColorRGBA

from giskardpy.configs.default_config import Giskard


class Boxy(Giskard):

    def __init__(self):
        super().__init__()
        self._general_config.default_link_color = ColorRGBA(1, 1, 1, 0.7)
        self.add_sync_tf_frame('map', 'odom')
        # self.set_odometry_topic('/donbot/base_footprint')
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/base/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/base/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/left_arm/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/left_arm/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/right_arm/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/right_arm/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/neck/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/neck/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/torso/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/torso/state',
                                                fill_velocity_values=True)
        # self.add_omni_drive_interface(cmd_vel_topic='/donbot/cmd_vel',
        #                               parent_link_name='odom',
        #                               child_link_name='base_footprint')
        self.collision_avoidance_config.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                                                 hard_threshold=0.0)
        self.collision_avoidance_config.overwrite_external_collision_avoidance('odom_z_joint',
                                                                               number_of_repeller=2,
                                                                               soft_threshold=0.1,
                                                                               hard_threshold=0.05)
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
        # self.general_config.joint_limits = {
        #     'velocity': defaultdict(lambda: 0.5),
        #     'acceleration': defaultdict(lambda: 1e3),
        #     'jerk': defaultdict(lambda: 15),
        # }
        # self.general_config.joint_limits['velocity']['odom_x_joint'] = 0.1
        # self.general_config.joint_limits['velocity']['odom_y_joint'] = 0.1
        # self.general_config.joint_limits['velocity']['odom_z_joint'] = 0.05
