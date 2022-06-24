from giskardpy.configs.default_config import Giskard


class PR2(Giskard):

    def __init__(self):
        super().__init__()
        self.set_odometry_topic('/pr2_calibrated_with_ft2_without_virtual_joints/base_footprint')
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_follow_joint_trajectory_server(namespace='/pr2/whole_body_controller/follow_joint_trajectory',
                                                state_topic='/pr2/whole_body_controller/state')
        self.add_omni_drive_interface(cmd_vel_topic='/pr2_calibrated_with_ft2_without_virtual_joints/cmd_vel',
                                      parent_link_name='odom_combined',
                                      child_link_name='base_footprint')

        link_to_ignore = ['bl_caster_l_wheel_link',
                          'bl_caster_r_wheel_link',
                          'bl_caster_rotation_link',
                          'br_caster_l_wheel_link',
                          'br_caster_r_wheel_link',
                          'br_caster_rotation_link',
                          'fl_caster_l_wheel_link',
                          'fl_caster_r_wheel_link',
                          'fl_caster_rotation_link',
                          'fr_caster_l_wheel_link',
                          'fr_caster_r_wheel_link',
                          'fr_caster_rotation_link',
                          'l_shoulder_lift_link',
                          'r_shoulder_lift_link',
                          'base_link']
        for link_name in link_to_ignore:
            self.collision_avoidance_config.ignore_all_self_collisions_of_link(link_name)
        self.collision_avoidance_config.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                                                 hard_threshold=0.0)
        for joint_name in ['r_wrist_roll_joint', 'l_wrist_roll_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   number_of_repeller=4,
                                                                                   soft_threshold=0.05,
                                                                                   hard_threshold=0.0,
                                                                                   max_velocity=0.2)
        for joint_name in ['r_wrist_flex_joint', 'l_wrist_flex_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   number_of_repeller=2,
                                                                                   soft_threshold=0.05,
                                                                                   hard_threshold=0.0,
                                                                                   max_velocity=0.2)
        for joint_name in ['r_elbow_flex_joint', 'l_elbow_flex_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   soft_threshold=0.05,
                                                                                   hard_threshold=0.0)
        for joint_name in ['r_forearm_roll_joint', 'l_forearm_roll_joint']:
            self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
                                                                                   soft_threshold=0.025,
                                                                                   hard_threshold=0.0)
