from giskardpy.configs.default_config import Giskard
from giskardpy.data_types import PrefixName


class PR22(Giskard):
    def __init__(self):
        super().__init__()
        self.add_robot_from_parameter_server(parameter_name='pr2_a/robot_description',
                                             joint_state_topics=['pr2_a/joint_states'],
                                             group_name='pr2_a')
        self.add_robot_from_parameter_server(parameter_name='pr2_b/robot_description',
                                             joint_state_topics=['pr2_b/joint_states'],
                                             group_name='pr2_b')
        self.add_sync_tf_frame('map', PrefixName('odom_combined', 'pr2_a'))
        self.add_sync_tf_frame('map', PrefixName('odom_combined', 'pr2_b'))
        self.add_follow_joint_trajectory_server(namespace='/pr2_a/whole_body_controller/base/follow_joint_trajectory',
                                                state_topic='/pr2_a/whole_body_controller/base/state',
                                                group_name='pr2_a')
        self.add_follow_joint_trajectory_server(namespace='/pr2_a/whole_body_controller/body/follow_joint_trajectory',
                                                state_topic='/pr2_a/whole_body_controller/body/state',
                                                group_name='pr2_a')
        self.add_follow_joint_trajectory_server(namespace='/pr2_b/whole_body_controller/base/follow_joint_trajectory',
                                                state_topic='/pr2_b/whole_body_controller/base/state',
                                                group_name='pr2_b')
        self.add_follow_joint_trajectory_server(namespace='/pr2_b/whole_body_controller/body/follow_joint_trajectory',
                                                state_topic='/pr2_b/whole_body_controller/body/state',
                                                group_name='pr2_b')
        # self.add_omni_drive_interface(cmd_vel_topic='/pr2_calibrated_with_ft2_without_virtual_joints/cmd_vel',
        #                               parent_link_name='odom_combined',
        #                               child_link_name='base_footprint',
        #                               odometry_topic='/pr2_calibrated_with_ft2_without_virtual_joints/base_footprint')

        # link_to_ignore = [PrefixName('bl_caster_l_wheel_link', self.get_default_group_name()),
        #                   PrefixName('bl_caster_r_wheel_link', self.get_default_group_name()),
        #                   PrefixName('bl_caster_rotation_link', self.get_default_group_name()),
        #                   PrefixName('br_caster_l_wheel_link', self.get_default_group_name()),
        #                   PrefixName('br_caster_r_wheel_link', self.get_default_group_name()),
        #                   PrefixName('br_caster_rotation_link', self.get_default_group_name()),
        #                   PrefixName('fl_caster_l_wheel_link', self.get_default_group_name()),
        #                   PrefixName('fl_caster_r_wheel_link', self.get_default_group_name()),
        #                   PrefixName('fl_caster_rotation_link', self.get_default_group_name()),
        #                   PrefixName('fr_caster_l_wheel_link', self.get_default_group_name()),
        #                   PrefixName('fr_caster_r_wheel_link', self.get_default_group_name()),
        #                   PrefixName('fr_caster_rotation_link', self.get_default_group_name()),
        #                   PrefixName('l_shoulder_lift_link', self.get_default_group_name()),
        #                   PrefixName('r_shoulder_lift_link', self.get_default_group_name()),
        #                   PrefixName('base_link', self.get_default_group_name())]
        # for link_name in link_to_ignore:
        #     self.collision_avoidance_config.ignore_all_self_collisions_of_link(link_name)
        # self.collision_avoidance_config.set_default_external_collision_avoidance(soft_threshold=0.1,
        #                                                                          hard_threshold=0.0)
        # for joint_name in [PrefixName('r_wrist_roll_joint', self.get_default_group_name()),
        #                    PrefixName('l_wrist_roll_joint', self.get_default_group_name())]:
        #     self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
        #                                                                            number_of_repeller=4,
        #                                                                            soft_threshold=0.05,
        #                                                                            hard_threshold=0.0,
        #                                                                            max_velocity=0.2)
        # for joint_name in [PrefixName('r_wrist_flex_joint', self.get_default_group_name()),
        #                    PrefixName('l_wrist_flex_joint', self.get_default_group_name())]:
        #     self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
        #                                                                            number_of_repeller=2,
        #                                                                            soft_threshold=0.05,
        #                                                                            hard_threshold=0.0,
        #                                                                            max_velocity=0.2)
        # for joint_name in [PrefixName('r_elbow_flex_joint', self.get_default_group_name()),
        #                    PrefixName('l_elbow_flex_joint', self.get_default_group_name())]:
        #     self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
        #                                                                            soft_threshold=0.05,
        #                                                                            hard_threshold=0.0)
        # for joint_name in [PrefixName('r_forearm_roll_joint', self.get_default_group_name()),
        #                    PrefixName('l_forearm_roll_joint', self.get_default_group_name())]:
        #     self.collision_avoidance_config.overwrite_external_collision_avoidance(joint_name,
        #                                                                            soft_threshold=0.025,
        #                                                                            hard_threshold=0.0)

