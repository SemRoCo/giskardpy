from giskardpy.configs.data_types import ControlModes
from giskardpy.configs.default_config import Giskard


class PR2_Base(Giskard):
    def __init__(self):
        super().__init__()
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


class PR2_Mujoco(PR2_Base):
    def __init__(self):
        super().__init__()
        self.add_odometry_topic('/pr2_calibrated_with_ft2_without_virtual_joints/base_footprint')
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_follow_joint_trajectory_server(namespace='/pr2/whole_body_controller/follow_joint_trajectory',
                                                state_topic='/pr2/whole_body_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/pr2/l_gripper_l_finger_controller/follow_joint_trajectory',
                                                state_topic='/pr2/l_gripper_l_finger_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/pr2/r_gripper_l_finger_controller/follow_joint_trajectory',
                                                state_topic='/pr2/r_gripper_l_finger_controller/state')
        self.add_omni_drive_interface(cmd_vel_topic='/pr2_calibrated_with_ft2_without_virtual_joints/cmd_vel',
                                      parent_link_name='odom_combined',
                                      child_link_name='base_footprint')


class PR2_Real_Time(PR2_Base):
    def __init__(self):
        super().__init__()
        self.add_sync_tf_frame('map', 'odom_combined')
        # self.add_robot_from_parameter_server(parameter_name='giskard/robot_description',
        #                                      joint_state_topics=['base/joint_states',
        #                                                          'body/joint_states'])
        # self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/base/follow_joint_trajectory',
        #                                         state_topic='/whole_body_controller/base/state',
        #                                         fill_velocity_values=True)
        self.add_odometry_topic('/robot_pose_ekf/odom_combined')
        self.add_omni_drive_interface(cmd_vel_topic='/base_controller/command',
                                      parent_link_name='odom_combined',
                                      child_link_name='base_footprint',
                                      translation_jerk_limit=5,
                                      rotation_jerk_limit=5)
        self.add_follow_joint_trajectory_server(namespace='/l_arm_controller/follow_joint_trajectory',
                                                state_topic='/l_arm_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/r_arm_controller/follow_joint_trajectory',
                                                state_topic='/r_arm_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state',
                                                fill_velocity_values=True)

class PR2_Real(PR2_Base):
    def __init__(self):
        super().__init__()
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_robot_from_parameter_server(parameter_name='giskard/robot_description',
                                             joint_state_topics=['base/joint_states',
                                                                 'body/joint_states'])
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/base/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/base/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/l_arm_controller/follow_joint_trajectory',
                                                state_topic='/l_arm_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/r_arm_controller/follow_joint_trajectory',
                                                state_topic='/r_arm_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state',
                                                fill_velocity_values=True)



class PR2StandAlone(PR2_Base):
    def __init__(self):
        super().__init__()
        self.general_config.control_mode = ControlModes.stand_alone
        self.root_link_name = 'map'
        self.disable_tf_publishing()
        self.add_fixed_joint(parent_link='map', child_link='odom_combined')
        self.add_robot_from_parameter_server('robot_description')
        self.register_controlled_joints([
            'torso_lift_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'r_shoulder_pan_joint',
            'r_shoulder_lift_joint',
            'r_upper_arm_roll_joint',
            'r_forearm_roll_joint',
            'r_elbow_flex_joint',
            'r_wrist_flex_joint',
            'r_wrist_roll_joint',
            'l_shoulder_pan_joint',
            'l_shoulder_lift_joint',
            'l_upper_arm_roll_joint',
            'l_forearm_roll_joint',
            'l_elbow_flex_joint',
            'l_wrist_flex_joint',
            'l_wrist_roll_joint',
        ])
        self.add_omni_drive_interface(parent_link_name='odom_combined',
                                      child_link_name='base_footprint',
                                      translation_jerk_limit=5,
                                      rotation_jerk_limit=5)
