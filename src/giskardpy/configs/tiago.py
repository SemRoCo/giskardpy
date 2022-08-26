from collections import defaultdict

from giskardpy.configs.data_types import CollisionCheckerLib
from giskardpy.configs.default_config import Giskard, ControlModes
from giskardpy.data_types import PrefixName


class TiagoBase(Giskard):
    def __init__(self):
        super().__init__()
        self.collision_avoidance_config.load_moveit_self_collision_matrix(
            'package://tiago_dual_moveit_config/config/srdf/tiago.srdf')
        self.collision_avoidance_config.overwrite_external_collision_avoidance('brumbrum',
                                                                               number_of_repeller=2,
                                                                               soft_threshold=0.2,
                                                                               hard_threshold=0.1)
        self.collision_avoidance_config.ignored_collisions = ['wheel_left_link',
                                                              'wheel_right_link',
                                                              'caster_back_left_2_link',
                                                              'caster_back_right_2_link',
                                                              'caster_front_left_2_link',
                                                              'caster_front_right_2_link']
        self.collision_avoidance_config.fix_joints_for_self_collision_avoidance(['head_1_joint',
                                                                                 'head_2_joint',
                                                                                 'gripper_left_left_finger_joint',
                                                                                 'gripper_left_right_finger_joint',
                                                                                 'gripper_right_left_finger_joint',
                                                                                 'gripper_right_right_finger_joint'])
        self.collision_avoidance_config.fix_joints_for_external_collision_avoidance(['gripper_left_left_finger_joint',
                                                                                     'gripper_left_right_finger_joint',
                                                                                     'gripper_right_left_finger_joint',
                                                                                     'gripper_right_right_finger_joint'])
        self.collision_avoidance_config.overwrite_external_collision_avoidance('arm_right_7_joint',
                                                                               number_of_repeller=4,
                                                                               soft_threshold=0.05,
                                                                               hard_threshold=0.0,
                                                                               max_velocity=0.2)
        self.collision_avoidance_config.overwrite_external_collision_avoidance('arm_left_7_joint',
                                                                               number_of_repeller=4,
                                                                               soft_threshold=0.05,
                                                                               hard_threshold=0.0,
                                                                               max_velocity=0.2)
        self.collision_avoidance_config.set_default_self_collision_avoidance(hard_threshold=0.03,
                                                                             soft_threshold=0.07)
        self.collision_avoidance_config.set_default_external_collision_avoidance(hard_threshold=0.03,
                                                                                 soft_threshold=0.08)
        # self.general_config.joint_limits['jerk'] = defaultdict(lambda: 60)
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_right_3_link', 'torso_lift_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_left_3_link', 'torso_lift_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_right_2_link', 'torso_lift_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_left_2_link', 'torso_lift_link')
        #
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_right_3_link', 'torso_fixed_column_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_left_3_link', 'torso_fixed_column_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_right_2_link', 'torso_fixed_column_link')
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_left_2_link', 'torso_fixed_column_link')
        #
        # self.collision_avoidance_config.ignore_self_collisions_of_pair('arm_left_2_link', 'torso_fixed_column_link')


class TiagoMujoco(TiagoBase):
    def __init__(self):
        super().__init__()
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
                                      child_link_name='base_footprint',
                                      translation_acceleration_limit=1,
                                      rotation_acceleration_limit=1)
        self.qp_solver_config.joint_weights['velocity']['brumbrum'] = 0.1


class IAI_Tiago(TiagoBase):
    def __init__(self):
        super().__init__()
        self.add_sync_tf_frame('map', 'odom')
        self.add_odometry_topic('/mobile_base_controller/odom')
        self.add_robot_from_parameter_server()
        self.add_follow_joint_trajectory_server(
            namespace='/arm_left_impedance_controller/follow_joint_trajectory',
            state_topic='/arm_left_impedance_controller/state'
            # namespace='/arm_left_controller/follow_joint_trajectory',
            # state_topic='/arm_left_controller/state'
        )
        self.add_follow_joint_trajectory_server(
            namespace='/arm_right_impedance_controller/follow_joint_trajectory',
            state_topic='/arm_right_impedance_controller/state'
            # namespace='/arm_right_controller/follow_joint_trajectory',
            # state_topic='/arm_right_controller/state'
        )
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
                                      child_link_name='base_footprint',
                                      translation_acceleration_limit=0.5,
                                      rotation_acceleration_limit=None)


class TiagoStandAlone(TiagoBase):
    def __init__(self):
        super().__init__()
        self.general_config.control_mode = ControlModes.stand_alone
        self.root_link_name = 'map'
        # self.collision_avoidance_config.collision_checker = CollisionCheckerLib.none
        # self.disable_visualization()
        self.disable_tf_publishing()
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
                                      child_link_name='base_footprint',
                                      translation_velocity_limit=0.19,
                                      rotation_velocity_limit=0.19)
