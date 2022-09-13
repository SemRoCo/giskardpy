from collections import defaultdict

from std_msgs.msg import ColorRGBA

from giskardpy.configs.data_types import ControlModes
from giskardpy.configs.default_config import Giskard


class PR2_Base(Giskard):
    def __init__(self):
        super().__init__()
        self.collision_avoidance_config.load_moveit_self_collision_matrix('package://giskardpy/config/pr2.srdf')
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
        self.collision_avoidance_config.fix_joints_for_self_collision_avoidance(['head_pan_joint',
                                                                                 'head_tilt_joint'])
        # self.general_config.joint_limits = {
        #     'velocity': defaultdict(lambda: 0.5),
        #     'acceleration': defaultdict(lambda: 1e3),
        #     'jerk': defaultdict(lambda: 10)
        # }
        # self.qp_solver_config.joint_weights = {
        #     'velocity': defaultdict(lambda: 0.001),
        #     'acceleration': defaultdict(float),
        #     'jerk': defaultdict(lambda: 0.001)
        # }
        self.general_config.joint_limits = {
            'velocity': defaultdict(lambda: 1),
            'acceleration': defaultdict(lambda: 1.5),
        }
        self.qp_solver_config.joint_weights = {
            'velocity': defaultdict(lambda: 0.001),
            'acceleration': defaultdict(lambda: 0.001),
        }
        self.general_config.joint_limits['velocity']['head_pan_joint'] = 2
        self.general_config.joint_limits['velocity']['head_tilt_joint'] = 2
        self.general_config.joint_limits['acceleration']['head_pan_joint'] = 4
        self.general_config.joint_limits['acceleration']['head_tilt_joint'] = 4
        # self.general_config.joint_limits['jerk']['head_pan_joint'] = 30
        # self.general_config.joint_limits['jerk']['head_tilt_joint'] = 30


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


class PR2_IAI(PR2_Base):
    def __init__(self):
        super().__init__()
        self.general_config.default_link_color = ColorRGBA(20/255, 27.1/255, 80/255, 0.2)
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_odometry_topic('/robot_pose_ekf/odom_combined')
        self.add_omni_drive_interface(cmd_vel_topic='/base_controller/command',
                                      parent_link_name='odom_combined',
                                      child_link_name='base_footprint',
                                      translation_acceleration_limit=0.25,
                                      rotation_acceleration_limit=0.25,
                                      translation_jerk_limit=5,
                                      rotation_jerk_limit=5)
        fill_velocity_values = True
        self.add_follow_joint_trajectory_server(namespace='/l_arm_controller/follow_joint_trajectory',
                                                state_topic='/l_arm_controller/state',
                                                fill_velocity_values=fill_velocity_values)
        self.add_follow_joint_trajectory_server(namespace='/r_arm_controller/follow_joint_trajectory',
                                                state_topic='/r_arm_controller/state',
                                                fill_velocity_values=fill_velocity_values)
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state',
                                                fill_velocity_values=fill_velocity_values)
        self.add_follow_joint_trajectory_server(namespace='/head_traj_controller/follow_joint_trajectory',
                                                state_topic='/head_traj_controller/state',
                                                fill_velocity_values=fill_velocity_values)
        self.collision_avoidance_config.overwrite_external_collision_avoidance('brumbrum',
                                                                               number_of_repeller=2,
                                                                               soft_threshold=0.2,
                                                                               hard_threshold=0.1)


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
