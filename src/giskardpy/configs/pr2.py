from typing import Optional

from giskardpy.configs.data_types import ControlModes, CollisionCheckerLib, SupportedQPSolver
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import Derivatives


class PR2_Base(Giskard):
    def __init__(self, root_link_name: Optional[str] = None):
        super().__init__(root_link_name=root_link_name)
        # self.set_collision_checker(CollisionCheckerLib.none)
        # self.set_qp_solver(SupportedQPSolver.qpalm)
        # self.configure_PlotTrajectory(enabled=True, wait=True)
        # self.configure_PlotDebugExpressions(enabled=True, wait=True)
        # self.configure_DebugMarkerPublisher(enabled=True)
        # self.configure_PublishDebugExpressions(
        #     publish_lb=True,
        #     publish_ub=True,
        #     publish_lbA=True,
        #     publish_ubA=True,
        #     publish_bE=True,
        #     publish_Ax=True,
        #     publish_Ex=True,
        #     publish_xdot=True,
        #     publish_weights=True,
        #     publish_debug=True,
        # )
        self.configure_MaxTrajectoryLength(length=30)
        self.load_moveit_self_collision_matrix('package://giskardpy/config/pr2.srdf')
        self.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                      hard_threshold=0.0)
        for joint_name in ['r_wrist_roll_joint', 'l_wrist_roll_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        number_of_repeller=4,
                                                        soft_threshold=0.05,
                                                        hard_threshold=0.0,
                                                        max_velocity=0.2)
        for joint_name in ['r_wrist_flex_joint', 'l_wrist_flex_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        number_of_repeller=2,
                                                        soft_threshold=0.05,
                                                        hard_threshold=0.0,
                                                        max_velocity=0.2)
        for joint_name in ['r_elbow_flex_joint', 'l_elbow_flex_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        soft_threshold=0.05,
                                                        hard_threshold=0.0)
        for joint_name in ['r_forearm_roll_joint', 'l_forearm_roll_joint']:
            self.overwrite_external_collision_avoidance(joint_name,
                                                        soft_threshold=0.025,
                                                        hard_threshold=0.0)
        self.ignore_all_collisions_of_links(['bl_caster_l_wheel_link', 'bl_caster_r_wheel_link',
                                             'fl_caster_l_wheel_link', 'fl_caster_r_wheel_link',
                                             'br_caster_l_wheel_link', 'br_caster_r_wheel_link',
                                             'fr_caster_l_wheel_link', 'fr_caster_r_wheel_link'])
        self.fix_joints_for_self_collision_avoidance(['head_pan_joint',
                                                      'head_tilt_joint',
                                                      'r_gripper_l_finger_joint',
                                                      'l_gripper_l_finger_joint'])
        self.fix_joints_for_external_collision_avoidance(['r_gripper_l_finger_joint',
                                                          'l_gripper_l_finger_joint'])
        # self.set_maximum_derivative(Derivatives.acceleration)
        # self.set_default_joint_limits(velocity_limit=1,
        #                               acceleration_limit=1.5)
        # self.overwrite_joint_velocity_limits(joint_name='head_pan_joint',
        #                                      velocity_limit=2)
        # self.overwrite_joint_acceleration_limits(joint_name='head_pan_joint',
        #                                          acceleration_limit=4)
        # self.overwrite_joint_velocity_limits(joint_name='head_tilt_joint',
        #                                      velocity_limit=2)
        # self.overwrite_joint_acceleration_limits(joint_name='head_tilt_joint',
        #                                          acceleration_limit=4)
        # self.set_default_weights(velocity_weight=0.01,
        #                          acceleration_weight=0.01)


class PR2_Mujoco(PR2_Base):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__()
        self.set_default_visualization_marker_color(1, 1, 1, 0.7)
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_omni_drive_joint(name='brumbrum',
                                  parent_link_name='odom_combined',
                                  child_link_name='base_footprint',
                                  translation_limits={
                                      Derivatives.velocity: 0.4,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  },
                                  odometry_topic='/pr2/base_footprint')
        self.add_follow_joint_trajectory_server(namespace='/pr2/whole_body_controller/follow_joint_trajectory',
                                                state_topic='/pr2/whole_body_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/pr2/l_gripper_l_finger_controller/follow_joint_trajectory',
                                                state_topic='/pr2/l_gripper_l_finger_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/pr2/r_gripper_l_finger_controller/follow_joint_trajectory',
                                                state_topic='/pr2/r_gripper_l_finger_controller/state')
        self.add_base_cmd_velocity(cmd_vel_topic='/pr2/cmd_vel',
                                   track_only_velocity=True)
        self.overwrite_external_collision_avoidance('brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)


class PR2_IAI(PR2_Base):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__()
        self.set_default_visualization_marker_color(20 / 255, 27.1 / 255, 80 / 255, 0.2)
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_omni_drive_joint(name='brumbrum',
                                  parent_link_name='odom_combined',
                                  child_link_name='base_footprint',
                                  translation_limits={
                                      Derivatives.velocity: 0.4,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  },
                                  odometry_topic='/robot_pose_ekf/odom_combined')
        fill_velocity_values = False
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
        self.add_base_cmd_velocity(cmd_vel_topic='/base_controller/command',
                                   track_only_velocity=True)
        self.overwrite_external_collision_avoidance('brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)


class PR2_Unreal(PR2_Base):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__()
        # self.general_config.default_link_color = ColorRGBA(20/255, 27.1/255, 80/255, 0.2)
        # self.collision_avoidance_config.collision_checker = self.collision_avoidance_config.collision_checker.none
        self.add_sync_tf_frame('map', 'odom_combined')
        self.add_omni_drive_joint(name='brumbrum',
                                  parent_link_name='odom_combined',
                                  child_link_name='base_footprint',
                                  translation_limits={
                                      Derivatives.velocity: 0.4,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  },
                                  odometry_topic='/base_odometry/odom')
        fill_velocity_values = False
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/state',
                                                fill_velocity_values=fill_velocity_values)
        self.add_base_cmd_velocity(cmd_vel_topic='/base_controller/command',
                                   track_only_velocity=True)
        self.overwrite_external_collision_avoidance('brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)


class PR2_StandAlone(PR2_Base):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__('map')
        self.set_default_visualization_marker_color(1, 1, 1, 0.8)
        self.set_control_mode(ControlModes.stand_alone)
        self.publish_all_tf()
        self.configure_VisualizationBehavior(in_planning_loop=True)
        self.configure_CollisionMarker(in_planning_loop=True)
        # self.configure_VisualizationBehavior(enabled=False)
        # self.configure_CollisionMarker(enabled=False)
        self.add_fixed_joint(parent_link='map', child_link='odom_combined')
        self.add_omni_drive_joint(name='brumbrum',
                                  parent_link_name='odom_combined',
                                  child_link_name='base_footprint',
                                  translation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 6,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 6
                                  })
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
            'brumbrum'
        ])
        self.overwrite_external_collision_avoidance('brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)
