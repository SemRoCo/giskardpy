from typing import Optional

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.configs.default_giskard import Giskard, ControlModes
from giskardpy.my_types import Derivatives


class TiagoBase(Giskard):
    def __init__(self, root_link_name: Optional[str] = None):
        super().__init__(root_link_name=root_link_name)
        self.set_qp_solver(SupportedQPSolver.gurobi)
        self.set_default_visualization_marker_color(1, 1, 1, 0.7)
        self.configure_PublishDebugExpressions(
            publish_lb=True,
            publish_ub=True,
        #     publish_lbA=True,
        #     publish_ubA=True,
        #     publish_bE=True,
        #     publish_Ax=True,
        #     publish_Ex=True,
            publish_xdot=True,
        #     publish_weights=True,
        #     publish_g=True,
            publish_debug=True,
        )
        self.configure_PlotDebugExpressions(enabled=True, wait=True)
        self.configure_PlotTrajectory(enabled=True, wait=True)
        self.load_moveit_self_collision_matrix('package://giskardpy/config/tiago.srdf')
        self.overwrite_external_collision_avoidance('brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)
        self.ignored_collisions = ['wheel_left_link',
                                   'wheel_right_link',
                                   'caster_back_left_2_link',
                                   'caster_back_right_2_link',
                                   'caster_front_left_2_link',
                                   'caster_front_right_2_link']
        self.fix_joints_for_self_collision_avoidance(['head_1_joint',
                                                      'head_2_joint',
                                                      'gripper_left_left_finger_joint',
                                                      'gripper_left_right_finger_joint',
                                                      'gripper_right_left_finger_joint',
                                                      'gripper_right_right_finger_joint'])
        self.fix_joints_for_external_collision_avoidance(['gripper_left_left_finger_joint',
                                                          'gripper_left_right_finger_joint',
                                                          'gripper_right_left_finger_joint',
                                                          'gripper_right_right_finger_joint'])
        self.overwrite_external_collision_avoidance('arm_right_7_joint',
                                                    number_of_repeller=4,
                                                    soft_threshold=0.05,
                                                    hard_threshold=0.0,
                                                    max_velocity=0.2)
        self.overwrite_external_collision_avoidance('arm_left_7_joint',
                                                    number_of_repeller=4,
                                                    soft_threshold=0.05,
                                                    hard_threshold=0.0,
                                                    max_velocity=0.2)
        self.set_default_self_collision_avoidance(hard_threshold=0.04,
                                                  soft_threshold=0.08)
        self.set_default_external_collision_avoidance(hard_threshold=0.03,
                                                      soft_threshold=0.08)


class TiagoMujoco(TiagoBase):
    def __init__(self):
        self.add_robot_from_parameter_server(joint_state_topics=['/tiago/joint_states'])
        super().__init__()
        self.add_sync_tf_frame('map', 'odom')
        self.add_diff_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  odometry_topic='/tiago/base_footprint',
                                  name='brumbrum',
                                  translation_limits={
                                      Derivatives.velocity: 0.4,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  })
        self.add_follow_joint_trajectory_server(namespace='/tiago/arm_left_controller/follow_joint_trajectory',
                                                state_topic='/tiago/arm_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/tiago/arm_right_controller/follow_joint_trajectory',
                                                state_topic='/tiago/arm_right_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/tiago/head_controller/follow_joint_trajectory',
                                                state_topic='/tiago/head_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/tiago/left_gripper_controller/follow_joint_trajectory',
                                                state_topic='/tiago/left_gripper_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/tiago/right_gripper_controller/follow_joint_trajectory',
                                                state_topic='/tiago/right_gripper_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/tiago/torso_controller/follow_joint_trajectory',
                                                state_topic='/tiago/torso_controller/state')
        self.add_base_cmd_velocity(cmd_vel_topic='/tiago/cmd_vel')


class IAI_Tiago(TiagoBase):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__()
        self.add_sync_tf_frame('map', 'odom')
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
        self.add_diff_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  odometry_topic='/mobile_base_controller/odom',
                                  name='brumbrum',
                                  translation_limits={
                                      Derivatives.velocity: 0.4,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  })
        self.add_base_cmd_velocity(cmd_vel_topic='/mobile_base_controller/cmd_vel')


class Tiago_Standalone(TiagoBase):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__('map')
        self.set_default_visualization_marker_color(1, 1, 1, 1)
        self.set_control_mode(ControlModes.stand_alone)
        self.publish_all_tf()
        self.configure_VisualizationBehavior(in_planning_loop=True)
        self.configure_CollisionMarker(in_planning_loop=True)
        self.add_fixed_joint(parent_link='map', child_link='odom')
        self.add_diff_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  name='brumbrum',
                                  translation_limits={
                                      Derivatives.velocity: 0.4,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: 1,
                                      Derivatives.jerk: 5
                                  })
        self.register_controlled_joints(['torso_lift_joint', 'head_1_joint', 'head_2_joint', 'brumbrum'])
        self.register_controlled_joints(['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
                                         'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint'])
        self.register_controlled_joints(['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                                         'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                                         'arm_right_7_joint'])
        self.register_controlled_joints(['gripper_right_left_finger_joint', 'gripper_right_right_finger_joint',
                                         'gripper_left_left_finger_joint', 'gripper_left_right_finger_joint'])
