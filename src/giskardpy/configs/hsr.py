from typing import Optional

from giskardpy.configs.data_types import ControlModes, SupportedQPSolver
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import PrefixName


class HSR_Base(Giskard):
    def __init__(self, root_link_name: Optional[str] = None):
        super().__init__(root_link_name=root_link_name)
        # self.set_qp_solver(SupportedQPSolver.gurobi)
        # self.configure_PlotTrajectory(enabled=True, wait=True)
        self.load_moveit_self_collision_matrix('package://giskardpy/config/hsrb.srdf')
        self.set_default_external_collision_avoidance(soft_threshold=0.05,
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
        # self.fix_joints_for_self_collision_avoidance(['head_pan_joint',
        #                                               'head_tilt_joint'])


class HSR_Mujoco(HSR_Base):
    def __init__(self):
        self.add_robot_from_parameter_server(parameter_name='hsrb4s/robot_description', joint_state_topics=['hsrb4s/joint_states'])
        super().__init__()
        self.add_sync_tf_frame('map', 'odom')
        self.add_omni_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  name='brumbrum',
                                  x_name=PrefixName('odom_x', self.get_default_group_name()),
                                  y_name=PrefixName('odom_y', self.get_default_group_name()),
                                  yaw_vel_name=PrefixName('odom_t', self.get_default_group_name()),
                                  odometry_topic='/hsrb4s/base_footprint')
        self.add_follow_joint_trajectory_server(namespace='/hsrb4s/arm_trajectory_controller/follow_joint_trajectory',
                                                state_topic='/hsrb4s/arm_trajectory_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/hsrb4s/head_trajectory_controller/follow_joint_trajectory',
                                                state_topic='/hsrb4s/head_trajectory_controller/state',
                                                fill_velocity_values=True)
        self.add_base_cmd_velocity(cmd_vel_topic='/hsrb4s/cmd_vel')
        self.overwrite_external_collision_avoidance(joint_name='brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.1,
                                                    hard_threshold=0.03)


class HSR_StandAlone(HSR_Base):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__('map')
        self.set_control_mode(ControlModes.stand_alone)
        self.set_default_visualization_marker_color(1, 1, 1, 1)
        self.publish_all_tf()
        self.configure_VisualizationBehavior(in_planning_loop=True)
        self.configure_CollisionMarker(in_planning_loop=True)
        self.add_fixed_joint(parent_link='map', child_link='odom')
        self.add_omni_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  name='brumbrum',
                                  x_name=PrefixName('odom_x', self.get_default_group_name()),
                                  y_name=PrefixName('odom_y', self.get_default_group_name()),
                                  yaw_vel_name=PrefixName('odom_t', self.get_default_group_name())
                                  )
        self.register_controlled_joints([
            'arm_flex_joint',
            'arm_lift_joint',
            'arm_roll_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            'brumbrum'
        ])
        self.overwrite_external_collision_avoidance(joint_name='brumbrum',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.1,
                                                    hard_threshold=0.03)
