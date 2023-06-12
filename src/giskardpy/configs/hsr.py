import numpy as np

from giskardpy.configs.data_types import ControlModes, SupportedQPSolver
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import PrefixName, Derivatives


class HSR_Base(Giskard):
    localization_joint_name = 'localization'
    drive_joint_name = 'brumbrum'
    odom_link_name = 'odom'
    map_name = 'map'

    def configure_world(self):
        self.world.set_default_visualization_marker_color(1, 1, 1, 1)
        self.world.set_default_limits({Derivatives.velocity: 1,
                                       Derivatives.acceleration: np.inf,
                                       Derivatives.jerk: 30})
        self.world.set_root_link_name(self.map_name)
        self.world.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                                  joint_name=self.localization_joint_name)
        self.world.add_empty_link(self.odom_link_name)
        pr2_group_name = self.world.add_robot_from_parameter_server()
        root_link_name = self.world.get_root_link_of_group(pr2_group_name)
        self.world.add_omni_drive_joint(parent_link_name=self.odom_link_name,
                                        child_link_name=root_link_name,
                                        name=self.drive_joint_name,
                                        x_name=PrefixName('odom_x', pr2_group_name),
                                        y_name=PrefixName('odom_y', pr2_group_name),
                                        yaw_vel_name=PrefixName('odom_t', pr2_group_name),
                                        translation_limits={
                                            Derivatives.velocity: 0.2,
                                            Derivatives.acceleration: 1,
                                            Derivatives.jerk: 6,
                                        },
                                        rotation_limits={
                                            Derivatives.velocity: 0.2,
                                            Derivatives.acceleration: 1,
                                            Derivatives.jerk: 6
                                        },
                                        robot_group_name=pr2_group_name)

    def configure_collision_avoidance(self):
        # self.collision_avoidance.set_collision_checker(CollisionCheckerLib.none)
        self.collision_avoidance.load_moveit_self_collision_matrix('package://giskardpy/config/hsrb.srdf')
        self.collision_avoidance.set_default_external_collision_avoidance(soft_threshold=0.05,
                                                                          hard_threshold=0.0)
        self.collision_avoidance.overwrite_external_collision_avoidance('wrist_roll_joint',
                                                                        number_of_repeller=4,
                                                                        soft_threshold=0.05,
                                                                        hard_threshold=0.0,
                                                                        max_velocity=0.2)
        self.collision_avoidance.overwrite_external_collision_avoidance(joint_name='brumbrum',
                                                                        number_of_repeller=2,
                                                                        soft_threshold=0.1,
                                                                        hard_threshold=0.03)

    def configure_robot_interface(self):
        pass


class HSR_Mujoco(HSR_Base):
    def __init__(self):
        self.add_robot_from_parameter_server(parameter_name='hsrb4s/robot_description',
                                             joint_state_topics=['hsrb4s/joint_states'])
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

    def configure_execution(self):
        self.execution_config.set_control_mode(ControlModes.stand_alone)

    def configure_behavior_tree(self):
        self.behavior_tree.configure_VisualizationBehavior(add_to_sync=True,
                                                           add_to_control_loop=True)

    def configure_robot_interface(self):
        super().configure_robot_interface()
        self.robot_interface.register_controlled_joints([
            'arm_flex_joint',
            'arm_lift_joint',
            'arm_roll_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            self.drive_joint_name,
        ])
