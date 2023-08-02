import numpy as np

from giskardpy.configs.data_types import ControlModes, TfPublishingModes
from giskardpy.configs.giskard import Giskard
from giskardpy.my_types import PrefixName, Derivatives


class Tracy(Giskard):

    def configure_world(self):
        self.world_config.set_default_color(1, 1, 1, 0.8)
        self.world_config.set_default_limits({Derivatives.velocity: 0.2,
                                              Derivatives.acceleration: np.inf,
                                              Derivatives.jerk: 15})
        self.world_config.add_robot_from_parameter_server()

    def configure_collision_avoidance(self):
        self.collision_avoidance_config.load_self_collision_matrix('package://giskardpy/self_collision_matrices/iai/tracy.srdf')


class TracyReal(Tracy):

    def configure_execution(self):
        self.qp_controller_config.set_control_mode(ControlModes.open_loop)

    def configure_behavior_tree(self):
        self.behavior_tree_config.add_visualization_marker_publisher(add_to_sync=True,
                                                                     add_to_planning=True)

    def configure_robot_interface(self):
        self.robot_interface_config.sync_joint_state_topic('joint_states')
        self.robot_interface_config.add_follow_joint_trajectory_server(
            namespace='/left_arm/scaled_pos_joint_traj_controller_left/follow_joint_trajectory',
            state_topic='/left_arm/scaled_pos_joint_traj_controller_left/state')
        self.robot_interface_config.add_follow_joint_trajectory_server(
            namespace='/right_arm/scaled_pos_joint_traj_controller_right/follow_joint_trajectory',
            state_topic='/right_arm/scaled_pos_joint_traj_controller_right/state')


class TracyStandAlone(Tracy):

    def configure_execution(self):
        self.qp_controller_config.set_control_mode(ControlModes.standalone)

    def configure_behavior_tree(self):
        self.behavior_tree_config.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)
        self.behavior_tree_config.add_visualization_marker_publisher(add_to_sync=True,
                                                                     add_to_control_loop=True)

    def configure_robot_interface(self):
        self.robot_interface_config.register_controlled_joints([
            'left_shoulder_pan_joint',
            'left_shoulder_lift_joint',
            'left_elbow_joint',
            'left_wrist_1_joint',
            'left_wrist_2_joint',
            'left_wrist_3_joint',
            'right_shoulder_pan_joint',
            'right_shoulder_lift_joint',
            'right_elbow_joint',
            'right_wrist_1_joint',
            'right_wrist_2_joint',
            'right_wrist_3_joint',
        ])
