import numpy as np

from giskardpy.configs.data_types import ControlModes, TfPublishingModes
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import PrefixName, Derivatives


class Tracy(Giskard):

    def configure_world(self):
        self.world.set_default_color(1, 1, 1, 0.8)
        self.world.set_default_limits({Derivatives.velocity: 0.2,
                                       Derivatives.acceleration: np.inf,
                                       Derivatives.jerk: 15})
        self.world.add_robot_from_parameter_server()

    def configure_collision_avoidance(self):
        self.collision_avoidance.load_moveit_self_collision_matrix('package://giskardpy/config/iai_tracy.srdf')


class TracyReal(Tracy):

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.open_loop)

    def configure_behavior_tree(self):
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True,
                                                              add_to_planning=True)

    def configure_robot_interface(self):
        self.robot_interface.sync_joint_state_topic('joint_states')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/left_arm/scaled_pos_joint_traj_controller_left/follow_joint_trajectory',
            state_topic='/left_arm/scaled_pos_joint_traj_controller_left/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/right_arm/scaled_pos_joint_traj_controller_right/follow_joint_trajectory',
            state_topic='/right_arm/scaled_pos_joint_traj_controller_right/state')


class TracyStandAlone(Tracy):

    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.stand_alone)

    def configure_behavior_tree(self):
        self.behavior_tree.add_tf_publisher(include_prefix=True, mode=TfPublishingModes.all)
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True,
                                                              add_to_control_loop=True)

    def configure_robot_interface(self):
        self.robot_interface.register_controlled_joints([
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
