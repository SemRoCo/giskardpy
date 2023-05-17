from giskardpy.configs.data_types import ControlModes
from giskardpy.configs.default_giskard import Giskard
from giskardpy.my_types import PrefixName


class Tracebot(Giskard):
    def __init__(self):
        super().__init__(root_link_name='world')
        self.add_fixed_joint('world', 'tracy/world')


class TracebotMujoco(Tracebot):
    def __init__(self):
        self.add_robot_from_parameter_server(joint_state_topics=['tracebot/joint_states'])
        super().__init__()
        self.add_follow_joint_trajectory_server(namespace='/tracebot/whole_body_controller/follow_joint_trajectory',
                                                state_topic='/tracebot/whole_body_controller/state')


class TracyReal(Tracebot):
    def __init__(self):
        self.add_robot_from_parameter_server(joint_state_topics=['joint_states'], add_drive_joint_to_group=False)
        super().__init__()
        self.add_follow_joint_trajectory_server(
            namespace='/left_arm/scaled_pos_joint_traj_controller_left/follow_joint_trajectory',
            state_topic='/left_arm/scaled_pos_joint_traj_controller_left/state')
        self.add_follow_joint_trajectory_server(
            namespace='/right_arm/scaled_pos_joint_traj_controller_right/follow_joint_trajectory',
            state_topic='/right_arm/scaled_pos_joint_traj_controller_right/state')
        self.set_default_joint_limits(velocity_limit=0.2)


class Tracebot_StandAlone(Tracebot):
    def __init__(self):
        self.add_robot_from_parameter_server(add_drive_joint_to_group=False)
        super().__init__()
        self.set_default_visualization_marker_color(1, 1, 1, 0.8)
        self.set_control_mode(ControlModes.stand_alone)
        self.publish_all_tf()
        self.configure_VisualizationBehavior(in_planning_loop=True)
        self.configure_CollisionMarker(in_planning_loop=True)
        self.register_controlled_joints([
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
