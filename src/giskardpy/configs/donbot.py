from typing import Optional

from std_msgs.msg import ColorRGBA

from giskardpy.configs.data_types import ControlModes
from giskardpy.configs.default_giskard import Giskard


class Donbot_Base(Giskard):
    def __init__(self, root_link_name: Optional[str] = None):
        super().__init__(root_link_name=root_link_name)
        self.load_moveit_self_collision_matrix('package://giskardpy/config/iai_donbot.srdf')
        self.ignore_self_collisions_of_pair('ur5_forearm_link', 'ur5_wrist_3_link')
        self.ignore_self_collisions_of_pair('ur5_base_link', 'ur5_upper_arm_link')
        self.add_self_collision('plate', 'ur5_upper_arm_link')
        self.set_default_external_collision_avoidance(soft_threshold=0.1,
                                                      hard_threshold=0.0)
        self.overwrite_external_collision_avoidance('odom_z_joint',
                                                    number_of_repeller=2,
                                                    soft_threshold=0.1,
                                                    hard_threshold=0.05)
        close_links = ['ur5_wrist_1_link', 'ur5_wrist_2_link', 'ur5_wrist_3_link', 'ur5_forearm_link',
                       'ur5_upper_arm_link']
        for link_name in close_links:
            self.overwrite_self_collision_avoidance(link_name,
                                                    soft_threshold=0.02,
                                                    hard_threshold=0.005)
        super_close_links = ['gripper_gripper_left_link', 'gripper_gripper_right_link']
        for link_name in super_close_links:
            self.overwrite_self_collision_avoidance(link_name,
                                                    soft_threshold=0.00001,
                                                    hard_threshold=0.0)

        self.set_default_joint_limits(velocity_limit=0.5,
                                      jerk_limit=15)
        self.overwrite_joint_velocity_limits(joint_name='odom_x_joint',
                                             velocity_limit=0.1)
        self.overwrite_joint_velocity_limits(joint_name='odom_y_joint',
                                             velocity_limit=0.1)
        self.overwrite_joint_velocity_limits(joint_name='odom_z_joint',
                                             velocity_limit=0.05)


class Donbot_IAI(Donbot_Base):

    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__()
        self._general_config.default_link_color = ColorRGBA(1, 1, 1, 0.7)
        self.load_moveit_self_collision_matrix('package://giskardpy/config/iai_donbot.srdf')
        self.add_sync_tf_frame('map', 'odom')
        self.add_follow_joint_trajectory_server(namespace='/whole_body_controller/base/follow_joint_trajectory',
                                                state_topic='/whole_body_controller/base/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                                                state_topic='/scaled_pos_joint_traj_controller/state',
                                                fill_velocity_values=True)


class Donbot_Standalone(Donbot_Base):

    def __init__(self):
        self.add_robot_from_parameter_server(add_drive_joint_to_group=False)
        super().__init__('map')
        self.set_default_visualization_marker_color(r=1, g=1, b=1, a=1)
        self.set_control_mode(ControlModes.stand_alone)
        self.publish_all_tf()
        self.configure_VisualizationBehavior(in_planning_loop=True)
        self.configure_CollisionMarker(in_planning_loop=True)
        self.add_fixed_joint(parent_link='map', child_link='iai_donbot/odom')
        self.register_controlled_joints([
            'ur5_elbow_joint',
            'ur5_shoulder_lift_joint',
            'ur5_shoulder_pan_joint',
            'ur5_wrist_1_joint',
            'ur5_wrist_2_joint',
            'ur5_wrist_3_joint',
            'odom_x_joint',
            'odom_y_joint',
            'odom_z_joint',
        ])
