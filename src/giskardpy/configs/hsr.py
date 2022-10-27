from collections import defaultdict

from std_msgs.msg import ColorRGBA

from giskardpy.configs.data_types import ControlModes
from giskardpy.configs.default_config import Giskard


class HSR_Base(Giskard):
    def __init__(self):
        super().__init__()
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
        self.fix_joints_for_self_collision_avoidance(['head_pan_joint',
                                                      'head_tilt_joint'])


class HSR_Mujoco(HSR_Base):
    def __init__(self):
        self.add_robot_from_parameter_server(joint_state_topics=['hsrb4s/joint_states'])
        super().__init__()
        self.add_sync_tf_frame('map', 'odom')
        self.add_omni_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  odom_x_name='odom_x',
                                  odom_y_name='odom_y',
                                  odom_yaw_name='odom_t',
                                  odometry_topic='/hsrb4s/base_footprint')
        self.add_follow_joint_trajectory_server(namespace='/hsrb4s/arm_trajectory_controller/follow_joint_trajectory',
                                                state_topic='/hsrb4s/arm_trajectory_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/hsrb4s/head_trajectory_controller/follow_joint_trajectory',
                                                state_topic='/hsrb4s/head_trajectory_controller/state',
                                                fill_velocity_values=True)
        self.add_follow_joint_trajectory_server(namespace='/omni_pose_follower/follow_joint_trajectory',
                                                state_topic='/omni_pose_follower/state',
                                                fill_velocity_values=True)


class HSR_StandAlone(HSR_Base):
    def __init__(self):
        self.add_robot_from_parameter_server()
        super().__init__()
        self._general_config.control_mode = ControlModes.stand_alone
        self.publish_all_tf()
        self.root_link_name = 'map'
        self.add_fixed_joint(parent_link='map', child_link='odom')
        self.add_omni_drive_joint(parent_link_name='odom',
                                  child_link_name='base_footprint',
                                  odom_x_name='odom_x',
                                  odom_y_name='odom_y',
                                  odom_yaw_name='odom_t',
                                  name='brumbrum')
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
