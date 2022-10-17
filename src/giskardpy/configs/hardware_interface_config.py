from typing import Optional, List

from giskardpy.my_types import my_string


class HardwareConfig:
    def __init__(self):
        self.send_trajectory_to_cmd_vel: List[dict] = []
        self.follow_joint_trajectory_interfaces: List[dict] = []
        self.joint_state_topics: List[str] = []
        self.odometry_topics: List[str] = []

    def add_follow_joint_trajectory_server(self, namespace, state_topic, fill_velocity_values):
        self.follow_joint_trajectory_interfaces.append({'namespace': namespace,
                                                        'state_topic': state_topic,
                                                        'fill_velocity_values': fill_velocity_values})

    def add_base_cmd_velocity(self,
                              cmd_vel_topic: str,
                              track_only_velocity: bool = False,
                              joint_name: Optional[my_string] = None):
        self.send_trajectory_to_cmd_vel.append({'cmd_vel_topic': cmd_vel_topic,
                                                'track_only_velocity': track_only_velocity,
                                                'joint_name': joint_name})
