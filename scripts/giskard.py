#!/usr/bin/env python3
import rospy

from giskardpy.configs.behavior_tree_config import OpenLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.world_config import WorldConfig, Derivatives, np
from giskardpy.configs.robot_interface_config import RobotInterfaceConfig
from giskardpy.configs.iai_robots.tiago import TiagoCollisionAvoidanceConfig
from giskardpy.configs.qp_controller_config import QPControllerConfig

class WorldWithTiagoDual(WorldConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.map_name = map_name
        self.odom = 'odom'
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.set_default_limits({Derivatives.velocity: 1,
                                              Derivatives.acceleration: np.inf,
                                              Derivatives.jerk: 30})
        self.add_empty_link(self.map_name)
        self.add_empty_link(self.odom)
        self.add_fixed_joint(self.map_name,
                             self.odom)
        self.add_robot_from_parameter_server()
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_diff_drive_joint(name=self.drive_joint_name,
                                               parent_link_name=self.odom,
                                               child_link_name=root_link_name,
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
                                               robot_group_name=self.robot_group_name)

class TiagoMultiverse(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_joint_state_topic('/joint_states')
        self.sync_odometry_topic(odometry_topic='/odom',
                                 joint_name=self.drive_joint_name)
        self.add_follow_joint_trajectory_server(namespace='/arm_left_controller/follow_joint_trajectory',
                                                state_topic='/arm_left_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/arm_right_controller/follow_joint_trajectory',
                                                state_topic='/arm_right_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/head_controller/follow_joint_trajectory',
                                                state_topic='/head_controller/state')
        self.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                                state_topic='/torso_controller/state')
        self.add_base_cmd_velocity(cmd_vel_topic='/cmd_vel',
                                   joint_name=self.drive_joint_name)

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithTiagoDual(),
                      collision_avoidance_config=TiagoCollisionAvoidanceConfig(),
                      robot_interface_config=TiagoMultiverse(),
                      behavior_tree_config=OpenLoopBTConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
