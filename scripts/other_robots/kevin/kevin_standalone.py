#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig, VisualizationMode
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.other_robots.kevin import KevinStandaloneInterface, KevinCollisionAvoidanceConfig
from giskardpy.configs.world_config import WorldWithDiffDriveRobot

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithDiffDriveRobot(),
                      collision_avoidance_config=KevinCollisionAvoidanceConfig(),
                      robot_interface_config=KevinStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(publish_tf=True, publish_js=True, simulation_max_hz=20,
                                                              visualization_mode=VisualizationMode.VisualsFrameLocked))
    giskard.live()
