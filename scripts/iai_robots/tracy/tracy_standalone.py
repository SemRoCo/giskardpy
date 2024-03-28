#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.tracy import TracyWorldConfig, TracyCollisionAvoidanceConfig, \
    TracyStandAloneRobotInterfaceConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=TracyWorldConfig(),
                      collision_avoidance_config=TracyCollisionAvoidanceConfig(),
                      robot_interface_config=TracyStandAloneRobotInterfaceConfig(),
                      behavior_tree_config=StandAloneBTConfig(include_prefix=True))
    giskard.live()
