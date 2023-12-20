#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.other_robots.armar import WorldWithArmarConfig, ArmarCollisionAvoidanceConfig, \
    ArmarStandaloneInterface

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithArmarConfig(),
                      collision_avoidance_config=ArmarCollisionAvoidanceConfig(),
                      robot_interface_config=ArmarStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(publish_js=True))
    giskard.live()
