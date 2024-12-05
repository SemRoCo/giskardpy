#!/usr/bin/env python
import rospy

from giskardpy_ros.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.hsr import WorldWithHSRConfig, HSRCollisionAvoidanceConfig, \
    HSRStandaloneInterface

if __name__ == '__main__':
    rospy.init_node('giskard')
    debug_mode = rospy.get_param('~debug_mode', False)
    giskard = Giskard(world_config=WorldWithHSRConfig(),
                      collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                      robot_interface_config=HSRStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(publish_free_variables=True, publish_tf=True,
                                                              debug_mode=debug_mode))
    giskard.live()
