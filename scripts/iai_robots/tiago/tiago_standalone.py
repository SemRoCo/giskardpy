#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.tiago import TiagoCollisionAvoidanceConfig, TiagoStandaloneInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.configs.world_config import WorldWithDiffDriveRobot

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithDiffDriveRobot(),
                      collision_avoidance_config=TiagoCollisionAvoidanceConfig(),
                      robot_interface_config=TiagoStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
