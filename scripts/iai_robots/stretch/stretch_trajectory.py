#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import OpenLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.stretch import StretchCollisionAvoidanceConfig, StretchTrajectoryInterface
from giskardpy.configs.iai_robots.tiago import TiagoCollisionAvoidanceConfig, TiagoStandaloneInterface
from giskardpy.configs.world_config import WorldWithDiffDriveRobot

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithDiffDriveRobot(),
                      collision_avoidance_config=StretchCollisionAvoidanceConfig(),
                      robot_interface_config=StretchTrajectoryInterface(),
                      behavior_tree_config=OpenLoopBTConfig(simulation_max_hz=20, debug_mode=True),
                      )
    giskard.live()
