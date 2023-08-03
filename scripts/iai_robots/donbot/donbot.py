#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import OpenLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.donbot import WorldWithBoxyBaseConfig, DonbotCollisionAvoidanceConfig, \
    DonbotJointTrajInterfaceConfig
from giskardpy.configs.qp_controller_config import QPControllerConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithBoxyBaseConfig(),
                      collision_avoidance_config=DonbotCollisionAvoidanceConfig(),
                      robot_interface_config=DonbotJointTrajInterfaceConfig(),
                      behavior_tree_config=OpenLoopBTConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
