#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import OpenLoopBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.tracy import TracyWorldConfig, TracyCollisionAvoidanceConfig, \
    TracyJointTrajServerMujocoInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=TracyWorldConfig(),
                      collision_avoidance_config=TracyCollisionAvoidanceConfig(),
                      robot_interface_config=TracyJointTrajServerMujocoInterface(),
                      behavior_tree_config=OpenLoopBTConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
