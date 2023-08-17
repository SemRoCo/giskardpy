#!/usr/bin/env python
import rospy

from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.tracy import TracyWorldConfig, TracyCollisionAvoidanceConfig, \
    TracyJointTrajServerMujocoInterface

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=TracyWorldConfig(),
                      collision_avoidance_config=TracyCollisionAvoidanceConfig(),
                      robot_interface_config=TracyJointTrajServerMujocoInterface())
    giskard.live()
