#!/usr/bin/env python
import rospy

from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.hsr import HSRCollisionAvoidanceConfig, WorldWithHSRConfigMujoco, HSRMujocoVelocityInterface
from giskardpy.configs.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.configs.qp_controller_config import QPControllerConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithHSRConfigMujoco(),
                      collision_avoidance_config=HSRCollisionAvoidanceConfig(),
                      robot_interface_config=HSRMujocoVelocityInterface(),
                      behavior_tree_config=ClosedLoopBTConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()