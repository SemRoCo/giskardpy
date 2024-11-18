#!/usr/bin/env python3
import rospy

from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.ur5_dualarm import UR5DualArmWorldConfig, UR5DualArmCollisionAvoidanceConfig, \
    UR5DualArmJointTrajServerInterface, UR5DualArmVelocityInterface
from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.behavior_tree_config import ClosedLoopBTConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=UR5DualArmWorldConfig(),
                      # collision_avoidance_config=UR5DualArmCollisionAvoidanceConfig(),
                      robot_interface_config=UR5DualArmVelocityInterface(),
                      behavior_tree_config=ClosedLoopBTConfig())
    giskard.live()