#!/usr/bin/env python
import rospy

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.behavior_tree_config import OpenLoopBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.pr2 import PR2CollisionAvoidance, PR2JointTrajServerMujocoInterface, \
    WorldWithPR2Config

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithPR2Config(),
                      collision_avoidance_config=PR2CollisionAvoidance(),
                      robot_interface_config=PR2JointTrajServerMujocoInterface(),
                      behavior_tree_config=OpenLoopBTConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
