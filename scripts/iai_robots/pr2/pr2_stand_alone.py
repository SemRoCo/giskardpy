#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.pr2 import PR2World, PR2CollisionAvoidance, PR2StandaloneInterface
from giskardpy.configs.qp_controller_config import QPControllerConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    drive_joint_name = 'brumbrum'
    giskard = Giskard(world_config=PR2World(drive_joint_name),
                      collision_avoidance_config=PR2CollisionAvoidance(drive_joint_name),
                      robot_interface_config=PR2StandaloneInterface(drive_joint_name),
                      behavior_tree_config=StandAloneConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
