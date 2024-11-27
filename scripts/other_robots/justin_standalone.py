#!/usr/bin/env python3
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.other_robots.justin import JustinStandaloneInterface, JustinCollisionAvoidanceConfig, WorldWithJustinConfig
from giskardpy.configs.iai_robots.tiago import TiagoCollisionAvoidanceConfig, TiagoStandaloneInterface
from giskardpy.configs.world_config import WorldWithDiffDriveRobot, WorldWithOmniDriveRobot

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithJustinConfig(),
                      # collision_avoidance_config=TalosCollisionAvoidanceConfig(),
                      robot_interface_config=JustinStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(publish_tf=True, publish_js=True, debug_mode=True,
                                                              simulation_max_hz=20),
                      qp_controller_config=QPControllerConfig(qp_solver=SupportedQPSolver.qpalm))
    giskard.live()