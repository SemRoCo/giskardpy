#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.stretch import StretchCollisionAvoidanceConfig, StretchStandaloneInterface
from giskardpy.configs.world_config import WorldWithDiffDriveRobot
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.model.ros_msg_visualization import VisualizationMode

if __name__ == '__main__':
    rospy.init_node('giskard')
    giskard = Giskard(world_config=WorldWithDiffDriveRobot(),
                      collision_avoidance_config=StretchCollisionAvoidanceConfig(),
                      robot_interface_config=StretchStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(publish_js=False, publish_tf=True, simulation_max_hz=20,
                                                              debug_mode=True,
                                                              visualization_mode=VisualizationMode.CollisionsDecomposedFrameLocked),
                      qp_controller_config=QPControllerConfig(qp_solver=SupportedQPSolver.qpSWIFT))
    giskard.live()
