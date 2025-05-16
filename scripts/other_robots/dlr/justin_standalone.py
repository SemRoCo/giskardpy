#!/usr/bin/env python
import rospy

from giskardpy.middleware import set_middleware
from giskardpy.qp.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy_ros.configs.behavior_tree_config import OpenLoopBTConfig, StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.other_robots.justin import WorldWithJustinConfig, JustinStandaloneInterface, JustinCollisionAvoidanceConfig
from giskardpy_ros.ros1.interface import ROS1Wrapper
from giskardpy_ros.ros1.visualization_mode import VisualizationMode

if __name__ == '__main__':
    rospy.init_node('giskard')
    set_middleware(ROS1Wrapper())
    giskard = Giskard(world_config=WorldWithJustinConfig(),
                      collision_avoidance_config=JustinCollisionAvoidanceConfig(),
                      robot_interface_config=JustinStandaloneInterface(),
                      behavior_tree_config=StandAloneBTConfig(publish_tf=True, debug_mode=True,
                                                              visualization_mode=VisualizationMode.VisualsFrameLocked),
                      qp_controller_config=QPControllerConfig(mpc_dt=0.05))
    giskard.live()
