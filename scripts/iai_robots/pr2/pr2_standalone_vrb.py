#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.iai_robots.pr2 import PR2CollisionAvoidance, WorldWithPR2Config
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.configs.robot_interface_config import StandAloneRobotInterfaceConfig

if __name__ == '__main__':
    rospy.init_node('giskard')
    drive_joint_name = 'brumbrum'
    giskard = Giskard(world_config=WorldWithPR2Config(drive_joint_name=drive_joint_name),
                      collision_avoidance_config=PR2CollisionAvoidance(drive_joint_name=drive_joint_name),
                      robot_interface_config=StandAloneRobotInterfaceConfig(
                          [
                              'torso_lift_joint',
                              'head_pan_joint',
                              'head_tilt_joint',
                              'r_shoulder_pan_joint',
                              'r_shoulder_lift_joint',
                              'r_upper_arm_roll_joint',
                              'r_forearm_roll_joint',
                              'r_elbow_flex_joint',
                              'r_wrist_flex_joint',
                              'r_wrist_roll_joint',
                              'l_shoulder_pan_joint',
                              'l_shoulder_lift_joint',
                              'l_upper_arm_roll_joint',
                              'l_forearm_roll_joint',
                              'l_elbow_flex_joint',
                              'l_wrist_flex_joint',
                              'l_wrist_roll_joint',
                              drive_joint_name,
                          ]
                      ),
                      behavior_tree_config=StandAloneBTConfig(publish_tf=True, publish_js=False, debug_mode=True,
                                                              simulation_max_hz=20),
                      qp_controller_config=QPControllerConfig(qp_solver=SupportedQPSolver.qpSWIFT))
    giskard.live()
