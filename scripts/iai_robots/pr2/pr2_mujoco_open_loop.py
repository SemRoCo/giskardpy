#!/usr/bin/env python
import rospy

from giskardpy.configs.behavior_tree_config import OpenLoopConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.pr2 import PR2CollisionAvoidance
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.configs.world_config import WorldWithOmniDriveRobot


class PR2MujocoOpenLoop(PR2WorldSetup):
    def configure_execution(self):
        self.execution.set_control_mode(ControlModes.open_loop)

    def configure_behavior_tree(self):
        self.behavior_tree.add_visualization_marker_publisher(add_to_sync=True, add_to_planning=True,
                                                              add_to_control_loop=False)

    def configure_robot_interface(self):
        self.robot_interface.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                                           tf_parent_frame=self.map_name,
                                                           tf_child_frame=self.odom_link_name)
        self.robot_interface.sync_joint_state_topic('/joint_states')
        self.robot_interface.sync_odometry_topic('/pr2/base_footprint', self.drive_joint_name)
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/pr2/whole_body_controller/follow_joint_trajectory',
            state_topic='/pr2/whole_body_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/pr2/l_gripper_l_finger_controller/follow_joint_trajectory',
            state_topic='/pr2/l_gripper_l_finger_controller/state')
        self.robot_interface.add_follow_joint_trajectory_server(
            namespace='/pr2/r_gripper_l_finger_controller/follow_joint_trajectory',
            state_topic='/pr2/r_gripper_l_finger_controller/state')
        self.robot_interface.add_base_cmd_velocity(cmd_vel_topic='/pr2/cmd_vel',
                                                   track_only_velocity=True,
                                                   joint_name=self.drive_joint_name)


if __name__ == '__main__':
    rospy.init_node('giskard')
    drive_joint_name = 'brumbrum'
    giskard = Giskard(world_config=WorldWithOmniDriveRobot(drive_joint_name=drive_joint_name),
                      collision_avoidance_config=PR2CollisionAvoidance(drive_joint_name=drive_joint_name),
                      robot_interface_config=PR2StandaloneInterface(drive_joint_name=drive_joint_name),
                      behavior_tree_config=OpenLoopConfig(),
                      qp_controller_config=QPControllerConfig())
    giskard.live()
