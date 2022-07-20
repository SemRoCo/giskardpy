#!/usr/bin/env python
import rospy

from giskardpy.configs.default_config import Giskard
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    giskard = Giskard()
    giskard.robot_interface_config.joint_state_topic = 'joint_states'
    giskard.add_sync_tf_frame('map', 'odom')
    # giskard.set_odometry_topic('/mobile_base_controller/odom')
    giskard.set_odometry_topic('/tiago/base_footprint')
    giskard.add_follow_joint_trajectory_server(namespace='/arm_left_controller/follow_joint_trajectory',
                                               state_topic='/arm_left_controller/state')
    giskard.add_follow_joint_trajectory_server(namespace='/arm_right_controller/follow_joint_trajectory',
                                               state_topic='/arm_right_controller/state')
    # giskard.add_follow_joint_trajectory_server(namespace='/head_controller/follow_joint_trajectory',
    # state_topic='/head_controller/state')
    giskard.add_follow_joint_trajectory_server(namespace='/torso_controller/follow_joint_trajectory',
                                               state_topic='/torso_controller/state')
    # giskard.add_diff_drive_interface(cmd_vel_topic='/mobile_base_controller/cmd_vel',
    giskard.add_diff_drive_interface(cmd_vel_topic='/tiago/cmd_vel',
                                     parent_link_name='odom',
                                     child_link_name='base_footprint')
    giskard.live()
