#!/usr/bin/env python
import rospy

from giskardpy.configs.default_config import GiskardConfig
from giskardpy.configs.drives import OmniDriveCmdVelInterface
from giskardpy.configs.follow_joint_trajectory import FollowJointTrajectoryInterface
from giskardpy.configs.pr2 import PR2
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    giskard = PR2()
    giskard.plugin_config['SyncTfFrames'] = {
        'frames': [['map', 'odom_combined']]
    }
    giskard.follow_joint_trajectory_interfaces = [
        FollowJointTrajectoryInterface(namespace='/pr2/whole_body_controller/follow_joint_trajectory',
                                       state_topic='/pr2/whole_body_controller/state')
    ]
    giskard.drive_interface = OmniDriveCmdVelInterface(cmd_vel_topic='/pr2/cmd_vel',
                                                       parent_link_name='odom_combined',
                                                       child_link_name='base_footprint')
    giskard.live()
