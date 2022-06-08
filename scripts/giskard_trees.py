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
    giskard.live()
