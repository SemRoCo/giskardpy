#!/usr/bin/env python
import rospy

from giskardpy.configs.pr2 import PR2_Mujoco, PR2_Real
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    giskard = PR2_Real()
    giskard.live()
