#!/usr/bin/env python
import rospy

from giskardpy.configs.donbot import Donbot
from giskardpy.configs.pr2 import PR2
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    giskard = Donbot()
    giskard.live()
