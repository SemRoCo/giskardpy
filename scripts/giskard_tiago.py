#!/usr/bin/env python
import rospy

from giskardpy.configs.tiago import Tiago
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    giskard = Tiago()
    giskard.live()
