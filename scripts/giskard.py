#!/usr/bin/env python
import rospy

from giskardpy.configs.default_giskard import Giskard
from giskardpy.utils.dependency_checking import check_dependencies
from giskardpy.utils.utils import get_all_classes_in_package

if __name__ == '__main__':
    rospy.init_node('giskard')
    config = rospy.get_param('~config')
    check_dependencies()
    possible_classes = get_all_classes_in_package('giskardpy.configs', Giskard)
    giskard: Giskard = possible_classes[config]()
    giskard.live()
