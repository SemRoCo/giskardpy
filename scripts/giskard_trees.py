#!/usr/bin/env python
import rospy

from giskardpy.tree.tree_manager import TreeManager
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    tree = TreeManager.from_param_server()
    tree.live()
