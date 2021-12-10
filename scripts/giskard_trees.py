#!/usr/bin/env python
import rospy
from giskardpy.garden import grow_tree
from giskardpy import identifier

# TODO add pytest to package xml
# TODO add transform3d to package xml
from giskardpy.utils import logging
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    tree = grow_tree()
    tree_tick_rate = 1. / rospy.get_param(rospy.get_name() +'/' +'/'.join(identifier.tree_tick_rate[1:]))

    sleeper = rospy.Rate(tree_tick_rate)
    logging.loginfo('giskard is ready')
    while not rospy.is_shutdown():
        try:
            tree.tick()
            sleeper.sleep()
        except KeyboardInterrupt:
            break
    logging.loginfo('\n')