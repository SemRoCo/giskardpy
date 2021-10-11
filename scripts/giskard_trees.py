#!/usr/bin/env python
import rospy
from giskardpy.garden import grow_tree
from giskardpy.utils import check_dependencies
from giskardpy import logging, identifier

# TODO add pytest to package xml
# TODO add transform3d to package xml


if __name__ == u'__main__':
    rospy.init_node(u'giskard')
    check_dependencies()
    tree = grow_tree()
    tree_tick_rate = 1. / rospy.get_param(rospy.get_name() +u'/' +u'/'.join(identifier.tree_tick_rate[1:]))

    sleeper = rospy.Rate(tree_tick_rate)
    logging.loginfo(u'giskard is ready')
    while not rospy.is_shutdown():
        try:
            tree.tick()
            sleeper.sleep()
        except KeyboardInterrupt:
            break
    logging.loginfo(u'\n')