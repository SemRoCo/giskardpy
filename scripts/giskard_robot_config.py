#!/usr/bin/env python

import rospy
from giskardpy import logging
from giskardpy.config_loader import load_robot_yaml


if __name__ == u'__main__':
    rospy.init_node(u'giskard_robot_config_uploader')

    if not rospy.is_shutdown():
        try:
            old_params = rospy.get_param('giskard')
            config_file_name = rospy.get_param(rospy.get_name() + u'/' + ''.join(u'config'))
            robot_description_dict = load_robot_yaml(config_file_name)
            old_params.update(robot_description_dict)
            rospy.set_param(u'giskard', robot_description_dict)
        except KeyboardInterrupt:
            exit()
    logging.loginfo(u'uploaded config of robot')
