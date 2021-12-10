#!/usr/bin/env python

import rospy
from giskardpy.utils import logging
from giskardpy.utils.config_loader import load_robot_yaml


if __name__ == '__main__':
    rospy.init_node('giskard_robot_config_uploader')

    if not rospy.is_shutdown():
        try:
            old_params = rospy.get_param('giskard')
            config_file_name = rospy.get_param(rospy.get_name() + '/' + ''.join('config'))
            robot_description_dict = load_robot_yaml(config_file_name)
            old_params.update(robot_description_dict)
            rospy.set_param('giskard', robot_description_dict)
        except KeyboardInterrupt:
            exit()
    logging.loginfo('uploaded config of robot')
