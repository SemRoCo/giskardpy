#!/usr/bin/env python
import rospy
from giskardpy.urdf_object import URDFObject
from geometry_msgs.msg import Pose
from giskardpy import logging


if not rospy.has_param('/robot_description'):
    logging.loginfo('waiting for rosparam /robot_description')
    rospy.sleep(0.1)


hsrb_urdf = rospy.get_param('/robot_description')
#hsrb_urdf = hsrb_urdf.replace('hsrb_description', 'hsr_description')
#hsrb_urdf = hsrb_urdf.replace('hsrb_parts_description', 'hsr_description')
hsrb = URDFObject(hsrb_urdf)
hsrb = URDFObject.from_urdf_file('/home/kevin/Documents/catkin_ws/src/hsr_description/robots/hsrb4s.urdf')
hsrb_base = URDFObject.from_urdf_file('/home/kevin/Documents/catkin_ws/src/giskardpy/test/urdfs/hsr_base.urdf')

hsrb_base.attach_urdf_object(hsrb, 'odom_t_link', Pose())
hsrb_with_base_urdf = hsrb_base.get_urdf_str()

rospy.set_param('/giskard/robot_description', hsrb_with_base_urdf)
#rospy.set_param('/robot_description', hsrb_with_base_urdf)