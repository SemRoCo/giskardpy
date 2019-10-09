#!/usr/bin/env python
import rospy
from giskardpy.urdf_object import URDFObject
from geometry_msgs.msg import Pose
from giskardpy import logging
import rospkg


if not rospy.has_param('/robot_description'):
    logging.loginfo('waiting for rosparam /robot_description')
    rospy.sleep(1.0)

rospack = rospkg.RosPack()

hsrb_urdf = rospy.get_param('/robot_description')
hsrb = URDFObject(hsrb_urdf)
#hsrb_urdf_path = rospack.get_path('hsr_description') + '/robots/hsrb4s.urdf'
#hsrb = URDFObject.from_urdf_file(hsrb_urdf_path)
hsr_base_urdf_path = rospack.get_path('giskardpy') + '/test/urdfs/hsr_base.urdf'
hsrb_base = URDFObject.from_urdf_file(hsr_base_urdf_path )

hsrb_base.attach_urdf_object(hsrb, 'odom_t_link', Pose())
hsrb_with_base_urdf = hsrb_base.get_urdf_str()

rospy.set_param('/giskard/robot_description', hsrb_with_base_urdf)