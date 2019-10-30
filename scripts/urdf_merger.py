#!/usr/bin/env python
import rospy
from giskardpy.urdf_object import URDFObject
from geometry_msgs.msg import Pose, Point, Quaternion
from giskardpy import logging


while not rospy.has_param('/urdf_merger/urdf_sources'):
    logging.loginfo('waiting for rosparam /urdf_merger/urdf_sources')
    rospy.sleep(1.0)

while not rospy.has_param('/urdf_merger/robot_name'):
    logging.loginfo('waiting for rosparam /urdf_merger/robot_name')
    rospy.sleep(1.0)

urdf_sources = rospy.get_param('/urdf_merger/urdf_sources')
robot_name = rospy.get_param('/urdf_merger/robot_name')

merged_object = None

if rospy.has_param(urdf_sources[0]['source']):
    urdf_string = rospy.get_param(urdf_sources[0]['source'])
    merged_object = URDFObject(urdf_string)
else:
    merged_object = URDFObject.from_urdf_file(urdf_sources[0]['source'])


for source in urdf_sources[1:]:
    if rospy.has_param(source['source']):
        urdf_string = rospy.get_param(source['source'])
        urdf_object = URDFObject(urdf_string)
    else:
        urdf_object = URDFObject.from_urdf_file(source['source'])


    positon = Point()
    rotation = Quaternion()
    if 'position' in source.keys():
        position =  Point(source['position'][0], source['position'][1], source['position'][2])
    if 'rotation' in source.keys():
        rotation = Quaternion(source['rotation'][0], source['rotation'][1], source['rotation'][2], source['rotation'][3])

    merged_object.attach_urdf_object(urdf_object, source['link'], Pose(position, rotation))


merged_object.set_name(robot_name)
merged_urdf = merged_object.get_urdf_str()

rospy.set_param('/giskard/robot_description', merged_urdf)