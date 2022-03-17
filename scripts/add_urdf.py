#!/usr/bin/env python
import rospy
import sys

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler

from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils.tfwrapper import lookup_pose
from giskardpy.utils import logging

if __name__ == '__main__':
    rospy.init_node('add_urdf')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        path = rospy.get_param('~path', None)
        param_name = rospy.get_param('~param', None)
        position = rospy.get_param('~position', None)
        orientation = rospy.get_param('~rpy', None)
        root_frame = rospy.get_param('~root_frame', None)
        map_frame = rospy.get_param('~frame_id', 'map')
        if root_frame is not None:
            pose = lookup_pose(map_frame, root_frame)
        else:
            pose = PoseStamped()
            pose.header.frame_id = map_frame
            if position is not None:
                pose.pose.position = Point(*position)
            if orientation is not None:
                pose.pose.orientation = Quaternion(*quaternion_from_euler(*orientation))
            else:
                pose.pose.orientation.w = 1
        if path is None:
            if param_name is None:
                logging.logwarn('neither _param nor _path specified')
                sys.exit()
            else:
                urdf = rospy.get_param(param_name)
        else:
            with open(path, 'r') as f:
                urdf = f.read()
        result = giskard.add_urdf(name=name, urdf=urdf, pose=pose, js_topic=rospy.get_param('~js', None))
        if result.error_codes == result.SUCCESS:
            logging.loginfo('urdfs \'{}\' added'.format(name))
        else:
            logging.logwarn('failed to add urdfs \'{}\''.format(name))
    except KeyError:
        logging.loginfo('Example call: rosrun giskardpy add_urdf.py _name:=kitchen _param:=kitchen_description _js:=kitchen/joint_states _root_frame:=iai_kitchen/world _frame_id:=map')
