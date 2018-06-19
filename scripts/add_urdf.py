#!/usr/bin/env python
import rospy
import sys
from giskardpy.python_interface import GiskardWrapper

if __name__ == '__main__':
    rospy.init_node('add_urdf')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        path = rospy.get_param('~path', None)
        param_name = rospy.get_param('~param', None)
        if path is None:
            if param_name is None:
                rospy.logwarn('neither _param nor _path specified')
                sys.exit()
            else:
                urdf = rospy.get_param(param_name)
        else:
            with open(path, 'r') as f:
                urdf = f.read()
        result = giskard.add_urdf(name=name,
                                  urdf=urdf,
                                  js_topic=rospy.get_param('~js', None),
                                  map_frame=rospy.get_param('~frame_frame', 'map'),
                                  root_frame=rospy.get_param('~root_frame', None))
        if result.error_codes == result.SUCCESS:
            rospy.loginfo('urdf \'{}\' added'.format(name))
        else:
            rospy.logwarn('failed to add urdf \'{}\''.format(name))
    except KeyError:
        rospy.loginfo('Example call: rosrun giskardpy add_urdf.py _name:=kitchen _param:=kitchen_description _js:=kitchen_joint_states _root_frame:=world _frame_id:=map')
