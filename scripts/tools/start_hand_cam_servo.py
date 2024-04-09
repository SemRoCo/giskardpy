from giskardpy.python_interface.python_interface import GiskardWrapper
import rospy
from geometry_msgs.msg import Vector3Stamped

rospy.init_node('handCamServoNode')
gis = GiskardWrapper()
optical_axis = Vector3Stamped()
optical_axis.header.frame_id = 'hand_palm_link'
optical_axis.vector.z = 1
gis.motion_goals.add_motion_goal(motion_goal_class='HandCamServoGoal',
                                 name='hnadcamservo',
                                 root_link='map',
                                 cam_link='hand_palm_link',
                                 optical_axis=optical_axis,
                                 transform_from_image_coordinates=True,
                                 distance_threshold=0.1)
"""
After starting this goal it reacts to three different topics:
1. /hand_cam/movement Vector3Stamped => contains 2D Vector (x,y) that describes desired camera movment in image coordinates
2. /hand_cam/angle Float64 => contains desired rotation angle for the camera (positive value should equal rightward 
rotation of the gripper)
3. /hand_cam/distance Float64 => contains the current distance to the goal and makes the gripper move along the optical 
axis until the distance is below the distance threshold
"""
gis.execute()
