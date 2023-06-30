from copy import deepcopy

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Pose, Vector3Stamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rospy import Timer
from rospy.timer import TimerEvent
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_multiply, quaternion_about_axis
from visualization_msgs.msg import InteractiveMarkerControl, MarkerArray, InteractiveMarkerFeedback, InteractiveMarker, \
    Marker

from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils import logging
from giskardpy.utils.math import qv_mult
import giskardpy.utils.tfwrapper as tf

MARKER_SCALE = 0.3


def line_circle_intersection(line_point, line_direction, circle_center, circle_radius):
    # Compute the direction vector of the line
    line_direction = line_direction / np.linalg.norm(line_direction)

    # Compute the coefficients for the quadratic equation
    a = np.dot(line_direction, line_direction)
    b = 2 * np.dot(line_direction, line_point - circle_center)
    c = np.dot(circle_center, circle_center) + np.dot(line_point, line_point) - 2 * np.dot(circle_center,
                                                                                           line_point) - circle_radius ** 2

    # Compute the discriminant
    discriminant = b ** 2 - 4 * a * c

    # Check for intersection
    if discriminant < 0:
        # No intersection
        return np.inf
    elif discriminant == 0:
        # One intersection
        t = -b / (2 * a)
        intersection_point = line_point + t * line_direction
        if t < 0:
            return np.inf
        else:
            return t
    else:
        # Two intersections
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        intersection_point1 = line_point + t1 * line_direction
        intersection_point2 = line_point + t2 * line_direction
        t = min(t1, t2)
        if t < 0:
            return np.inf
        else:
            return t


class IMServer(object):
    """
    Spawns interactive Marker which send cart goals to action server.
    Does not interact with god map.
    """

    def __init__(self, topic_name: str):
        """
        :param root_tips: list containing root->tip tuple for each interactive marker.
        :type root_tips: list
        :param suffix: the marker will be called 'eef_control{}'.format(suffix)
        :type suffix: str
        """
        self.topic_name = topic_name
        self.root = 'map'
        self.tip = 'head_mount_kinect_rgb_optical_frame'
        self.point = PointStamped()
        self.point.header.frame_id = 'base_footprint'
        self.point.point.x = 1
        self.point.point.z = 1
        self.point = tf.transform_msg(self.root, self.point)
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = self.tip
        pointing_axis.vector.z = 1
        self.markers = {}

        # marker server
        self.server = InteractiveMarkerServer(self.topic_name)
        self.menu_handler = MenuHandler()

        int_marker = self.make_6dof_marker(InteractiveMarkerControl.MOVE_ROTATE_3D, self.root, self.tip)
        self.processor = self.process_feedback(self.server,
                                               int_marker.name,
                                               self.topic_name,
                                               self)
        self.server.insert(int_marker, self.processor
                           )
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()

    def make_sphere(self, msg):
        """
        :param msg:
        :return:
        :rtype: Marker
        """
        marker = Marker()

        marker.type = Marker.SPHERE
        marker.scale.x = msg.scale * MARKER_SCALE * 2
        marker.scale.y = msg.scale * MARKER_SCALE * 2
        marker.scale.z = msg.scale * MARKER_SCALE * 2
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.5

        return marker

    def make_sphere_control(self, msg):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.make_sphere(msg))
        msg.controls.append(control)
        return control

    def make_6dof_marker(self, interaction_mode, root_link, tip_link):
        def normed_q(x, y, z, w):
            return np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])

        int_marker = InteractiveMarker()

        int_marker.header.frame_id = self.root
        int_marker.pose.position = self.point.point
        int_marker.pose.orientation.w = 1
        int_marker.scale = MARKER_SCALE

        int_marker.name = 'eef_{}_to_{}'.format(root_link, tip_link)

        # insert a box
        self.make_sphere_control(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: 'MOVE_3D',
                InteractiveMarkerControl.ROTATE_3D: 'ROTATE_3D',
                InteractiveMarkerControl.MOVE_ROTATE_3D: 'MOVE_ROTATE_3D'}
            int_marker.name += '_' + control_modes_dict[interaction_mode]

        # control = InteractiveMarkerControl()
        # control.orientation = Quaternion(0, 0, 0, 1)
        # control.name = 'rotate_x'
        # control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = 'move_x'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # control = InteractiveMarkerControl()
        # control.orientation = Quaternion(*normed_q(0, 1, 0, 1))
        # control.name = 'rotate_z'
        # control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 1, 0, 1))
        control.name = 'move_z'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # control = InteractiveMarkerControl()
        # control.orientation = Quaternion(*normed_q(0, 0, 1, 1))
        # control.name = 'rotate_y'
        # control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        # int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 0, 1, 1))
        control.name = 'move_y'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        self.markers[int_marker.name] = int_marker
        return int_marker

    class process_feedback(object):
        def __init__(self, i_server, marker_name, topic_name, parent):
            """
            :param i_server:
            :type i_server: InteractiveMarkerServer
            :param marker_name:
            :type marker_name: str
            :param client:
            :type client: SimpleActionClient
            :param root_link:
            :type root_link: str
            :param tip_link:
            :type tip_link: str
            :param enable_self_collision:
            :type enable_self_collision: bool
            """
            self.parent = parent
            self.i_server = i_server
            self.marker_name = marker_name
            self.topic_name = topic_name
            self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
            # self.laser_pub = rospy.Publisher('/hsrb/base_scan', LaserScan, queue_size=10)
            self.laser_pub = rospy.Publisher(self.topic_name, LaserScan, queue_size=10)
            self.target = PointStamped()
            self.target.header.frame_id = 'map'
            self.target.point.x = 1
            self.target.point.z = 1
            self.stop_after = 5
            self.radius = 0.2
            self.transformed_target = None
            self.frame_id = 'base_laser_link'
            self.timer = Timer(rospy.Duration(0.1), self.timer_cb)

        def timer_cb(self, timer_event: TimerEvent):
            # if self.transformed_target is None:
            #     return
            self.transformed_target = tf.transform_point(self.frame_id, self.target)
            # print('asdf')
            scan = LaserScan()
            scan.header.frame_id = self.frame_id
            scan.header.stamp = timer_event.current_real
            scan.angle_min = -2.0987584590911865
            scan.angle_max = 2.0987584590911865
            scan.angle_increment = 0.004363323096185923
            scan.time_increment = 1.736111516947858e-05
            scan.scan_time = 0.02500000037252903
            scan.range_min = 0.019999999552965164
            scan.range_max = 30.0
            angles = np.arange(scan.angle_min,
                               scan.angle_max,
                               scan.angle_increment)
            for angle in angles:
                x = np.cos(angle)
                y = np.sin(angle)
                distance = line_circle_intersection(line_point=np.array([0, 0]),
                                                    line_direction=np.array([x, y]),
                                                    circle_center=np.array([self.transformed_target.point.x,
                                                                            self.transformed_target.point.y]),
                                                    circle_radius=self.radius)
                scan.ranges.append(max(min(distance, scan.range_max), scan.range_min))

            self.laser_pub.publish(scan)

        def __call__(self, feedback: InteractiveMarkerFeedback):
            if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                logging.loginfo('got interactive goal update')
                p = PointStamped()
                p.header.frame_id = feedback.header.frame_id
                p.header.stamp = rospy.get_rostime()
                p.point = feedback.pose.position
                p.point.z = 0.5
                self.target = deepcopy(p)
                self.target.header.stamp = rospy.Time()
                self.parent.point = p
                # self.goal_pub.publish(p)
                # self.pub_goal_marker(feedback.header, feedback.pose)
                self.i_server.setPose(self.marker_name, feedback.pose)
            self.i_server.applyChanges()

        def pub_goal_marker(self, header, pose: Pose):
            """
            :param header:
            :type header: std_msgs.msg._Header.Header
            :param pose:
            :type pose: Pose
            """
            ma = MarkerArray()
            m = Marker()
            m.action = Marker.ADD
            m.type = Marker.CYLINDER
            m.header = header
            old_q = [pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z,
                     pose.orientation.w]
            # x
            m.pose = deepcopy(pose)
            m.scale.x = 0.05 * MARKER_SCALE
            m.scale.y = 0.05 * MARKER_SCALE
            m.scale.z = MARKER_SCALE
            muh = qv_mult(old_q, [m.scale.z / 2, 0, 0])
            m.pose.position.x += muh[0]
            m.pose.position.y += muh[1]
            m.pose.position.z += muh[2]
            m.pose.orientation = Quaternion(*quaternion_multiply(old_q, quaternion_about_axis(np.pi / 2, [0, 1, 0])))
            m.color.r = 1
            m.color.g = 0
            m.color.b = 0
            m.color.a = 1
            m.ns = 'interactive_marker_{}_{}'.format(self.root_link, self.tip_link)
            m.id = 0
            ma.markers.append(m)
            # y
            m = deepcopy(m)
            m.pose = deepcopy(pose)
            muh = qv_mult(old_q, [0, m.scale.z / 2, 0])
            m.pose.position.x += muh[0]
            m.pose.position.y += muh[1]
            m.pose.position.z += muh[2]
            m.pose.orientation = Quaternion(*quaternion_multiply(old_q, quaternion_about_axis(-np.pi / 2, [1, 0, 0])))
            m.color.r = 0
            m.color.g = 1
            m.color.b = 0
            m.color.a = 1
            m.ns = 'interactive_marker_{}_{}'.format(self.root_link, self.tip_link)
            m.id = 1
            ma.markers.append(m)
            # z
            m = deepcopy(m)
            m.pose = deepcopy(pose)
            muh = qv_mult(old_q, [0, 0, m.scale.z / 2])
            m.pose.position.x += muh[0]
            m.pose.position.y += muh[1]
            m.pose.position.z += muh[2]
            m.color.r = 0
            m.color.g = 0
            m.color.b = 1
            m.color.a = 1
            m.ns = 'interactive_marker_{}_{}'.format(self.root_link, self.tip_link)
            m.id = 2
            ma.markers.append(m)
            self.marker_pub.publish(ma)


rospy.init_node('laser_scanner')
im1 = IMServer('scan')
im2 = IMServer('hsrb/base_scan')
rospy.spin()
