#!/usr/bin/env python
import numpy as np
from copy import deepcopy

import rospy
from actionlib.simple_action_client import SimpleActionClient
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg._Quaternion import Quaternion
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from tf.transformations import quaternion_multiply, quaternion_about_axis
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg._InteractiveMarker import InteractiveMarker
from visualization_msgs.msg._InteractiveMarkerControl import InteractiveMarkerControl
from visualization_msgs.msg._InteractiveMarkerFeedback import InteractiveMarkerFeedback
from visualization_msgs.msg._Marker import Marker

from giskardpy.python_interface.old_python_interface import OldGiskardWrapper
from giskardpy.utils import logging
from giskardpy.utils.math import qv_mult

MARKER_SCALE = 0.15



class IMServer(object):
    """
    Spawns interactive Marker which send cart goals to action server.
    Does not interact with god map.
    """
    def __init__(self, root_tips, suffix=''):
        """
        :param root_tips: list containing root->tip tuple for each interactive marker.
        :type root_tips: list
        :param suffix: the marker will be called 'eef_control{}'.format(suffix)
        :type suffix: str
        """
        self.enable_self_collision = rospy.get_param('~enable_self_collision', True)
        self.giskard = OldGiskardWrapper()
        self.robot_name = self.giskard.robot_name
        if len(root_tips) > 0:
            self.roots, self.tips = zip(*root_tips)
        else:
            self.roots = []
            self.tips = []
        self.suffix = suffix
        self.markers = {}

        # marker server
        self.server = InteractiveMarkerServer('eef_control{}'.format(self.suffix))
        self.menu_handler = MenuHandler()


        for root, tip in zip(self.roots, self.tips):
            root = root
            tip = tip
            int_marker = self.make_6dof_marker(InteractiveMarkerControl.MOVE_ROTATE_3D, root, tip)
            self.server.insert(int_marker,
                               self.process_feedback(self.server,
                                                     int_marker.name,
                                                     root,
                                                     tip,
                                                     self.giskard,
                                                     self.enable_self_collision))
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

        int_marker.header.frame_id = tip_link
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

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = 'rotate_x'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = 'move_x'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 1, 0, 1))
        control.name = 'rotate_z'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 1, 0, 1))
        control.name = 'move_z'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 0, 1, 1))
        control.name = 'rotate_y'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 0, 1, 1))
        control.name = 'move_y'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        self.markers[int_marker.name] = int_marker
        return int_marker

    class process_feedback(object):
        def __init__(self, i_server, marker_name, root_link, tip_link, giskard, enable_self_collision):
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
            :param giskard:
            :type giskard: OldGiskardWrapper
            :param enable_self_collision:
            :type enable_self_collision: bool
            """
            self.i_server = i_server
            self.marker_name = marker_name
            self.tip_link = tip_link
            self.root_link = root_link
            self.giskard = giskard
            self.enable_self_collision = enable_self_collision
            self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)


        def __call__(self, feedback):
            if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                logging.loginfo('got interactive goal update')

                p = PoseStamped()
                p.header.frame_id = feedback.header.frame_id
                p.pose = feedback.pose
                # self.giskard.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
                self.giskard.set_straight_cart_goal(root_link=self.root_link,
                                           tip_link=self.tip_link,
                                           goal_pose=p)

                if not self.enable_self_collision:
                    self.giskard.allow_self_collision()
                self.giskard.allow_all_collisions()
                self.giskard.execute(wait=False)
                # self.giskard.plan(wait=False)
                self.pub_goal_marker(feedback.header, feedback.pose)
                self.i_server.setPose(self.marker_name, Pose())
            self.i_server.applyChanges()

        def pub_goal_marker(self, header, pose):
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
            muh = qv_mult(old_q, [0,0,m.scale.z / 2])
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



if __name__ == '__main__':
    rospy.init_node('giskard_interactive_marker')
    root_tips = rospy.get_param('~interactive_marker_chains')
    im = IMServer(root_tips)
    while not rospy.is_shutdown():
        rospy.sleep(1)