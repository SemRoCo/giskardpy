import numpy as np
from collections import defaultdict
from copy import deepcopy

import rospy
from actionlib.simple_action_client import SimpleActionClient
from geometry_msgs.msg import Pose
from geometry_msgs.msg._Quaternion import Quaternion
from giskard_msgs.msg._Controller import Controller
from giskard_msgs.msg._MoveAction import MoveAction
from giskard_msgs.msg._MoveCmd import MoveCmd
from giskard_msgs.msg._MoveGoal import MoveGoal
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from tf.transformations import quaternion_multiply, quaternion_about_axis, quaternion_conjugate
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg._InteractiveMarker import InteractiveMarker
from visualization_msgs.msg._InteractiveMarkerControl import InteractiveMarkerControl
from visualization_msgs.msg._InteractiveMarkerFeedback import InteractiveMarkerFeedback
from visualization_msgs.msg._Marker import Marker

from giskardpy.plugin import Plugin
from giskardpy.utils import qv_mult

MARKER_SCALE = 0.15



class InteractiveMarkerPlugin(Plugin):
    def __init__(self, root_tips, suffix=u''):
        """
        :param roots:
        :type roots: list
        :param tips:
        :type tips: list
        :param suffix:
        :type suffix: str
        """
        if len(root_tips) > 0:
            self.roots, self.tips = zip(*root_tips)
        else:
            self.roots = []
            self.tips = []
        self.suffix = suffix
        self.markers = {}
        super(InteractiveMarkerPlugin, self).__init__()

    def start_once(self):
        # giskard goal client
        self.client = SimpleActionClient(u'qp_controller/command', MoveAction)
        self.client.wait_for_server()

        # marker server
        self.server = InteractiveMarkerServer(u'eef_control{}'.format(self.suffix))
        self.menu_handler = MenuHandler()

        all_goals = {}

        for root, tip in zip(self.roots, self.tips):
            int_marker = self.make6DofMarker(InteractiveMarkerControl.MOVE_ROTATE_3D, root, tip)
            self.server.insert(int_marker,
                               self.process_feedback(self.server, int_marker.name, self.client, root, tip, all_goals))
            self.menu_handler.apply(self.server, int_marker.name)

        self.server.applyChanges()

    def makeSphere(self, msg):
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

    def makeBoxControl(self, msg):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeSphere(msg))
        msg.controls.append(control)
        return control

    def make6DofMarker(self, interaction_mode, root_link, tip_link):
        def normed_q(x, y, z, w):
            return np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])

        int_marker = InteractiveMarker()

        int_marker.header.frame_id = tip_link
        int_marker.pose.orientation.w = 1
        int_marker.scale = MARKER_SCALE

        int_marker.name = u'eef_{}_to_{}'.format(root_link, tip_link)

        # insert a box
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: u'MOVE_3D',
                InteractiveMarkerControl.ROTATE_3D: u'ROTATE_3D',
                InteractiveMarkerControl.MOVE_ROTATE_3D: u'MOVE_ROTATE_3D'}
            int_marker.name += u'_' + control_modes_dict[interaction_mode]

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = u'rotate_x'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = u'move_x'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 1, 0, 1))
        control.name = u'rotate_z'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 1, 0, 1))
        control.name = u'move_z'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 0, 1, 1))
        control.name = u'rotate_y'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0, 0, 1, 1))
        control.name = u'move_y'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        self.markers[int_marker.name] = int_marker
        return int_marker

    class process_feedback(object):
        def __init__(self, i_server, marker_name, client, root_link, tip_link, all_goals):
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
            :param all_goals:
            :type all_goals: dict
            """
            self.i_server = i_server
            self.marker_name = marker_name
            self.client = client
            self.tip_link = tip_link
            self.root_link = root_link
            self.all_goals = all_goals
            self.reset_goal()
            self.marker_pub = rospy.Publisher(u'visualization_marker_array', MarkerArray, queue_size=10)

        def reset_goal(self):
            p = Pose()
            p.orientation.w = 1
            self.all_goals[self.root_link, self.tip_link] = []
            self.all_goals[self.root_link, self.tip_link].append(self.make_translation_controller(self.tip_link, p))
            self.all_goals[self.root_link, self.tip_link].append(self.make_rotation_controller(self.tip_link, p))

        def __call__(self, feedback):
            if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                self.all_goals = defaultdict(list)
                self.all_goals[self.root_link, self.tip_link] = []
                print(u'got interactive goal update')
                # translation
                controller = self.make_translation_controller(feedback.header.frame_id,
                                                              feedback.pose)
                self.all_goals[self.root_link, self.tip_link].append(controller)

                # rotation
                controller = self.make_rotation_controller(feedback.header.frame_id,
                                                           feedback.pose)
                self.all_goals[self.root_link, self.tip_link].append(controller)
                self.send_all_goals()
                self.pub_goal_marker(feedback.header, feedback.pose)
                self.reset_goal()
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
            m.ns = u'interactive_marker_{}_{}'.format(self.root_link, self.tip_link)
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
            m.ns = u'interactive_marker_{}_{}'.format(self.root_link, self.tip_link)
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
            m.ns = u'interactive_marker_{}_{}'.format(self.root_link, self.tip_link)
            m.id = 2
            ma.markers.append(m)
            self.marker_pub.publish(ma)

        def make_translation_controller(self, frame_id, pose):
            """
            :param frame_id:
            :type frame_id: str
            :param pose:
            :type pose: Pose
            :return:
            :rtype: giskard_msgs.msg._Controller.Controller
            """
            controller = self.make_controller(frame_id, pose)
            controller.type = Controller.TRANSLATION_3D
            controller.p_gain = 3
            controller.max_speed = 0.3
            controller.weight = 1.0
            return controller

        def make_rotation_controller(self, frame_id, pose):
            controller = self.make_controller(frame_id, pose)
            controller.type = Controller.ROTATION_3D
            controller.p_gain = 3
            controller.max_speed = 0.5
            controller.weight = 1.0
            return controller

        def make_controller(self, frame_id, pose):
            controller = Controller()
            controller.tip_link = self.tip_link
            controller.root_link = self.root_link
            controller.goal_pose.header.frame_id = frame_id
            controller.goal_pose.pose = pose
            return controller

        def send_all_goals(self):
            goal = MoveGoal()
            goal.type = MoveGoal.PLAN_AND_EXECUTE
            move_cmd = MoveCmd()
            # move_cmd.max_trajectory_length = 20
            for g in self.all_goals.values():
                move_cmd.controllers.extend(g)
            goal.cmd_seq.append(move_cmd)
            self.client.send_goal(goal)

        def stop(self):
            self.marker_pub.unregister()

    def copy(self):
        return self
