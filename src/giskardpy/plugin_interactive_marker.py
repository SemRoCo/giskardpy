import numpy as np
import rospy
from actionlib.simple_action_client import SimpleActionClient
from geometry_msgs.msg._PoseStamped import PoseStamped
from geometry_msgs.msg._Quaternion import Quaternion
from giskard_msgs.msg._Controller import Controller
from giskard_msgs.msg._ControllerListAction import ControllerListAction
from giskard_msgs.msg._ControllerListGoal import ControllerListGoal
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg._InteractiveMarker import InteractiveMarker
from visualization_msgs.msg._InteractiveMarkerControl import InteractiveMarkerControl
from visualization_msgs.msg._InteractiveMarkerFeedback import InteractiveMarkerFeedback
from visualization_msgs.msg._Marker import Marker

from giskardpy.plugin import Plugin
from giskardpy.tfwrapper import TfWrapper


class InteractiveMarkerPlugin(Plugin):
    def __init__(self, root_link, tip_links, suffix=''):
        self.root_link = root_link
        self.tip_links = tip_links
        self.suffix = suffix
        # tf
        self.tf = TfWrapper()
        self.started = False


    def start(self, god_map):
        if not self.started:
            # giskard goal client
            self.client = SimpleActionClient('qp_controller/command', ControllerListAction)
            self.client.wait_for_server()

            # marker server
            self.server = InteractiveMarkerServer("eef_control{}".format(self.suffix))
            self.menu_handler = MenuHandler()

            for tip_link in self.tip_links:
                int_marker = self.make6DofMarker(InteractiveMarkerControl.MOVE_ROTATE_3D, self.root_link, tip_link)
                self.server.insert(int_marker, self.process_feedback(self.server, self.client, self.root_link, tip_link))
                self.menu_handler.apply(self.server, int_marker.name)

            self.server.applyChanges()
            self.started = True

    def makeBox(self, msg):
        marker = Marker()

        marker.type = Marker.SPHERE
        marker.scale.x = msg.scale * 0.2
        marker.scale.y = msg.scale * 0.2
        marker.scale.z = msg.scale * 0.2
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.5

        return marker

    def makeBoxControl(self, msg):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeBox(msg))
        msg.controls.append(control)
        return control

    def make6DofMarker(self, interaction_mode, root_link, tip_link):
        def normed_q(x,y,z,w):
            return np.array([x,y,z,w])/np.linalg.norm([x,y,z,w])

        int_marker = InteractiveMarker()

        p = PoseStamped()
        p.header.frame_id = tip_link
        p.pose.orientation.w = 1

        int_marker.header.frame_id = tip_link
        int_marker.pose.orientation.w = 1
        int_marker.pose = self.tf.transform_pose(root_link, p).pose
        int_marker.header.frame_id = root_link
        int_marker.scale = .2

        int_marker.name = "eef_{}_to_{}".format(root_link, tip_link)

        # insert a box
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D: "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D: "MOVE_ROTATE_3D"}
            int_marker.name += "_" + control_modes_dict[interaction_mode]

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0,1,0,1))
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0,1,0,1))
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0,0,1,1))
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(*normed_q(0,0,1,1))
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        return int_marker

    class process_feedback(object):
        def __init__(self, i_server, client, root_link, tip_link):
            self.i_server = i_server
            self.client = client
            self.tip_link = tip_link
            self.root_link = root_link

        def __call__(self, feedback):
            if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                print('interactive goal update')
                goal = ControllerListGoal()
                goal.type = ControllerListGoal.STANDARD_CONTROLLER
                # translation
                controller = Controller()
                controller.type = Controller.TRANSLATION_3D
                controller.tip_link = self.tip_link
                controller.root_link = self.root_link

                controller.goal_pose.header = feedback.header
                controller.goal_pose.pose = feedback.pose

                controller.p_gain = 3
                controller.enable_error_threshold = True
                controller.threshold_value = 0.05
                controller.weight = 1.0
                goal.controllers.append(controller)

                # rotation
                controller = Controller()
                controller.type = Controller.ROTATION_3D
                controller.tip_link = self.tip_link
                controller.root_link = self.root_link

                controller.goal_pose.header = feedback.header
                controller.goal_pose.pose = feedback.pose

                controller.p_gain = 3
                controller.enable_error_threshold = True
                controller.threshold_value = 0.2
                controller.weight = 1.0
                goal.controllers.append(controller)

                self.client.send_goal(goal)
            self.i_server.applyChanges()

    def get_replacement_parallel_universe(self):
        return self
