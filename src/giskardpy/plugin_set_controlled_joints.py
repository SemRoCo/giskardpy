import rospy
from control_msgs.msg import JointTrajectoryControllerState

from giskardpy.plugin import PluginBase


class SetControlledJointsPlugin(PluginBase):
    """
    Gets controlled joints from whole body controller and writes it to the god map.
    """
    def __init__(self, controlled_joints_identifier):
        """
        :type controlled_joints_identifier: str
        """
        self.controlled_joints = []
        self.controlled_joints_identifier = controlled_joints_identifier
        super(SetControlledJointsPlugin, self).__init__()

    def initialize(self):
        # TODO make topic name a parameter
        msg = rospy.wait_for_message(u'/whole_body_controller/state', JointTrajectoryControllerState) # type: JointTrajectoryControllerState
        self.controlled_joints = msg.joint_names
        self.god_map.safe_set_data([self.controlled_joints_identifier], self.controlled_joints)

    def copy(self):
        c = self.__class__(self.controlled_joints_identifier)
        return c


class UploadRobotDescriptionPlugin(PluginBase):
    """
    Gets robot description from parameter server and writes it to the god map.
    """
    def __init__(self, robot_description_identifier, param_name=u'robot_description'):
        """
        :type robot_description_identifier: str
        :type param_name: str
        """
        self.urdf = ''
        self.param_name = param_name
        self.robot_description_identifier = robot_description_identifier
        super(UploadRobotDescriptionPlugin, self).__init__()

    def initialize(self):
        self.urdf = rospy.get_param(self.param_name)
        self.god_map.safe_set_data([self.robot_description_identifier], self.urdf)

    def copy(self):
        c = self.__class__(self.robot_description_identifier, self.param_name)
        return c


# class UploadUrdfPlugin(Plugin):
#     def __init__(self, robot_description_identifier, urdf):
#         self.urdf = urdf
#         self.robot_description_identifier = robot_description_identifier
#         super(UploadUrdfPlugin, self).__init__()
#
#     def start_always(self):
#         self.god_map.set_data([self.robot_description_identifier], self.urdf)
#
#     def copy(self):
#         c = self.__class__(self.robot_description_identifier, self.urdf)
#         return c