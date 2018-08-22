import rospy
from control_msgs.msg import JointTrajectoryControllerState

from giskardpy.plugin import Plugin


class SetControlledJointsPlugin(Plugin):
    def __init__(self, controlled_joints_identifier):
        self.controlled_joints = []
        self.controlled_joints_identifier = controlled_joints_identifier
        super(SetControlledJointsPlugin, self).__init__()

    def start_always(self):
        msg = rospy.wait_for_message('/whole_body_controller/state', JointTrajectoryControllerState) # type: JointTrajectoryControllerState
        self.controlled_joints = msg.joint_names
        self.god_map.set_data([self.controlled_joints_identifier], self.controlled_joints)

    def copy(self):
        c = self.__class__(self.controlled_joints_identifier)
        return c


class UploadRobotDescriptionPlugin(Plugin):
    def __init__(self, robot_description_identifier, param_name='robot_description'):
        self.urdf = ''
        self.param_name = param_name
        self.robot_description_identifier = robot_description_identifier
        super(UploadRobotDescriptionPlugin, self).__init__()

    def start_always(self):
        self.urdf = rospy.get_param(self.param_name)
        self.god_map.set_data([self.robot_description_identifier], self.urdf)

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