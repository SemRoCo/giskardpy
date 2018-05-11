import rospy
from control_msgs.msg import JointTrajectoryControllerState

from giskardpy.plugin import Plugin


class SetControlledJointsPlugin(Plugin):
    def __init__(self, controlled_joints_identifier=''):
        self.controlled_joints = []
        self.controlled_joints_identifier = controlled_joints_identifier
        super(SetControlledJointsPlugin, self).__init__()

    def start_always(self):
        msg = rospy.wait_for_message('/whole_body_controller/state', JointTrajectoryControllerState) # type: JointTrajectoryControllerState
        self.controlled_joints = msg.joint_names
        # self.controlled_joints.remove('head_pan_joint')
        # self.controlled_joints.remove('head_tilt_joint')

    def get_readings(self):
        return {self.controlled_joints_identifier: self.controlled_joints}

    def copy(self):
        c = self.__class__(self.controlled_joints_identifier)
        return c
