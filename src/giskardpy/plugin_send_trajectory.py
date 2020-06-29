import py_trees
import rospy
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState, \
    FollowJointTrajectoryResult
from py_trees_ros.actions import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import giskardpy.identifier as identifier
from giskardpy.logging import loginfo
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import traj_to_msg


class SendTrajectory(ActionClient, GiskardBehavior):
    error_code_to_str = {value: name for name, value in vars(FollowJointTrajectoryResult).items() if
                         isinstance(value, int)}

    def __init__(self, name, action_namespace=u'/whole_body_controller/follow_joint_trajectory'):
        GiskardBehavior.__init__(self, name)
        loginfo(u'waiting for action server \'{}\' to appear'.format(action_namespace))
        ActionClient.__init__(self, name, FollowJointTrajectoryAction, None, action_namespace)
        loginfo(u'successfully connected to action server')
        self.fill_velocity_values = self.get_god_map().safe_get_data(identifier.fill_velocity_values)

    def setup(self, timeout):
        # TODO get this from god map
        # self.controller_joints = rospy.wait_for_message(u'/whole_body_controller/state',
        #                                                 JointTrajectoryControllerState).joint_names
        return super(SendTrajectory, self).setup(timeout)

    def initialise(self):
        super(SendTrajectory, self).initialise()
        trajectory = self.get_god_map().safe_get_data(identifier.trajectory)
        goal = FollowJointTrajectoryGoal()
        sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        controlled_joints = self.get_robot().controlled_joints
        goal.trajectory = traj_to_msg(sample_period, trajectory, controlled_joints, self.fill_velocity_values)
        self.action_goal = goal



    def update(self):
        """
        Check only to see whether the underlying action server has
        succeeded, is running, or has cancelled/aborted for some reason and
        map these to the usual behaviour return states.

        overriding this shit because of the fucking prints
        """
        self.logger.debug("{0}.update()".format(self.__class__.__name__))
        if not self.action_client:
            self.feedback_message = "no action client, did you call setup() on your tree?"
            return py_trees.Status.INVALID
        # pity there is no 'is_connected' api like there is for c++
        if not self.sent_goal:
            self.action_client.send_goal(self.action_goal)
            self.sent_goal = True
            self.feedback_message = "sent goal to the action server"
            return py_trees.Status.RUNNING
        if self.action_client.get_state() == GoalStatus.ABORTED:
            result = self.action_client.get_result()
            self.feedback_message = self.error_code_to_str[result.error_code]
            return py_trees.Status.FAILURE
        result = self.action_client.get_result()
        if result:
            self.feedback_message = "goal reached"
            return py_trees.Status.SUCCESS
        else:
            self.feedback_message = "moving"
            return py_trees.Status.RUNNING
