import py_trees
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryResult
from py_trees_ros.actions import ActionClient

import giskardpy.identifier as identifier
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.tree.plugin import GiskardBehavior


class SendTrajectories(GiskardBehavior):

    def __init__(self, name, ros_namespaces):
        GiskardBehavior.__init__(self, name)
        self.ros_namespaces = ros_namespaces
        self.action_clients = dict()
        self.succeeded = dict()

    def setup(self, timeout):
        for _, action_client in self.action_clients.items():
            action_client.setup(timeout)

    def initialise(self):
        for ros_namespace in self.ros_namespaces:
            action_clients = SendTrajectory(u'{}{}'.format(ros_namespace, self.name), ros_namespace=ros_namespace)
            action_clients.initialise()

    def update(self):
        for name, action_client in self.action_clients.items():
            r = action_client.update()
            if r == py_trees.Status.RUNNING:
                pass
            elif r == py_trees.Status.INVALID or py_trees.Status.FAILURE:
                return r
            else:
                self.succeeded[name] = True
        if len(self.succeeded.values()) == len(self.action_clients.values()) and all(self.succeeded.values()):
            self.succeeded = dict()
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.RUNNING


class SendTrajectory(ActionClient, GiskardBehavior):
    error_code_to_str = {value: name for name, value in vars(FollowJointTrajectoryResult).items() if
                         isinstance(value, int)}

    def __init__(self, name, ros_namespace='/', action_namespace=u'whole_body_controller/follow_joint_trajectory'):
        GiskardBehavior.__init__(self, name)
        self.ros_namespace = ros_namespace
        full_action_namespace = u'{}{}'.format(self.ros_namespace, action_namespace)
        loginfo(u'waiting for action server \'{}\' to appear'.format(full_action_namespace))
        ActionClient.__init__(self, name, FollowJointTrajectoryAction, None, full_action_namespace)
        loginfo(u'successfully connected to action server')
        self.fill_velocity_values = self.get_god_map().get_data(identifier.fill_velocity_values)

    def setup(self, timeout):
        return super(SendTrajectory, self).setup(timeout)

    def initialise(self):
        super(SendTrajectory, self).initialise()
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        goal = FollowJointTrajectoryGoal()
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        controlled_joints = self.get_god_map().get_data(identifier.controlled_joints)
        goal.trajectory = trajectory.to_msg(sample_period, controlled_joints, self.fill_velocity_values,
                                            prefix=self.ros_namespace if self.ros_namespace != '/' else None)
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
            logging.loginfo(u'Sending trajectory to robot.')
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
            logging.loginfo(u'Robot successfully executed the trajectory.')
            return py_trees.Status.SUCCESS
        else:
            self.feedback_message = "moving"
            return py_trees.Status.RUNNING
