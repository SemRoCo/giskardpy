import rospy
from py_trees import Status

from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils import logging


class GoalReceived(ActionServerBehavior):
    def update(self):
        if self.get_as().has_goal():
            rospy.sleep(.5)
            logging.loginfo('Received new goal.')
            return Status.SUCCESS
        return Status.FAILURE