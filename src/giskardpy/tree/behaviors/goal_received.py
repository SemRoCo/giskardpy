import rospy
from py_trees import Status

from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils import logging


class GoalReceived(ActionServerBehavior):
    @profile
    def update(self):
        if self.get_as().has_goal():
            logging.loginfo('Received new goal.')
            return Status.SUCCESS
        return Status.FAILURE