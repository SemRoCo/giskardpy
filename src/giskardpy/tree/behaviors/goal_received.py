from py_trees import Status

from giskardpy import identifier
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils import logging


class GoalReceived(ActionServerBehavior):
    @profile
    def update(self):
        if self.get_as().has_goal():
            logging.loginfo('Received new goal.')
            god_map.set_data(identifier.goal_msg, self.pop_goal())
            return Status.SUCCESS
        return Status.FAILURE
