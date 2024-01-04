from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils import logging


class GoalReceived(ActionServerBehavior):
    @profile
    def update(self):
        if self.get_as().has_goal():
            logging.loginfo(f'Received new goal #{god_map.goal_id + 1}.')
            god_map.goal_msg = self.pop_goal()
            return Status.SUCCESS
        return Status.FAILURE
