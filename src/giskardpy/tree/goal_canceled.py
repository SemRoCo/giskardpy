from py_trees import Status

from giskardpy.exceptions import PreemptedException
from giskardpy.tree.action_server import ActionServerBehavior
from giskardpy.utils import logging


class GoalCanceled(ActionServerBehavior):
    def update(self):
        if self.get_as().is_preempt_requested() and self.get_blackboard_exception() is None:
            logging.logerr('preempted')
            self.raise_to_blackboard(PreemptedException(''))
        if self.get_blackboard_exception() is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE