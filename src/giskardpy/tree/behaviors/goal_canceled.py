from py_trees import Status

from giskardpy.exceptions import PreemptedException
from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils import logging


class GoalCanceled(ActionServerBehavior):
    def __init__(self, name, as_name, action_type=None, feedback=None):
        super().__init__(name, as_name, action_type)
        self.feedback = feedback

    def update(self):
        if self.get_as().is_preempt_requested() and self.get_blackboard_exception() is None:
            logging.logerr('preempted')
            self.raise_to_blackboard(PreemptedException(''))
        if self.feedback:
            self.as_handler.send_feedback(self.feedback)
        if self.get_blackboard_exception() is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE