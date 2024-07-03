from py_trees import Status

from giskardpy.exceptions import PreemptedException
from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import raise_to_blackboard


class GoalCanceled(GiskardBehavior):
    @profile
    def __init__(self, action_server: ActionServerHandler):
        name = f'is \'{action_server.name}\' cancelled?'
        self.action_server = action_server
        super().__init__(name)

    @record_time
    @profile
    def update(self) -> Status:
        if (self.action_server.is_preempt_requested() and self.get_blackboard_exception() is None or
                not self.action_server.is_client_alive()):
            msg = f'\'{self.action_server.name}\' preempted'
            logging.logerr(msg)
            raise_to_blackboard(PreemptedException(msg))
        if self.get_blackboard_exception() is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE
