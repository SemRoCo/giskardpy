from py_trees import Status

from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class SendResult(GiskardBehavior):

    def __init__(self, action_server: ActionServerHandler):
        self.action_server = action_server
        name = f'send result to \'{action_server.name}\''
        super().__init__(name)

    @record_time
    @profile
    def update(self):
        self.action_server.send_result()
        return Status.SUCCESS
